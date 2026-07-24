#include <inttypes.h>
#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>
#include <string.h>

#include "freertos/FreeRTOS.h"
#include "freertos/task.h"

#include "esp_bt.h"
#include "esp_bt_defs.h"
#include "esp_bt_device.h"
#include "esp_event.h"
#include "esp_gap_ble_api.h"
#include "esp_gattc_api.h"
#include "esp_hid_common.h"
#include "esp_hidh.h"
#include "esp_hidh_gattc.h"
#include "esp_log.h"
#include "nvs_flash.h"

#include "esp_hid_gap.h"

#define SCAN_SECONDS 8
#define RESCAN_DELAY_MS 3000

static const char *TAG = "gamepad_hid_demo";

static const char *button_name(int bit)
{
    switch (bit) {
    case 0:
        return "A";
    case 1:
        return "B";
    case 3:
        return "X";
    case 4:
        return "Y";
    case 6:
        return "L1";
    case 7:
        return "R1";
    case 8:
        return "L2";
    case 9:
        return "R2";
    default:
        return NULL;
    }
}

static void log_button_changes(uint32_t buttons)
{
    static bool have_prev_buttons;
    static uint32_t prev_buttons;
    uint32_t changed = have_prev_buttons ? (buttons ^ prev_buttons) : buttons;

    for (int bit = 0; bit < 24; bit++) {
        uint32_t mask = 1UL << bit;
        if ((changed & mask) == 0) {
            continue;
        }

        const char *name = button_name(bit);
        if (name) {
            ESP_LOGI(TAG, "button %s %s", name, (buttons & mask) ? "pressed" : "released");
        } else {
            ESP_LOGI(TAG, "button B%02d %s", bit, (buttons & mask) ? "pressed" : "released");
        }
    }

    prev_buttons = buttons;
    have_prev_buttons = true;
}

static bool looks_like_gamepad(esp_hid_scan_result_t *result)
{
    if (result->usage == ESP_HID_USAGE_GAMEPAD || result->usage == ESP_HID_USAGE_JOYSTICK) {
        return true;
    }

    if (result->name == NULL) {
        return false;
    }

    const char *name = result->name;
    return strstr(name, "Gamepad") || strstr(name, "GAMEPAD") ||
           strstr(name, "Joystick") || strstr(name, "JOYSTICK") ||
           strstr(name, "Xbox") || strstr(name, "XBOX") ||
           strstr(name, "DualShock") || strstr(name, "DualSense");
}

static bool log_xiaomi_gamepad_report(const uint8_t *data, uint16_t len)
{
    if (len < 20) {
        return false;
    }

    uint32_t buttons = data[0] | ((uint32_t)data[1] << 8) | ((uint32_t)data[2] << 16);
    uint8_t hat = data[3] & 0x0F;
    uint8_t lx = data[4];
    uint8_t ly = data[5];
    uint8_t rx = data[6];
    uint8_t ry = data[7];
    uint8_t l2 = data[8];
    uint8_t r2 = data[9];
    uint8_t battery = data[18];

    ESP_LOGI(TAG, "xiaomi/classic buttons=0x%06" PRIx32
             " hat=%u lx=%u ly=%u rx=%u ry=%u l2=%u r2=%u battery=%u%%",
             buttons, hat, lx, ly, rx, ry, l2, r2, battery);
    log_button_changes(buttons);
    return true;
}

static void log_common_gamepad_guess(const uint8_t *data, uint16_t len)
{
    if (log_xiaomi_gamepad_report(data, len)) {
        return;
    }

    if (len < 4) {
        return;
    }

    uint8_t lx = data[0];
    uint8_t ly = data[1];
    uint8_t rx = data[2];
    uint8_t ry = data[3];
    uint8_t hat = (len > 4) ? (data[4] & 0x0F) : 0x0F;
    uint32_t buttons = 0;

    for (uint16_t i = 4; i < len && i < 8; i++) {
        buttons |= ((uint32_t)data[i]) << ((i - 4) * 8);
    }

    ESP_LOGI(TAG, "guess lx=%u ly=%u rx=%u ry=%u hat=%u buttons=0x%08" PRIx32,
             lx, ly, rx, ry, hat, buttons);
}

static void hidh_callback(void *handler_args, esp_event_base_t base, int32_t id, void *event_data)
{
    esp_hidh_event_t event = (esp_hidh_event_t)id;
    esp_hidh_event_data_t *param = (esp_hidh_event_data_t *)event_data;

    switch (event) {
    case ESP_HIDH_OPEN_EVENT:
        if (param->open.status == ESP_OK) {
            const uint8_t *bda = esp_hidh_dev_bda_get(param->open.dev);
            ESP_LOGI(TAG, "OPEN " ESP_BD_ADDR_STR " name=%s",
                     ESP_BD_ADDR_HEX(bda), esp_hidh_dev_name_get(param->open.dev));
            esp_hidh_dev_dump(param->open.dev, stdout);
        } else {
            ESP_LOGE(TAG, "OPEN failed status=%s", esp_err_to_name(param->open.status));
        }
        break;

    case ESP_HIDH_INPUT_EVENT: {
        ESP_LOGI(TAG, "INPUT usage=%s map=%u report=%u len=%u",
                 esp_hid_usage_str(param->input.usage),
                 param->input.map_index,
                 param->input.report_id,
                 param->input.length);
        ESP_LOG_BUFFER_HEX(TAG, param->input.data, param->input.length);
        log_common_gamepad_guess(param->input.data, param->input.length);
        break;
    }

    case ESP_HIDH_BATTERY_EVENT:
        ESP_LOGI(TAG, "BATTERY level=%d%%", param->battery.level);
        break;

    case ESP_HIDH_CLOSE_EVENT:
        ESP_LOGW(TAG, "CLOSE name=%s", esp_hidh_dev_name_get(param->close.dev));
        break;

    default:
        ESP_LOGD(TAG, "event=%" PRId32, id);
        break;
    }
}

static void scan_and_connect_task(void *arg)
{
    while (true) {
        size_t results_len = 0;
        esp_hid_scan_result_t *results = NULL;
        esp_hid_scan_result_t *candidate = NULL;

        ESP_LOGI(TAG, "Scanning for Bluetooth HID gamepads for %d seconds...", SCAN_SECONDS);
        esp_err_t err = esp_hid_scan(SCAN_SECONDS, &results_len, &results);
        if (err != ESP_OK) {
            ESP_LOGE(TAG, "scan failed: %s", esp_err_to_name(err));
            vTaskDelay(pdMS_TO_TICKS(RESCAN_DELAY_MS));
            continue;
        }

        ESP_LOGI(TAG, "scan results=%u", (unsigned)results_len);
        for (esp_hid_scan_result_t *r = results; r; r = r->next) {
            ESP_LOGI(TAG, "found transport=%s addr=" ESP_BD_ADDR_STR " rssi=%d usage=%s name=%s",
                     r->transport == ESP_HID_TRANSPORT_BLE ? "BLE" : "BT",
                     ESP_BD_ADDR_HEX(r->bda),
                     r->rssi,
                     esp_hid_usage_str(r->usage),
                     r->name ? r->name : "");

            if (candidate == NULL && looks_like_gamepad(r)) {
                candidate = r;
            }
        }

        if (candidate == NULL) {
            candidate = results;
        }

        if (candidate) {
            ESP_LOGI(TAG, "opening name=%s addr=" ESP_BD_ADDR_STR,
                     candidate->name ? candidate->name : "",
                     ESP_BD_ADDR_HEX(candidate->bda));
            esp_hidh_dev_t *dev = esp_hidh_dev_open(candidate->bda, candidate->transport, candidate->ble.addr_type);
            if (dev == NULL) {
                ESP_LOGE(TAG, "failed to start opening HID device");
            }
            esp_hid_scan_results_free(results);
            vTaskDelete(NULL);
        }

        ESP_LOGW(TAG, "no HID device found, rescanning...");
        esp_hid_scan_results_free(results);
        vTaskDelay(pdMS_TO_TICKS(RESCAN_DELAY_MS));
    }
}

void app_main(void)
{
    esp_err_t ret = nvs_flash_init();
    if (ret == ESP_ERR_NVS_NO_FREE_PAGES || ret == ESP_ERR_NVS_NEW_VERSION_FOUND) {
        ESP_ERROR_CHECK(nvs_flash_erase());
        ret = nvs_flash_init();
    }
    ESP_ERROR_CHECK(ret);

    ESP_LOGI(TAG, "Starting ESP32 Bluetooth HID gamepad receiver demo");
    ESP_ERROR_CHECK(esp_hid_gap_init(HID_HOST_MODE));

#if CONFIG_BT_BLE_ENABLED
    ESP_ERROR_CHECK(esp_ble_gattc_register_callback(esp_hidh_gattc_event_handler));
#endif

    esp_hidh_config_t config = {
        .callback = hidh_callback,
        .event_stack_size = 4096,
        .callback_arg = NULL,
    };
    ESP_ERROR_CHECK(esp_hidh_init(&config));

    const uint8_t *addr = esp_bt_dev_get_address();
    ESP_LOGI(TAG, "ESP32 Bluetooth addr=" ESP_BD_ADDR_STR, ESP_BD_ADDR_HEX(addr));
    xTaskCreate(scan_and_connect_task, "scan_connect", 6 * 1024, NULL, 2, NULL);
}
