#include <inttypes.h>
#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>
#include <string.h>

#include "freertos/FreeRTOS.h"
#include "freertos/task.h"

#include "driver/gpio.h"
#include "esp_bt.h"
#include "esp_bt_defs.h"
#include "esp_bt_device.h"
#include "esp_event.h"
#include "esp_gap_ble_api.h"
#include "esp_gap_bt_api.h"
#include "esp_gattc_api.h"
#include "esp_hid_common.h"
#include "esp_hidh.h"
#include "esp_hidh_gattc.h"
#include "esp_check.h"
#include "esp_log.h"
#include "esp_spp_api.h"
#include "nvs_flash.h"

#include "car_control.h"
#include "esp_hid_gap.h"

#define DEVICE_NAME "CruiseCar-ESP32"
#define LED_GPIO GPIO_NUM_2
#define SPP_SERVER_NAME "CruiseCar-SPP"
#define PACKET_SIZE 10
#define SCAN_SECONDS 8
#define RESCAN_DELAY_MS 3000
#define CONTROL_VERBOSE_LOG 0
/* 若 SPP 已"连接"但超过此时长没有任何数据到达, 判定为上一次(非正常)断连未被
   栈上报, 强制复位连接态并重启 SPP 服务, 避免卡在"假连接"(灯不闪且无法重连)。 */
#define SPP_STALE_MS 3000

static const char *TAG = "cruise_car";
static uint8_t rx_buffer[PACKET_SIZE];
static size_t rx_len;
static volatile bool spp_connected;
static volatile bool hidh_opening;
static volatile bool hidh_connected;
static volatile TickType_t led_cmd_tick;   /* tick of last received command, 0 = none */
static volatile TickType_t spp_last_data_tick; /* tick of last SPP byte received, 0 = none */
#if CONTROL_VERBOSE_LOG
static bool have_prev_buttons;
static uint32_t prev_buttons;
#endif

typedef struct {
    uint8_t lx;
    uint8_t ly;
    uint8_t rx;
    uint8_t ry;
    uint32_t buttons;
} gamepad_state_t;

static uint8_t checksum(const uint8_t *packet)
{
    uint16_t sum = 0;
    for (int i = 0; i < 9; i++) {
        sum += packet[i];
    }
    return (uint8_t)(sum & 0xFF);
}

static bool parse_spp_packet(const uint8_t *packet, gamepad_state_t *state)
{
    if (packet[0] != 0xAA || packet[1] != 0x55 || packet[2] != 0x01) {
        return false;
    }
    if (checksum(packet) != packet[9]) {
        return false;
    }
    state->lx = packet[3];
    state->ly = packet[4];
    state->rx = packet[5];
    state->ry = packet[6];
    state->buttons = packet[7] | ((uint32_t)packet[8] << 8);
    return true;
}

/* 舵机控制包: 0xAA 0x55 0x03 [index] [angle 0~180] [res] [res] [res] [res] [checksum] */
static bool parse_servo_packet(const uint8_t *packet, uint8_t *index, uint16_t *angle)
{
    if (packet[0] != 0xAA || packet[1] != 0x55 || packet[2] != 0x03) {
        return false;
    }
    if (checksum(packet) != packet[9]) {
        return false;
    }
    *index = packet[3];
    uint16_t a = packet[4];
    if (a > 180) {
        a = 180;
    }
    *angle = a;
    return true;
}

static bool parse_xiaomi_gamepad_report(const uint8_t *data, uint16_t len, gamepad_state_t *state)
{
    if (len < 20) {
        return false;
    }

    state->buttons = data[0] | ((uint32_t)data[1] << 8) | ((uint32_t)data[2] << 16);
    uint8_t hat = data[3] & 0x0F;
    state->lx = data[4];
    state->ly = data[5];
    state->rx = data[6];
    state->ry = data[7];

    if (CONTROL_VERBOSE_LOG) {
        uint8_t l2 = data[8];
        uint8_t r2 = data[9];
        uint8_t battery = data[18];
        ESP_LOGI(TAG, "xiaomi/classic buttons=0x%06" PRIx32
                 " hat=%u lx=%u ly=%u rx=%u ry=%u l2=%u r2=%u battery=%u%%",
                 state->buttons, hat, state->lx, state->ly, state->rx, state->ry, l2, r2, battery);
    }
    return true;
}

static bool parse_hid_gamepad_report(const uint8_t *data, uint16_t len, gamepad_state_t *state)
{
    if (parse_xiaomi_gamepad_report(data, len, state)) {
        return true;
    }

    if (len < 4) {
        return false;
    }

    state->lx = data[0];
    state->ly = data[1];
    state->rx = data[2];
    state->ry = data[3];
    state->buttons = 0;

    for (uint16_t i = 4; i < len && i < 8; i++) {
        state->buttons |= ((uint32_t)data[i]) << ((i - 4) * 8);
    }

    if (CONTROL_VERBOSE_LOG) {
        uint8_t hat = (len > 4) ? (data[4] & 0x0F) : 0x0F;
        ESP_LOGI(TAG, "hid guess lx=%u ly=%u rx=%u ry=%u hat=%u buttons=0x%08" PRIx32,
                 state->lx, state->ly, state->rx, state->ry, hat, state->buttons);
    }
    return true;
}

#if CONTROL_VERBOSE_LOG
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
#endif

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

static void apply_gamepad_state(const char *source, const gamepad_state_t *state)
{
    if (CONTROL_VERBOSE_LOG) {
        int throttle = 128 - state->ly;
        int steering = state->lx - 128;
        ESP_LOGI(TAG, "%s lx=%u ly=%u rx=%u ry=%u buttons=0x%08" PRIx32 " throttle=%d steering=%d",
                 source, state->lx, state->ly, state->rx, state->ry, state->buttons, throttle, steering);
    }
#if CONTROL_VERBOSE_LOG
    log_button_changes(state->buttons);
#endif
    led_cmd_tick = xTaskGetTickCount();
    car_control_update(state->lx, state->ly, state->rx, state->ry, state->buttons);
}

static void maybe_stop_after_disconnect(void)
{
    if (!spp_connected && !hidh_connected && !hidh_opening) {
        car_control_stop();
    }
}

static void feed_spp_bytes(const uint8_t *data, size_t len)
{
    spp_last_data_tick = xTaskGetTickCount();
    for (size_t i = 0; i < len; i++) {
        if (rx_len == 0 && data[i] != 0xAA) {
            continue;
        }
        if (rx_len == 1 && data[i] != 0x55) {
            rx_len = 0;
            continue;
        }
        rx_buffer[rx_len++] = data[i];
        if (rx_len == PACKET_SIZE) {
            if (checksum(rx_buffer) != rx_buffer[9]) {
                ESP_LOGW(TAG, "invalid SPP checksum");
            } else if (rx_buffer[2] == 0x01) {
                gamepad_state_t state;
                if (parse_spp_packet(rx_buffer, &state)) {
                    apply_gamepad_state("spp", &state);
                } else {
                    ESP_LOGW(TAG, "bad gamepad packet");
                }
            } else if (rx_buffer[2] == 0x03) {
                uint8_t sidx = 0;
                uint16_t sangle = 0;
                if (parse_servo_packet(rx_buffer, &sidx, &sangle)) {
                    car_control_servo_set(sidx, sangle);
                    ESP_LOGI(TAG, "spp servo idx=%u angle=%u", sidx, sangle);
                } else {
                    ESP_LOGW(TAG, "bad servo packet");
                }
            } else {
                ESP_LOGW(TAG, "unknown SPP packet type 0x%02x", rx_buffer[2]);
            }
            rx_len = 0;
        }
    }
}

static void spp_callback(esp_spp_cb_event_t event, esp_spp_cb_param_t *param)
{
    switch (event) {
    case ESP_SPP_INIT_EVT:
        ESP_LOGI(TAG, "SPP initialized");
        ESP_ERROR_CHECK(esp_bt_gap_set_device_name(DEVICE_NAME));
        ESP_ERROR_CHECK(esp_bt_gap_set_scan_mode(ESP_BT_CONNECTABLE, ESP_BT_GENERAL_DISCOVERABLE));
        ESP_ERROR_CHECK(esp_spp_start_srv(ESP_SPP_SEC_NONE, ESP_SPP_ROLE_SLAVE, 0, SPP_SERVER_NAME));
        break;
    case ESP_SPP_START_EVT:
        ESP_LOGI(TAG, "SPP server started");
        break;
    case ESP_SPP_SRV_OPEN_EVT:
        spp_connected = true;
        ESP_LOGI(TAG, "SPP client connected");
        break;
    case ESP_SPP_CLOSE_EVT:
        spp_connected = false;
        ESP_LOGI(TAG, "SPP client disconnected");
        maybe_stop_after_disconnect();
        break;
    case ESP_SPP_DATA_IND_EVT:
        feed_spp_bytes(param->data_ind.data, param->data_ind.len);
        break;
    default:
        break;
    }
}

/* 重启 SPP 服务, 让接收端在(非正常)断连后可以重新发起连接。 */
static void spp_rearm(void)
{
    ESP_LOGI(TAG, "re-arming SPP server");
    esp_spp_start_srv(ESP_SPP_SEC_NONE, ESP_SPP_ROLE_SLAVE, 0, SPP_SERVER_NAME);
}

static void hidh_callback(void *handler_args, esp_event_base_t base, int32_t id, void *event_data)
{
    (void)handler_args;
    (void)base;
    esp_hidh_event_t event = (esp_hidh_event_t)id;
    esp_hidh_event_data_t *param = (esp_hidh_event_data_t *)event_data;

    switch (event) {
    case ESP_HIDH_OPEN_EVENT:
        hidh_opening = false;
        if (param->open.status == ESP_OK) {
            hidh_connected = true;
            const uint8_t *bda = esp_hidh_dev_bda_get(param->open.dev);
            ESP_LOGI(TAG, "HID open " ESP_BD_ADDR_STR " name=%s",
                     ESP_BD_ADDR_HEX(bda), esp_hidh_dev_name_get(param->open.dev));
            esp_hidh_dev_dump(param->open.dev, stdout);
        } else {
            hidh_connected = false;
            ESP_LOGE(TAG, "HID open failed status=%s", esp_err_to_name(param->open.status));
        }
        ESP_ERROR_CHECK(esp_bt_gap_set_scan_mode(ESP_BT_CONNECTABLE, ESP_BT_GENERAL_DISCOVERABLE));
        break;

    case ESP_HIDH_INPUT_EVENT: {
        gamepad_state_t state;
        if (parse_hid_gamepad_report(param->input.data, param->input.length, &state)) {
            apply_gamepad_state("hid", &state);
        }
        break;
    }

    case ESP_HIDH_BATTERY_EVENT:
        ESP_LOGI(TAG, "HID battery level=%d%%", param->battery.level);
        break;

    case ESP_HIDH_CLOSE_EVENT:
        hidh_opening = false;
        hidh_connected = false;
        ESP_LOGW(TAG, "HID close name=%s", esp_hidh_dev_name_get(param->close.dev));
        maybe_stop_after_disconnect();
        break;

    default:
        ESP_LOGD(TAG, "HID event=%" PRId32, id);
        break;
    }
}

static void scan_and_connect_hid_task(void *arg)
{
    (void)arg;

    while (true) {
        if (hidh_opening || hidh_connected) {
            vTaskDelay(pdMS_TO_TICKS(1000));
            continue;
        }

        size_t results_len = 0;
        esp_hid_scan_result_t *results = NULL;
        esp_hid_scan_result_t *candidate = NULL;

        ESP_LOGI(TAG, "Scanning for Bluetooth HID gamepads for %d seconds...", SCAN_SECONDS);
        esp_err_t err = esp_hid_scan(SCAN_SECONDS, &results_len, &results);
        ESP_ERROR_CHECK_WITHOUT_ABORT(esp_bt_gap_set_scan_mode(ESP_BT_CONNECTABLE, ESP_BT_GENERAL_DISCOVERABLE));
        if (err != ESP_OK) {
            ESP_LOGE(TAG, "HID scan failed: %s", esp_err_to_name(err));
            vTaskDelay(pdMS_TO_TICKS(RESCAN_DELAY_MS));
            continue;
        }

        ESP_LOGI(TAG, "HID scan results=%u", (unsigned)results_len);
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
            ESP_LOGI(TAG, "opening HID name=%s addr=" ESP_BD_ADDR_STR,
                     candidate->name ? candidate->name : "",
                     ESP_BD_ADDR_HEX(candidate->bda));
            hidh_opening = true;
            esp_hidh_dev_t *dev = esp_hidh_dev_open(candidate->bda, candidate->transport, candidate->ble.addr_type);
            if (dev == NULL) {
                hidh_opening = false;
                ESP_LOGE(TAG, "failed to start opening HID device");
            }
            esp_hid_scan_results_free(results);
            continue;
        }

        ESP_LOGW(TAG, "no HID device found, rescanning...");
        esp_hid_scan_results_free(results);
        vTaskDelay(pdMS_TO_TICKS(RESCAN_DELAY_MS));
    }
}

static void start_spp_server(void)
{
    ESP_ERROR_CHECK(esp_spp_register_callback(spp_callback));

    esp_spp_cfg_t spp_cfg = BT_SPP_DEFAULT_CONFIG();
    spp_cfg.mode = ESP_SPP_MODE_CB;
    ESP_ERROR_CHECK(esp_spp_enhanced_init(&spp_cfg));
}

/* ---- BLE advertising (discovery only; communication stays on Classic BT SPP) ---- */

#if CONFIG_BT_BLE_ENABLED
static void start_ble_adv(void)
{
    esp_ble_adv_data_t adv_data = {
        .set_scan_rsp        = false,
        .include_name        = true,
        .include_txpower     = false,
        .min_interval        = 0x0020,
        .max_interval        = 0x0040,
        .appearance          = 0x0000,
        .manufacturer_len    = 0,
        .p_manufacturer_data = NULL,
        .service_data_len    = 0,
        .p_service_data      = NULL,
        .service_uuid_len    = 0,
        .p_service_uuid      = NULL,
        /* 本设备是双模(同时跑 BLE 发现 + Classic SPP), 不能声明 BR/EDR 不支持,
           否则 Android 会把设备判定为纯 LE, 导致 RFCOMM(SPP) 连接失败。 */
        .flag                = (ESP_BLE_ADV_FLAG_GEN_DISC |
                                 ESP_BLE_ADV_FLAG_DMT_CONTROLLER_SPT |
                                 ESP_BLE_ADV_FLAG_DMT_HOST_SPT),
    };

    esp_ble_gap_set_device_name(DEVICE_NAME);
    esp_ble_gap_config_adv_data(&adv_data);

    esp_ble_adv_params_t adv_params = {
        .adv_int_min        = 0x20,   /* 20   ms */
        .adv_int_max        = 0x40,   /* 40   ms */
        .adv_type           = ADV_TYPE_IND,
        .own_addr_type      = BLE_ADDR_TYPE_PUBLIC,
        .channel_map        = ADV_CHNL_ALL,
        .adv_filter_policy  = ADV_FILTER_ALLOW_SCAN_ANY_CON_ANY,
    };
    esp_ble_gap_start_advertising(&adv_params);
    ESP_LOGI(TAG, "BLE advertising started (name: %s)", DEVICE_NAME);
}
#endif

/* ---- Indicator LED (GPIO 2) ---- */
/* 未配对: 持续闪烁  |  配对后: 灭  |  收到指令: 闪一下再灭 */

static inline bool any_connected(void)
{
    return spp_connected || hidh_connected || hidh_opening;
}

static void indicator_led_task(void *arg)
{
    (void)arg;
    TickType_t blink_toggle_tick = 0;
    bool blink_on = false;

    while (true) {
        TickType_t now = xTaskGetTickCount();

        /* 假连接看门狗: 已"连接"却长时间无数据 → 视为上次非正常断连未被栈上报,
           复位连接态并重启 SPP 服务, 使接收端可以重新连接。 */
        if (spp_connected && (now - spp_last_data_tick) > pdMS_TO_TICKS(SPP_STALE_MS)) {
            ESP_LOGW(TAG, "SPP stale (no data for %ums), assuming dead link", SPP_STALE_MS);
            spp_connected = false;
            maybe_stop_after_disconnect();
            spp_rearm();
        }

        if (!any_connected()) {
            /* not paired → blink every 500 ms */
            if ((now - blink_toggle_tick) >= pdMS_TO_TICKS(500)) {
                blink_on = !blink_on;
                gpio_set_level(LED_GPIO, blink_on ? 1 : 0);
                blink_toggle_tick = now;
            }
        } else {
            /* paired → flash 80 ms on command, else off */
            if (led_cmd_tick != 0 && (now - led_cmd_tick) < pdMS_TO_TICKS(80)) {
                gpio_set_level(LED_GPIO, 1);
            } else {
                gpio_set_level(LED_GPIO, 0);
                if (led_cmd_tick != 0) {
                    led_cmd_tick = 0;
                }
            }
        }

        vTaskDelay(pdMS_TO_TICKS(20));
    }
}

static void indicator_led_init(void)
{
    gpio_config_t cfg = {
        .pin_bit_mask = (1ULL << LED_GPIO),
        .mode         = GPIO_MODE_OUTPUT,
        .pull_up_en   = GPIO_PULLUP_DISABLE,
        .pull_down_en = GPIO_PULLDOWN_DISABLE,
        .intr_type    = GPIO_INTR_DISABLE,
    };
    gpio_config(&cfg);
    gpio_set_level(LED_GPIO, 0);

    BaseType_t ok = xTaskCreate(indicator_led_task, "indicator_led", 2048, NULL, 1, NULL);
    ESP_RETURN_VOID_ON_FALSE(ok == pdPASS, TAG, "create indicator LED task");
    ESP_LOGI(TAG, "indicator LED on GPIO %d ready", LED_GPIO);
}

static void start_hid_host(void)
{
#if CONFIG_BT_BLE_ENABLED
    ESP_ERROR_CHECK(esp_ble_gattc_register_callback(esp_hidh_gattc_event_handler));
#endif

    esp_hidh_config_t config = {
        .callback = hidh_callback,
        .event_stack_size = 4096,
        .callback_arg = NULL,
    };
    ESP_ERROR_CHECK(esp_hidh_init(&config));
    xTaskCreate(scan_and_connect_hid_task, "hid_scan_connect", 6 * 1024, NULL, 2, NULL);
}

void app_main(void)
{
    esp_err_t ret = nvs_flash_init();
    if (ret == ESP_ERR_NVS_NO_FREE_PAGES || ret == ESP_ERR_NVS_NEW_VERSION_FOUND) {
        ESP_ERROR_CHECK(nvs_flash_erase());
        ret = nvs_flash_init();
    }
    ESP_ERROR_CHECK(ret);

    ESP_ERROR_CHECK(car_control_init());
    indicator_led_init();

    ESP_LOGI(TAG, "Starting ESP32 SPP + HID gamepad receiver");
    ESP_ERROR_CHECK(esp_hid_gap_init(HID_HOST_MODE));
    start_hid_host();
    start_spp_server();
#if CONFIG_BT_BLE_ENABLED
    start_ble_adv();
#endif

    const uint8_t *addr = esp_bt_dev_get_address();
    ESP_LOGI(TAG, "Bluetooth device name: %s", DEVICE_NAME);
    ESP_LOGI(TAG, "Bluetooth address: %02x:%02x:%02x:%02x:%02x:%02x",
             addr[0], addr[1], addr[2], addr[3], addr[4], addr[5]);
}
