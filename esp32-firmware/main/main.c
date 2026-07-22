#include <stdint.h>
#include <stdbool.h>
#include <string.h>

#include "esp_bt.h"
#include "esp_bt_device.h"
#include "esp_bt_main.h"
#include "esp_gap_bt_api.h"
#include "esp_log.h"
#include "esp_spp_api.h"
#include "nvs_flash.h"

#define DEVICE_NAME "CruiseCar-ESP32"
#define SPP_SERVER_NAME "CruiseCar-SPP"
#define PACKET_SIZE 10

static const char *TAG = "cruise_car";
static uint8_t rx_buffer[PACKET_SIZE];
static size_t rx_len;

typedef struct {
    uint8_t lx;
    uint8_t ly;
    uint8_t rx;
    uint8_t ry;
    uint16_t buttons;
} gamepad_state_t;

static uint8_t checksum(const uint8_t *packet)
{
    uint16_t sum = 0;
    for (int i = 0; i < 9; i++) {
        sum += packet[i];
    }
    return (uint8_t)(sum & 0xFF);
}

static bool parse_packet(const uint8_t *packet, gamepad_state_t *state)
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
    state->buttons = packet[7] | ((uint16_t)packet[8] << 8);
    return true;
}

static void apply_gamepad_state(const gamepad_state_t *state)
{
    int throttle = 128 - state->ly;
    int steering = state->lx - 128;
    ESP_LOGI(TAG, "lx=%u ly=%u rx=%u ry=%u buttons=0x%04x throttle=%d steering=%d",
             state->lx, state->ly, state->rx, state->ry, state->buttons, throttle, steering);

    /*
     * TODO: map throttle/steering to the actual motor driver GPIO/PWM channels.
     * Keep this function as the single output point so SPP and future HID Host
     * inputs can share the same car-control behavior.
     */
}

static void feed_bytes(const uint8_t *data, size_t len)
{
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
            gamepad_state_t state;
            if (parse_packet(rx_buffer, &state)) {
                apply_gamepad_state(&state);
            } else {
                ESP_LOGW(TAG, "invalid control packet");
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
        esp_bt_gap_set_device_name(DEVICE_NAME);
        esp_bt_gap_set_scan_mode(ESP_BT_CONNECTABLE, ESP_BT_GENERAL_DISCOVERABLE);
        esp_spp_start_srv(ESP_SPP_SEC_NONE, ESP_SPP_ROLE_SLAVE, 0, SPP_SERVER_NAME);
        break;
    case ESP_SPP_START_EVT:
        ESP_LOGI(TAG, "SPP server started");
        break;
    case ESP_SPP_SRV_OPEN_EVT:
        ESP_LOGI(TAG, "SPP client connected");
        break;
    case ESP_SPP_CLOSE_EVT:
        ESP_LOGI(TAG, "SPP client disconnected");
        break;
    case ESP_SPP_DATA_IND_EVT:
        feed_bytes(param->data_ind.data, param->data_ind.len);
        break;
    default:
        break;
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

    ESP_ERROR_CHECK(esp_bt_controller_mem_release(ESP_BT_MODE_BLE));

    esp_bt_controller_config_t bt_cfg = BT_CONTROLLER_INIT_CONFIG_DEFAULT();
    ESP_ERROR_CHECK(esp_bt_controller_init(&bt_cfg));
    ESP_ERROR_CHECK(esp_bt_controller_enable(ESP_BT_MODE_CLASSIC_BT));
    ESP_ERROR_CHECK(esp_bluedroid_init());
    ESP_ERROR_CHECK(esp_bluedroid_enable());
    ESP_ERROR_CHECK(esp_spp_register_callback(spp_callback));

    esp_spp_cfg_t spp_cfg = BT_SPP_DEFAULT_CONFIG();
    spp_cfg.mode = ESP_SPP_MODE_CB;
    ESP_ERROR_CHECK(esp_spp_enhanced_init(&spp_cfg));

    const uint8_t *addr = esp_bt_dev_get_address();
    ESP_LOGI(TAG, "Bluetooth device name: %s", DEVICE_NAME);
    ESP_LOGI(TAG, "Bluetooth address: %02x:%02x:%02x:%02x:%02x:%02x",
             addr[0], addr[1], addr[2], addr[3], addr[4], addr[5]);
}
