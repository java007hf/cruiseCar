#include <inttypes.h>
#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "freertos/FreeRTOS.h"
#include "freertos/task.h"

#include "driver/gpio.h"
#include "driver/ledc.h"
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

#define LEFT_PWMA GPIO_NUM_13
#define LEFT_AIN1 GPIO_NUM_14
#define LEFT_AIN2 GPIO_NUM_12
#define RIGHT_PWMB GPIO_NUM_33
#define RIGHT_BIN1 GPIO_NUM_25
#define RIGHT_BIN2 GPIO_NUM_26
#define STBY_PIN GPIO_NUM_27

#define LEFT_PWM_CHANNEL LEDC_CHANNEL_0
#define RIGHT_PWM_CHANNEL LEDC_CHANNEL_1
#define MOTOR_PWM_TIMER LEDC_TIMER_0
#define MOTOR_PWM_MODE LEDC_LOW_SPEED_MODE
#define MOTOR_PWM_FREQ_HZ 20000
#define MOTOR_PWM_MAX 255
#define MOTOR_DEADZONE 10
#define MOTOR_START_PWM 18
#define MOTOR_MAX_PWM 210

static const char *TAG = "gamepad_hid_demo";

typedef struct {
    uint8_t lx;
    uint8_t ly;
    uint8_t rx;
    uint8_t ry;
    uint32_t buttons;
} gamepad_state_t;

static int clamp_int(int value, int min_value, int max_value)
{
    if (value < min_value) {
        return min_value;
    }
    if (value > max_value) {
        return max_value;
    }
    return value;
}

static int axis_to_signed(uint8_t axis, bool invert)
{
    int value = invert ? (128 - axis) : (axis - 128);
    if (abs(value) <= MOTOR_DEADZONE) {
        return 0;
    }
    return clamp_int(value, -127, 127);
}

static int scale_motor_pwm(int value)
{
    if (value == 0) {
        return 0;
    }

    int sign = value > 0 ? 1 : -1;
    int magnitude = abs(value);
    int pwm = MOTOR_START_PWM + (magnitude * (MOTOR_MAX_PWM - MOTOR_START_PWM)) / 127;
    return sign * clamp_int(pwm, 0, MOTOR_MAX_PWM);
}

static void motor_write(gpio_num_t pin1, gpio_num_t pin2, ledc_channel_t channel, int pwm)
{
    pwm = clamp_int(pwm, -MOTOR_PWM_MAX, MOTOR_PWM_MAX);
    gpio_set_level(STBY_PIN, 1);

    if (pwm == 0) {
        gpio_set_level(pin1, 0);
        gpio_set_level(pin2, 0);
        ledc_set_duty(MOTOR_PWM_MODE, channel, 0);
    } else if (pwm > 0) {
        gpio_set_level(pin1, 1);
        gpio_set_level(pin2, 0);
        ledc_set_duty(MOTOR_PWM_MODE, channel, pwm);
    } else {
        gpio_set_level(pin1, 0);
        gpio_set_level(pin2, 1);
        ledc_set_duty(MOTOR_PWM_MODE, channel, -pwm);
    }

    ledc_update_duty(MOTOR_PWM_MODE, channel);
}

static void car_stop(void)
{
    motor_write(LEFT_AIN1, LEFT_AIN2, LEFT_PWM_CHANNEL, 0);
    motor_write(RIGHT_BIN1, RIGHT_BIN2, RIGHT_PWM_CHANNEL, 0);
}

static void motor_init(void)
{
    gpio_config_t motor_gpio_config = {
        .pin_bit_mask = (1ULL << LEFT_AIN1) | (1ULL << LEFT_AIN2) |
                        (1ULL << RIGHT_BIN1) | (1ULL << RIGHT_BIN2) |
                        (1ULL << STBY_PIN),
        .mode = GPIO_MODE_OUTPUT,
        .pull_up_en = GPIO_PULLUP_DISABLE,
        .pull_down_en = GPIO_PULLDOWN_DISABLE,
        .intr_type = GPIO_INTR_DISABLE,
    };
    ESP_ERROR_CHECK(gpio_config(&motor_gpio_config));

    ledc_timer_config_t timer_config = {
        .speed_mode = MOTOR_PWM_MODE,
        .duty_resolution = LEDC_TIMER_8_BIT,
        .timer_num = MOTOR_PWM_TIMER,
        .freq_hz = MOTOR_PWM_FREQ_HZ,
        .clk_cfg = LEDC_AUTO_CLK,
    };
    ESP_ERROR_CHECK(ledc_timer_config(&timer_config));

    ledc_channel_config_t left_channel = {
        .gpio_num = LEFT_PWMA,
        .speed_mode = MOTOR_PWM_MODE,
        .channel = LEFT_PWM_CHANNEL,
        .intr_type = LEDC_INTR_DISABLE,
        .timer_sel = MOTOR_PWM_TIMER,
        .duty = 0,
        .hpoint = 0,
    };
    ESP_ERROR_CHECK(ledc_channel_config(&left_channel));

    ledc_channel_config_t right_channel = {
        .gpio_num = RIGHT_PWMB,
        .speed_mode = MOTOR_PWM_MODE,
        .channel = RIGHT_PWM_CHANNEL,
        .intr_type = LEDC_INTR_DISABLE,
        .timer_sel = MOTOR_PWM_TIMER,
        .duty = 0,
        .hpoint = 0,
    };
    ESP_ERROR_CHECK(ledc_channel_config(&right_channel));

    car_stop();
    ESP_LOGI(TAG, "TB6612 motor pins: left PWM=%d IN1=%d IN2=%d, right PWM=%d IN1=%d IN2=%d, STBY=%d",
             LEFT_PWMA, LEFT_AIN1, LEFT_AIN2, RIGHT_PWMB, RIGHT_BIN1, RIGHT_BIN2, STBY_PIN);
}

static void apply_gamepad_state(const gamepad_state_t *state)
{
    int throttle = axis_to_signed(state->ly, true);
    int steering = axis_to_signed(state->lx, false);
    int left = clamp_int(throttle + steering, -127, 127);
    int right = clamp_int(throttle - steering, -127, 127);
    int left_pwm = scale_motor_pwm(left);
    int right_pwm = scale_motor_pwm(right);

    /*
     * The right motor is mounted as the mirror of the left motor on the
     * reference TB6612 chassis, so its electrical direction is inverted.
     */
    motor_write(LEFT_AIN1, LEFT_AIN2, LEFT_PWM_CHANNEL, left_pwm);
    motor_write(RIGHT_BIN1, RIGHT_BIN2, RIGHT_PWM_CHANNEL, -right_pwm);

    ESP_LOGI(TAG, "drive lx=%u ly=%u throttle=%d steering=%d left=%d right=%d left_pwm=%d right_pwm=%d buttons=0x%06" PRIx32,
             state->lx, state->ly, throttle, steering, left, right, left_pwm, right_pwm, state->buttons);
}

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
    uint8_t l2 = data[8];
    uint8_t r2 = data[9];
    uint8_t battery = data[18];

    ESP_LOGI(TAG, "xiaomi/classic buttons=0x%06" PRIx32
             " hat=%u lx=%u ly=%u rx=%u ry=%u l2=%u r2=%u battery=%u%%",
             state->buttons, hat, state->lx, state->ly, state->rx, state->ry, l2, r2, battery);
    return true;
}

static bool parse_common_gamepad_guess(const uint8_t *data, uint16_t len, gamepad_state_t *state)
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
    uint8_t hat = (len > 4) ? (data[4] & 0x0F) : 0x0F;
    state->buttons = 0;

    for (uint16_t i = 4; i < len && i < 8; i++) {
        state->buttons |= ((uint32_t)data[i]) << ((i - 4) * 8);
    }

    ESP_LOGI(TAG, "guess lx=%u ly=%u rx=%u ry=%u hat=%u buttons=0x%08" PRIx32,
             state->lx, state->ly, state->rx, state->ry, hat, state->buttons);
    return true;
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
        gamepad_state_t state;
        if (parse_common_gamepad_guess(param->input.data, param->input.length, &state)) {
            log_button_changes(state.buttons);
            apply_gamepad_state(&state);
        }
        break;
    }

    case ESP_HIDH_BATTERY_EVENT:
        ESP_LOGI(TAG, "BATTERY level=%d%%", param->battery.level);
        break;

    case ESP_HIDH_CLOSE_EVENT:
        ESP_LOGW(TAG, "CLOSE name=%s", esp_hidh_dev_name_get(param->close.dev));
        car_stop();
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
    motor_init();
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
