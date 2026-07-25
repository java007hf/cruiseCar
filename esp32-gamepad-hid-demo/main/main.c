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
#include "driver/pulse_cnt.h"
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
#define LEFT_ENC_A GPIO_NUM_22
#define LEFT_ENC_B GPIO_NUM_23
#define RIGHT_ENC_A GPIO_NUM_21
#define RIGHT_ENC_B GPIO_NUM_19

#define LEFT_PWM_CHANNEL LEDC_CHANNEL_0
#define RIGHT_PWM_CHANNEL LEDC_CHANNEL_1
#define MOTOR_PWM_TIMER LEDC_TIMER_0
#define MOTOR_PWM_MODE LEDC_LOW_SPEED_MODE
#define MOTOR_PWM_FREQ_HZ 20000
#define MOTOR_PWM_MAX 255
#define MOTOR_DEADZONE 18
#define MOTOR_START_PWM 18
#define MOTOR_MAX_PWM 210
#define STEERING_PERCENT 45
#define ACTIVE_BRAKE_MS 0
#define HID_INPUT_TIMEOUT_MS 1500
#define HID_VERBOSE_LOG 0
#define LEFT_MOTOR_TRIM_PERCENT 100
#define RIGHT_MOTOR_TRIM_PERCENT 100
#define ENCODER_PCNT_HIGH_LIMIT 30000
#define ENCODER_PCNT_LOW_LIMIT -30000
#define ENCODER_GLITCH_FILTER_NS 1000
#define CONTROL_LOOP_MS 50
#define ENCODER_STRAIGHT_TOLERANCE 10
#define ENCODER_BALANCE_KP 1
#define ENCODER_BALANCE_MAX_CORRECTION 20

static const char *TAG = "gamepad_hid_demo";
static volatile TickType_t last_input_tick;
static volatile TickType_t brake_until_tick;
static volatile bool motors_running;
static volatile int target_left_cmd;
static volatile int target_right_cmd;
static volatile int target_throttle;
static volatile int target_steering;
static volatile uint32_t target_buttons;
static volatile bool hidh_opening;
static volatile bool hidh_connected;
static pcnt_unit_handle_t left_encoder;
static pcnt_unit_handle_t right_encoder;

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

static int apply_motor_trim(int pwm, int trim_percent)
{
    return (pwm * trim_percent) / 100;
}

static int signed_pwm_from_target(int target, int trim_percent)
{
    return apply_motor_trim(scale_motor_pwm(target), trim_percent);
}

static int sign_of(int value)
{
    if (value > 0) {
        return 1;
    }
    if (value < 0) {
        return -1;
    }
    return 0;
}

static void motor_write(gpio_num_t pin1, gpio_num_t pin2, ledc_channel_t channel, int pwm)
{
    pwm = clamp_int(pwm, -MOTOR_PWM_MAX, MOTOR_PWM_MAX);
    gpio_set_level(STBY_PIN, 1);

    if (pwm == 0) {
        gpio_set_level(pin1, 1);
        gpio_set_level(pin2, 1);
        ledc_set_duty(MOTOR_PWM_MODE, channel, MOTOR_PWM_MAX);
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

static void motor_coast(gpio_num_t pin1, gpio_num_t pin2, ledc_channel_t channel)
{
    gpio_set_level(pin1, 0);
    gpio_set_level(pin2, 0);
    ledc_set_duty(MOTOR_PWM_MODE, channel, 0);
    ledc_update_duty(MOTOR_PWM_MODE, channel);
}

static void car_stop(void)
{
    target_left_cmd = 0;
    target_right_cmd = 0;
    target_throttle = 0;
    target_steering = 0;
    target_buttons = 0;
    brake_until_tick = 0;
    motor_coast(LEFT_AIN1, LEFT_AIN2, LEFT_PWM_CHANNEL);
    motor_coast(RIGHT_BIN1, RIGHT_BIN2, RIGHT_PWM_CHANNEL);
    motors_running = false;
}

static void encoder_init_unit(const char *name, gpio_num_t enc_a, gpio_num_t enc_b, pcnt_unit_handle_t *unit)
{
    gpio_config_t encoder_gpio_config = {
        .pin_bit_mask = (1ULL << enc_a) | (1ULL << enc_b),
        .mode = GPIO_MODE_INPUT,
        .pull_up_en = GPIO_PULLUP_ENABLE,
        .pull_down_en = GPIO_PULLDOWN_DISABLE,
        .intr_type = GPIO_INTR_DISABLE,
    };
    ESP_ERROR_CHECK(gpio_config(&encoder_gpio_config));

    pcnt_unit_config_t unit_config = {
        .low_limit = ENCODER_PCNT_LOW_LIMIT,
        .high_limit = ENCODER_PCNT_HIGH_LIMIT,
        .flags.accum_count = true,
    };
    ESP_ERROR_CHECK(pcnt_new_unit(&unit_config, unit));

    pcnt_glitch_filter_config_t filter_config = {
        .max_glitch_ns = ENCODER_GLITCH_FILTER_NS,
    };
    ESP_ERROR_CHECK(pcnt_unit_set_glitch_filter(*unit, &filter_config));

    pcnt_chan_config_t chan_a_config = {
        .edge_gpio_num = enc_a,
        .level_gpio_num = enc_b,
    };
    pcnt_channel_handle_t chan_a = NULL;
    ESP_ERROR_CHECK(pcnt_new_channel(*unit, &chan_a_config, &chan_a));

    pcnt_chan_config_t chan_b_config = {
        .edge_gpio_num = enc_b,
        .level_gpio_num = enc_a,
    };
    pcnt_channel_handle_t chan_b = NULL;
    ESP_ERROR_CHECK(pcnt_new_channel(*unit, &chan_b_config, &chan_b));

    ESP_ERROR_CHECK(pcnt_channel_set_edge_action(chan_a, PCNT_CHANNEL_EDGE_ACTION_DECREASE, PCNT_CHANNEL_EDGE_ACTION_INCREASE));
    ESP_ERROR_CHECK(pcnt_channel_set_level_action(chan_a, PCNT_CHANNEL_LEVEL_ACTION_KEEP, PCNT_CHANNEL_LEVEL_ACTION_INVERSE));
    ESP_ERROR_CHECK(pcnt_channel_set_edge_action(chan_b, PCNT_CHANNEL_EDGE_ACTION_INCREASE, PCNT_CHANNEL_EDGE_ACTION_DECREASE));
    ESP_ERROR_CHECK(pcnt_channel_set_level_action(chan_b, PCNT_CHANNEL_LEVEL_ACTION_KEEP, PCNT_CHANNEL_LEVEL_ACTION_INVERSE));

    ESP_ERROR_CHECK(pcnt_unit_add_watch_point(*unit, ENCODER_PCNT_LOW_LIMIT));
    ESP_ERROR_CHECK(pcnt_unit_add_watch_point(*unit, ENCODER_PCNT_HIGH_LIMIT));
    ESP_ERROR_CHECK(pcnt_unit_enable(*unit));
    ESP_ERROR_CHECK(pcnt_unit_clear_count(*unit));
    ESP_ERROR_CHECK(pcnt_unit_start(*unit));

    ESP_LOGI(TAG, "%s encoder pins: A=%d B=%d", name, enc_a, enc_b);
}

static void encoder_init(void)
{
    encoder_init_unit("left", LEFT_ENC_A, LEFT_ENC_B, &left_encoder);
    encoder_init_unit("right", RIGHT_ENC_A, RIGHT_ENC_B, &right_encoder);
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
    int steering = (axis_to_signed(state->lx, true) * STEERING_PERCENT) / 100;
    int left = clamp_int(throttle + steering, -127, 127);
    int right = clamp_int(throttle - steering, -127, 127);

    bool was_moving = target_left_cmd != 0 || target_right_cmd != 0;
    bool now_moving = left != 0 || right != 0;

    if (!now_moving) {
        car_stop();
        if (was_moving || HID_VERBOSE_LOG) {
            ESP_LOGI(TAG, "target throttle=0 steering=0 left=0 right=0");
        }
        return;
    }

    target_left_cmd = left;
    target_right_cmd = right;
    target_throttle = throttle;
    target_steering = steering;
    target_buttons = state->buttons;
    motors_running = true;
    if (now_moving) {
        brake_until_tick = 0;
    }

    if (HID_VERBOSE_LOG) {
        ESP_LOGI(TAG, "target throttle=%d steering=%d left=%d right=%d buttons=0x%06" PRIx32,
                 throttle, steering, left, right, state->buttons);
    }
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
#if HID_VERBOSE_LOG
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
#else
    (void)buttons;
#endif
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

    if (HID_VERBOSE_LOG) {
        ESP_LOGI(TAG, "xiaomi/classic buttons=0x%06" PRIx32
                 " hat=%u lx=%u ly=%u rx=%u ry=%u l2=%u r2=%u battery=%u%%",
                 state->buttons, hat, state->lx, state->ly, state->rx, state->ry, l2, r2, battery);
    }
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

    if (HID_VERBOSE_LOG) {
        ESP_LOGI(TAG, "guess lx=%u ly=%u rx=%u ry=%u hat=%u buttons=0x%08" PRIx32,
                 state->lx, state->ly, state->rx, state->ry, hat, state->buttons);
    }
    return true;
}

static void hidh_callback(void *handler_args, esp_event_base_t base, int32_t id, void *event_data)
{
    esp_hidh_event_t event = (esp_hidh_event_t)id;
    esp_hidh_event_data_t *param = (esp_hidh_event_data_t *)event_data;

    switch (event) {
    case ESP_HIDH_OPEN_EVENT:
        if (param->open.status == ESP_OK) {
            hidh_opening = false;
            hidh_connected = true;
            const uint8_t *bda = esp_hidh_dev_bda_get(param->open.dev);
            ESP_LOGI(TAG, "OPEN " ESP_BD_ADDR_STR " name=%s",
                     ESP_BD_ADDR_HEX(bda), esp_hidh_dev_name_get(param->open.dev));
            esp_hidh_dev_dump(param->open.dev, stdout);
        } else {
            hidh_opening = false;
            hidh_connected = false;
            ESP_LOGE(TAG, "OPEN failed status=%s", esp_err_to_name(param->open.status));
        }
        break;

    case ESP_HIDH_INPUT_EVENT: {
        if (HID_VERBOSE_LOG) {
            ESP_LOGI(TAG, "INPUT usage=%s map=%u report=%u len=%u",
                     esp_hid_usage_str(param->input.usage),
                     param->input.map_index,
                     param->input.report_id,
                     param->input.length);
            ESP_LOG_BUFFER_HEX(TAG, param->input.data, param->input.length);
        }
        gamepad_state_t state;
        if (parse_common_gamepad_guess(param->input.data, param->input.length, &state)) {
            last_input_tick = xTaskGetTickCount();
            if (HID_VERBOSE_LOG) {
                log_button_changes(state.buttons);
            }
            apply_gamepad_state(&state);
        }
        break;
    }

    case ESP_HIDH_BATTERY_EVENT:
        ESP_LOGI(TAG, "BATTERY level=%d%%", param->battery.level);
        break;

    case ESP_HIDH_CLOSE_EVENT:
        hidh_opening = false;
        hidh_connected = false;
        ESP_LOGW(TAG, "CLOSE name=%s", esp_hidh_dev_name_get(param->close.dev));
        car_stop();
        break;

    default:
        ESP_LOGD(TAG, "event=%" PRId32, id);
        break;
    }
}

static void input_watchdog_task(void *arg)
{
    while (true) {
        TickType_t now = xTaskGetTickCount();
        TickType_t elapsed_ticks = now - last_input_tick;
        if (motors_running && elapsed_ticks > pdMS_TO_TICKS(HID_INPUT_TIMEOUT_MS)) {
            ESP_LOGW(TAG, "no HID input for %d ms, stopping motors", HID_INPUT_TIMEOUT_MS);
            car_stop();
        }
        vTaskDelay(pdMS_TO_TICKS(50));
    }
}

static void motor_control_task(void *arg)
{
    int prev_left_count = 0;
    int prev_right_count = 0;
    int log_divider = 0;
    int prev_logged_left_target = 0;
    int prev_logged_right_target = 0;
    uint32_t prev_logged_buttons = 0;

    ESP_ERROR_CHECK(pcnt_unit_get_count(left_encoder, &prev_left_count));
    ESP_ERROR_CHECK(pcnt_unit_get_count(right_encoder, &prev_right_count));
    ESP_LOGI(TAG, "motor control loop started, period=%d ms", CONTROL_LOOP_MS);

    while (true) {
        vTaskDelay(pdMS_TO_TICKS(CONTROL_LOOP_MS));

        int left_count = 0;
        int right_count = 0;
        esp_err_t left_err = pcnt_unit_get_count(left_encoder, &left_count);
        esp_err_t right_err = pcnt_unit_get_count(right_encoder, &right_count);
        if (left_err != ESP_OK || right_err != ESP_OK) {
            ESP_LOGE(TAG, "encoder read failed: left=%s right=%s",
                     esp_err_to_name(left_err), esp_err_to_name(right_err));
            continue;
        }

        int left_delta = left_count - prev_left_count;
        int right_delta = right_count - prev_right_count;
        prev_left_count = left_count;
        prev_right_count = right_count;

        int left_target = target_left_cmd;
        int right_target = target_right_cmd;
        int throttle = target_throttle;
        int steering = target_steering;
        uint32_t buttons = target_buttons;
        bool should_log = ++log_divider >= 4;
        if (should_log) {
            log_divider = 0;
        }

        if (left_target != prev_logged_left_target || right_target != prev_logged_right_target) {
            ESP_LOGI(TAG, "target throttle=%d steering=%d left=%d right=%d",
                     throttle, steering, left_target, right_target);
            prev_logged_left_target = left_target;
            prev_logged_right_target = right_target;
        }

        if (buttons != prev_logged_buttons) {
            log_button_changes(buttons);
            prev_logged_buttons = buttons;
        }

        if (left_target == 0 && right_target == 0) {
            if (ACTIVE_BRAKE_MS > 0 && xTaskGetTickCount() < brake_until_tick) {
                motor_write(LEFT_AIN1, LEFT_AIN2, LEFT_PWM_CHANNEL, 0);
                motor_write(RIGHT_BIN1, RIGHT_BIN2, RIGHT_PWM_CHANNEL, 0);
            } else {
                motor_coast(LEFT_AIN1, LEFT_AIN2, LEFT_PWM_CHANNEL);
                motor_coast(RIGHT_BIN1, RIGHT_BIN2, RIGHT_PWM_CHANNEL);
            }
            motors_running = false;
            if (should_log) {
                ESP_LOGI(TAG, "control idle ticks L=%d R=%d count L=%d R=%d",
                         left_delta, right_delta, left_count, right_count);
            }
            continue;
        }

        int left_pwm = signed_pwm_from_target(left_target, LEFT_MOTOR_TRIM_PERCENT);
        int right_pwm = signed_pwm_from_target(right_target, RIGHT_MOTOR_TRIM_PERCENT);
        int correction = 0;

        bool straight_target = left_target != 0 &&
                               right_target != 0 &&
                               sign_of(left_target) == sign_of(right_target) &&
                               abs(left_target - right_target) <= ENCODER_STRAIGHT_TOLERANCE;
        if (straight_target) {
            int left_ticks = abs(left_delta);
            int right_ticks = abs(right_delta);
            correction = clamp_int((left_ticks - right_ticks) * ENCODER_BALANCE_KP,
                                   -ENCODER_BALANCE_MAX_CORRECTION,
                                   ENCODER_BALANCE_MAX_CORRECTION);

            int left_mag = clamp_int(abs(left_pwm) - correction, 0, MOTOR_PWM_MAX);
            int right_mag = clamp_int(abs(right_pwm) + correction, 0, MOTOR_PWM_MAX);
            left_pwm = sign_of(left_pwm) * left_mag;
            right_pwm = sign_of(right_pwm) * right_mag;
        }

        /*
         * The right motor is mounted as the mirror of the left motor on the
         * reference TB6612 chassis, so its electrical direction is inverted.
         */
        motor_write(LEFT_AIN1, LEFT_AIN2, LEFT_PWM_CHANNEL, left_pwm);
        motor_write(RIGHT_BIN1, RIGHT_BIN2, RIGHT_PWM_CHANNEL, -right_pwm);
        motors_running = true;

        if (should_log) {
            ESP_LOGI(TAG, "control target L=%d R=%d ticks L=%d R=%d pwm L=%d R=%d corr=%d",
                     left_target, right_target, left_delta, right_delta, left_pwm, right_pwm, correction);
        }
    }
}

static void scan_and_connect_task(void *arg)
{
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
    encoder_init();
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
    last_input_tick = xTaskGetTickCount();
    xTaskCreate(motor_control_task, "motor_control", 3 * 1024, NULL, 4, NULL);
    xTaskCreate(input_watchdog_task, "input_watchdog", 2 * 1024, NULL, 3, NULL);
    xTaskCreate(scan_and_connect_task, "scan_connect", 6 * 1024, NULL, 2, NULL);
}
