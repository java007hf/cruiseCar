#include "car_control.h"

#include <inttypes.h>
#include <stdbool.h>
#include <stdlib.h>

#include "driver/gpio.h"
#include "driver/ledc.h"
#include "driver/pulse_cnt.h"
#include "esp_check.h"
#include "esp_log.h"
#include "freertos/FreeRTOS.h"
#include "freertos/task.h"

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
#define LEFT_MOTOR_TRIM_PERCENT 100
#define RIGHT_MOTOR_TRIM_PERCENT 100
#define ENCODER_PCNT_HIGH_LIMIT 30000
#define ENCODER_PCNT_LOW_LIMIT -30000
#define ENCODER_GLITCH_FILTER_NS 1000
#define CONTROL_LOOP_MS 50
/* 指令超时保护(failsafe): 超过该时间未收到新控制指令, 自动停车,
   防止遥控端停发/断流后小车保持最后一条指令一直行驶。 */
#define CONTROL_TIMEOUT_MS 500

/* ---- 舵机(Servo) ----
   注意: 经典 ESP32 的 GPIO 34/35/36/39 为输入专用, 不能输出 PWM。
   默认使用 GPIO 18(可输出且空闲); 改线后只需改 SERVO_GPIO。 */
#define SERVO_GPIO GPIO_NUM_18
#define SERVO_PWM_CHANNEL LEDC_CHANNEL_2
#define SERVO_PWM_TIMER LEDC_TIMER_1
#define SERVO_PWM_MODE LEDC_LOW_SPEED_MODE
#define SERVO_PWM_FREQ_HZ 50
#define SERVO_RES_BITS 16          /* 16-bit 占空比分辨率 */
#define SERVO_MIN_US 500           /* 0°  对应脉宽 */
#define SERVO_MAX_US 2500          /* 180° 对应脉宽 */
#define SERVO_PERIOD_US 20000      /* 50Hz 周期 */
#define SERVO_DEFAULT_ANGLE 90
#define ENCODER_STRAIGHT_TOLERANCE 10
#define ENCODER_BALANCE_KP 1
#define ENCODER_BALANCE_MAX_CORRECTION 20

static const char *TAG = "car_control";
static volatile TickType_t brake_until_tick;
static volatile bool motors_running;
static volatile int target_left_cmd;
static volatile int target_right_cmd;
static volatile int target_throttle;
static volatile int target_steering;
static volatile uint32_t target_buttons;
static volatile TickType_t last_cmd_tick;
static pcnt_unit_handle_t left_encoder;
static pcnt_unit_handle_t right_encoder;

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

void car_control_stop(void)
{
    target_left_cmd = 0;
    target_right_cmd = 0;
    target_throttle = 0;
    target_steering = 0;
    target_buttons = 0;
    brake_until_tick = 0;
    last_cmd_tick = 0;
    motor_coast(LEFT_AIN1, LEFT_AIN2, LEFT_PWM_CHANNEL);
    motor_coast(RIGHT_BIN1, RIGHT_BIN2, RIGHT_PWM_CHANNEL);
    motors_running = false;
}

static esp_err_t encoder_init_unit(const char *name, gpio_num_t enc_a, gpio_num_t enc_b, pcnt_unit_handle_t *unit)
{
    gpio_config_t encoder_gpio_config = {
        .pin_bit_mask = (1ULL << enc_a) | (1ULL << enc_b),
        .mode = GPIO_MODE_INPUT,
        .pull_up_en = GPIO_PULLUP_ENABLE,
        .pull_down_en = GPIO_PULLDOWN_DISABLE,
        .intr_type = GPIO_INTR_DISABLE,
    };
    ESP_RETURN_ON_ERROR(gpio_config(&encoder_gpio_config), TAG, "configure %s encoder GPIO", name);

    pcnt_unit_config_t unit_config = {
        .low_limit = ENCODER_PCNT_LOW_LIMIT,
        .high_limit = ENCODER_PCNT_HIGH_LIMIT,
        .flags.accum_count = true,
    };
    ESP_RETURN_ON_ERROR(pcnt_new_unit(&unit_config, unit), TAG, "new %s PCNT unit", name);

    pcnt_glitch_filter_config_t filter_config = {
        .max_glitch_ns = ENCODER_GLITCH_FILTER_NS,
    };
    ESP_RETURN_ON_ERROR(pcnt_unit_set_glitch_filter(*unit, &filter_config), TAG, "set %s PCNT filter", name);

    pcnt_chan_config_t chan_a_config = {
        .edge_gpio_num = enc_a,
        .level_gpio_num = enc_b,
    };
    pcnt_channel_handle_t chan_a = NULL;
    ESP_RETURN_ON_ERROR(pcnt_new_channel(*unit, &chan_a_config, &chan_a), TAG, "new %s PCNT channel A", name);

    pcnt_chan_config_t chan_b_config = {
        .edge_gpio_num = enc_b,
        .level_gpio_num = enc_a,
    };
    pcnt_channel_handle_t chan_b = NULL;
    ESP_RETURN_ON_ERROR(pcnt_new_channel(*unit, &chan_b_config, &chan_b), TAG, "new %s PCNT channel B", name);

    ESP_RETURN_ON_ERROR(pcnt_channel_set_edge_action(chan_a, PCNT_CHANNEL_EDGE_ACTION_DECREASE, PCNT_CHANNEL_EDGE_ACTION_INCREASE), TAG, "set %s channel A edge action", name);
    ESP_RETURN_ON_ERROR(pcnt_channel_set_level_action(chan_a, PCNT_CHANNEL_LEVEL_ACTION_KEEP, PCNT_CHANNEL_LEVEL_ACTION_INVERSE), TAG, "set %s channel A level action", name);
    ESP_RETURN_ON_ERROR(pcnt_channel_set_edge_action(chan_b, PCNT_CHANNEL_EDGE_ACTION_INCREASE, PCNT_CHANNEL_EDGE_ACTION_DECREASE), TAG, "set %s channel B edge action", name);
    ESP_RETURN_ON_ERROR(pcnt_channel_set_level_action(chan_b, PCNT_CHANNEL_LEVEL_ACTION_KEEP, PCNT_CHANNEL_LEVEL_ACTION_INVERSE), TAG, "set %s channel B level action", name);

    ESP_RETURN_ON_ERROR(pcnt_unit_add_watch_point(*unit, ENCODER_PCNT_LOW_LIMIT), TAG, "add %s PCNT low watch point", name);
    ESP_RETURN_ON_ERROR(pcnt_unit_add_watch_point(*unit, ENCODER_PCNT_HIGH_LIMIT), TAG, "add %s PCNT high watch point", name);
    ESP_RETURN_ON_ERROR(pcnt_unit_enable(*unit), TAG, "enable %s PCNT", name);
    ESP_RETURN_ON_ERROR(pcnt_unit_clear_count(*unit), TAG, "clear %s PCNT", name);
    ESP_RETURN_ON_ERROR(pcnt_unit_start(*unit), TAG, "start %s PCNT", name);

    ESP_LOGI(TAG, "%s encoder pins: A=%d B=%d", name, enc_a, enc_b);
    return ESP_OK;
}

static esp_err_t encoder_init(void)
{
    ESP_RETURN_ON_ERROR(encoder_init_unit("left", LEFT_ENC_A, LEFT_ENC_B, &left_encoder), TAG, "init left encoder");
    ESP_RETURN_ON_ERROR(encoder_init_unit("right", RIGHT_ENC_A, RIGHT_ENC_B, &right_encoder), TAG, "init right encoder");
    return ESP_OK;
}

static esp_err_t motor_init(void)
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
    ESP_RETURN_ON_ERROR(gpio_config(&motor_gpio_config), TAG, "configure motor GPIO");

    ledc_timer_config_t timer_config = {
        .speed_mode = MOTOR_PWM_MODE,
        .duty_resolution = LEDC_TIMER_8_BIT,
        .timer_num = MOTOR_PWM_TIMER,
        .freq_hz = MOTOR_PWM_FREQ_HZ,
        .clk_cfg = LEDC_AUTO_CLK,
    };
    ESP_RETURN_ON_ERROR(ledc_timer_config(&timer_config), TAG, "configure motor PWM timer");

    ledc_channel_config_t left_channel = {
        .gpio_num = LEFT_PWMA,
        .speed_mode = MOTOR_PWM_MODE,
        .channel = LEFT_PWM_CHANNEL,
        .intr_type = LEDC_INTR_DISABLE,
        .timer_sel = MOTOR_PWM_TIMER,
        .duty = 0,
        .hpoint = 0,
    };
    ESP_RETURN_ON_ERROR(ledc_channel_config(&left_channel), TAG, "configure left PWM channel");

    ledc_channel_config_t right_channel = {
        .gpio_num = RIGHT_PWMB,
        .speed_mode = MOTOR_PWM_MODE,
        .channel = RIGHT_PWM_CHANNEL,
        .intr_type = LEDC_INTR_DISABLE,
        .timer_sel = MOTOR_PWM_TIMER,
        .duty = 0,
        .hpoint = 0,
    };
    ESP_RETURN_ON_ERROR(ledc_channel_config(&right_channel), TAG, "configure right PWM channel");

    car_control_stop();
    ESP_LOGI(TAG, "TB6612 motor pins: left PWM=%d IN1=%d IN2=%d, right PWM=%d IN1=%d IN2=%d, STBY=%d",
             LEFT_PWMA, LEFT_AIN1, LEFT_AIN2, RIGHT_PWMB, RIGHT_BIN1, RIGHT_BIN2, STBY_PIN);
    return ESP_OK;
}

static esp_err_t servo_init(void)
{
    ledc_timer_config_t timer_config = {
        .speed_mode = SERVO_PWM_MODE,
        .duty_resolution = SERVO_RES_BITS,
        .timer_num = SERVO_PWM_TIMER,
        .freq_hz = SERVO_PWM_FREQ_HZ,
        .clk_cfg = LEDC_AUTO_CLK,
    };
    ESP_RETURN_ON_ERROR(ledc_timer_config(&timer_config), TAG, "configure servo PWM timer");

    ledc_channel_config_t channel = {
        .gpio_num = SERVO_GPIO,
        .speed_mode = SERVO_PWM_MODE,
        .channel = SERVO_PWM_CHANNEL,
        .intr_type = LEDC_INTR_DISABLE,
        .timer_sel = SERVO_PWM_TIMER,
        .duty = 0,
        .hpoint = 0,
    };
    ESP_RETURN_ON_ERROR(ledc_channel_config(&channel), TAG, "configure servo PWM channel");

    ESP_LOGI(TAG, "servo on GPIO %d ready (50Hz, %d-bit)", SERVO_GPIO, SERVO_RES_BITS);
    return ESP_OK;
}

void car_control_servo_set(uint8_t index, uint16_t angle_deg)
{
    (void)index; /* 当前仅一路舵机(索引 0); 预留多路扩展 */
    if (angle_deg > 180) {
        angle_deg = 180;
    }
    uint32_t pulse_us = SERVO_MIN_US +
                        (uint32_t)(angle_deg * (SERVO_MAX_US - SERVO_MIN_US)) / 180;
    uint32_t max_duty = (1u << SERVO_RES_BITS) - 1;
    uint32_t duty = (pulse_us * max_duty) / SERVO_PERIOD_US;

    ledc_set_duty(SERVO_PWM_MODE, SERVO_PWM_CHANNEL, duty);
    ledc_update_duty(SERVO_PWM_MODE, SERVO_PWM_CHANNEL);
}

void car_control_update(uint8_t lx, uint8_t ly, uint8_t rx, uint8_t ry, uint32_t buttons)
{
    (void)rx;
    (void)ry;

    last_cmd_tick = xTaskGetTickCount();
    int throttle = axis_to_signed(ly, true);
    int steering = (axis_to_signed(lx, true) * STEERING_PERCENT) / 100;
    int left = clamp_int(throttle + steering, -127, 127);
    int right = clamp_int(throttle - steering, -127, 127);

    bool was_moving = target_left_cmd != 0 || target_right_cmd != 0;
    bool now_moving = left != 0 || right != 0;

    if (!now_moving) {
        car_control_stop();
        if (was_moving) {
            ESP_LOGI(TAG, "target throttle=0 steering=0 left=0 right=0");
        }
        return;
    }

    target_left_cmd = left;
    target_right_cmd = right;
    target_throttle = throttle;
    target_steering = steering;
    target_buttons = buttons;
    motors_running = true;
    brake_until_tick = 0;
}

static void motor_control_task(void *arg)
{
    (void)arg;
    int prev_left_count = 0;
    int prev_right_count = 0;
    int log_divider = 0;
    int prev_logged_left_target = 0;
    int prev_logged_right_target = 0;

    ESP_ERROR_CHECK(pcnt_unit_get_count(left_encoder, &prev_left_count));
    ESP_ERROR_CHECK(pcnt_unit_get_count(right_encoder, &prev_right_count));
    ESP_LOGI(TAG, "motor control loop started, period=%d ms", CONTROL_LOOP_MS);

    while (true) {
        vTaskDelay(pdMS_TO_TICKS(CONTROL_LOOP_MS));

        /* 指令超时保护: 持续运动但很久没收到新指令, 立即停车 */
        if ((target_left_cmd != 0 || target_right_cmd != 0) &&
            (xTaskGetTickCount() - last_cmd_tick) > pdMS_TO_TICKS(CONTROL_TIMEOUT_MS)) {
            ESP_LOGW(TAG, "command timeout %" PRIu32 " ms, failsafe stop",
                     (uint32_t)(xTaskGetTickCount() - last_cmd_tick));
            car_control_stop();
            continue;
        }

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

        motor_write(LEFT_AIN1, LEFT_AIN2, LEFT_PWM_CHANNEL, left_pwm);
        motor_write(RIGHT_BIN1, RIGHT_BIN2, RIGHT_PWM_CHANNEL, -right_pwm);
        motors_running = true;

        if (should_log) {
            ESP_LOGI(TAG, "control target L=%d R=%d ticks L=%d R=%d pwm L=%d R=%d corr=%d buttons=0x%06" PRIx32,
                     left_target, right_target, left_delta, right_delta, left_pwm, right_pwm, correction, target_buttons);
        }
    }
}

esp_err_t car_control_init(void)
{
    ESP_RETURN_ON_ERROR(motor_init(), TAG, "init motors");
    ESP_RETURN_ON_ERROR(encoder_init(), TAG, "init encoders");
    ESP_RETURN_ON_ERROR(servo_init(), TAG, "init servo");

    /* 上电把舵机先归到默认角度, 避免随机占位 */
    car_control_servo_set(0, SERVO_DEFAULT_ANGLE);

    BaseType_t ok = xTaskCreate(motor_control_task, "motor_control", 3 * 1024, NULL, 4, NULL);
    ESP_RETURN_ON_FALSE(ok == pdPASS, ESP_ERR_NO_MEM, TAG, "create motor control task");
    return ESP_OK;
}
