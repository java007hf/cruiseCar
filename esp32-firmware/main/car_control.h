#pragma once

#include <stdint.h>

#include "esp_err.h"

#ifdef __cplusplus
extern "C" {
#endif

esp_err_t car_control_init(void);
void car_control_update(uint8_t lx, uint8_t ly, uint8_t rx, uint8_t ry, uint32_t buttons);
void car_control_stop(void);
/* 设置舵机角度: index 预留多路(当前 0), angle_deg 范围 0~180 */
void car_control_servo_set(uint8_t index, uint16_t angle_deg);

#ifdef __cplusplus
}
#endif
