#pragma once

#include <stdint.h>

#include "esp_err.h"

#ifdef __cplusplus
extern "C" {
#endif

esp_err_t car_control_init(void);
void car_control_update(uint8_t lx, uint8_t ly, uint8_t rx, uint8_t ry, uint32_t buttons);
void car_control_stop(void);

#ifdef __cplusplus
}
#endif
