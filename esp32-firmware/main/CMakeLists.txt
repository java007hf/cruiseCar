idf_component_register(
    SRCS "esp_hid_gap.c" "main.c" "car_control.c"
    INCLUDE_DIRS "."
    REQUIRES esp_hid
    PRIV_REQUIRES nvs_flash bt esp_driver_gpio esp_driver_ledc esp_driver_pcnt
)

