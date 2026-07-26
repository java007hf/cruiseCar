idf_component_register(
    SRCS "main.c" "car_control.c"
    INCLUDE_DIRS "."
    PRIV_REQUIRES nvs_flash bt esp_driver_gpio esp_driver_ledc esp_driver_pcnt
)

