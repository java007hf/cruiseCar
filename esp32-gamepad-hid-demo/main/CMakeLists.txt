idf_component_register(
    SRCS "main.c" "esp_hid_gap.c"
    INCLUDE_DIRS "."
    REQUIRES esp_hid
    PRIV_REQUIRES nvs_flash esp_driver_gpio esp_driver_ledc esp_driver_pcnt
)
