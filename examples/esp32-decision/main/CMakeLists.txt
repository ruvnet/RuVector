idf_component_register(SRCS "main.c" "app.c" "profile.c" "sensor.c" INCLUDE_DIRS "${RD_MODEL_DIR}" "."
    REQUIRES rvdecision esp_timer esp_driver_uart esp_driver_gpio)
