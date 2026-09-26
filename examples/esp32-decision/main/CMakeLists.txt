idf_component_register(SRCS "main.c" "app.c" INCLUDE_DIRS "."
    REQUIRES rvdecision esp_timer esp_driver_uart)
