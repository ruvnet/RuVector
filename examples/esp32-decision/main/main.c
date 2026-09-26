#include "app.h"
#include "driver/uart.h"
#include "esp_timer.h"
#include "esp_system.h"
#include "freertos/FreeRTOS.h"
#include "freertos/task.h"
#include "sdkconfig.h"
#include <stdio.h>

uint64_t rd_clock_us(void) { return (uint64_t)esp_timer_get_time(); }
const char *rd_target(void) { return CONFIG_IDF_TARGET; }
size_t rd_free_heap(void) { return esp_get_free_heap_size(); }
void rd_yield(void) { vTaskDelay(1); }
void app_main(void) {
    setvbuf(stdout, NULL, _IONBF, 0);
    /* UART0 115200 on the board's default TX/RX pins; USB-UART bridge.
       No Wi-Fi, PSRAM or external inference service required. */
    ESP_ERROR_CHECK(uart_driver_install(UART_NUM_0, 2048, 0, 0, NULL, 0));
    ESP_ERROR_CHECK(uart_set_baudrate(UART_NUM_0, 115200));
    if (!rd_app_init()) return;
    uint8_t bytes[128];
    for (;;) {
        int n = uart_read_bytes(UART_NUM_0, bytes, sizeof(bytes), pdMS_TO_TICKS(50));
        for (int i = 0; i < n; ++i) rd_app_byte(bytes[i]);
        vTaskDelay(1);
    }
}
