#include "app.h"
#include "ota.h"
#include "driver/uart.h"
#include "esp_timer.h"
#include "esp_system.h"
#include "esp_cpu.h"
#include "esp_private/esp_clk.h"
#include "driver/gpio.h"
#include "freertos/FreeRTOS.h"
#include "freertos/task.h"
#include "sdkconfig.h"
#include <stdio.h>

uint64_t rd_clock_us(void) { return (uint64_t)esp_timer_get_time(); }
const char *rd_target(void) { return CONFIG_IDF_TARGET; }
size_t rd_free_heap(void) { return esp_get_free_heap_size(); }
void rd_yield(void) { vTaskDelay(1); }
uint32_t rd_clock_cycles(void) { return esp_cpu_get_cycle_count(); }
unsigned rd_cpu_hz(void) { return (unsigned)esp_clk_cpu_freq(); }
unsigned rd_core_id(void) { return (unsigned)xPortGetCoreID(); }
bool rd_dynamic_frequency(void) {
#ifdef CONFIG_PM_ENABLE
    return true;
#else
    return false;
#endif
}
bool rd_fixed_affinity(void) {
#if defined(CONFIG_FREERTOS_UNICORE) || !defined(CONFIG_ESP_MAIN_TASK_AFFINITY_NO_AFFINITY)
    return true;
#else
    return false;
#endif
}
int rd_marker_gpio(void) { return CONFIG_RD_BENCH_GPIO; }
void rd_marker(bool active) {
    if(CONFIG_RD_BENCH_GPIO>=0) gpio_set_level(CONFIG_RD_BENCH_GPIO,active);
}
void app_main(void) {
    setvbuf(stdout, NULL, _IONBF, 0);
    /* UART0 115200 on the board's default TX/RX pins; USB-UART bridge.
       No Wi-Fi, PSRAM or external inference service required. */
    ESP_ERROR_CHECK(uart_driver_install(UART_NUM_0, 2048, 0, 0, NULL, 0));
    ESP_ERROR_CHECK(uart_set_baudrate(UART_NUM_0, 115200));
    if(CONFIG_RD_BENCH_GPIO>=0) {
        ESP_ERROR_CHECK(gpio_reset_pin(CONFIG_RD_BENCH_GPIO));
        ESP_ERROR_CHECK(gpio_set_direction(CONFIG_RD_BENCH_GPIO,GPIO_MODE_OUTPUT));
        rd_marker(false);
    }
    bool healthy = rd_app_init();
    rd_ota_boot_check(healthy); /* rolls back an unverified OTA image on failure */
    if (!healthy) return;
    uint8_t bytes[128];
    for (;;) {
        int n = uart_read_bytes(UART_NUM_0, bytes, sizeof(bytes), pdMS_TO_TICKS(50));
        for (int i = 0; i < n; ++i) {
            if (rd_ota_receiving()) rd_ota_byte(bytes[i]);
            else rd_app_byte(bytes[i]);
        }
        rd_ota_poll();
        vTaskDelay(1);
    }
}
