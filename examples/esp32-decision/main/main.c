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
#if CONFIG_ESP_CONSOLE_USB_SERIAL_JTAG
#include "driver/usb_serial_jtag.h"
#include "driver/usb_serial_jtag_vfs.h"
#endif
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

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
/* Negotiated link speed. "baud <rate>" replies at the current rate, then
 * switches; the host must send "baud ok" at the new rate within 2 s or the
 * board returns to 115200 by itself, so a failed switch cannot strand it.
 * Every reset starts at 115200. */
#define RD_BAUD_DEFAULT 115200u
#define RD_BAUD_CONFIRM_US 2000000ULL
static uint32_t baud_now = RD_BAUD_DEFAULT;
static bool baud_pending;
static uint64_t baud_deadline_us;
#if !CONFIG_ESP_CONSOLE_USB_SERIAL_JTAG
static void set_baud(uint32_t rate) {
    fflush(stdout);
    uart_wait_tx_done(UART_NUM_0, pdMS_TO_TICKS(200));
    uart_set_baudrate(UART_NUM_0, rate);
    uart_flush_input(UART_NUM_0);
    baud_now = rate;
}
#endif
static void baud_command(const char *arg) {
#if CONFIG_ESP_CONSOLE_USB_SERIAL_JTAG
    (void)arg;
    printf("{\"error\":\"baud_unsupported\",\"accepted\":false}\n");
#else
    if (!strcmp(arg, "ok")) {
        if (!baud_pending) { printf("{\"error\":\"baud_not_pending\",\"accepted\":false}\n"); return; }
        baud_pending = false;
        printf("{\"baud\":%u,\"confirmed\":true}\n", (unsigned)baud_now);
        return;
    }
    static const uint32_t allowed[] = {115200u, 230400u, 460800u, 921600u};
    char *end; unsigned long rate = strtoul(arg, &end, 10);
    bool ok = end != arg && !*end && *arg != '-';
    bool listed = false;
    for (size_t i = 0; ok && i < sizeof(allowed)/sizeof(allowed[0]); ++i) listed |= rate == allowed[i];
    if (!ok || !listed) { printf("{\"error\":\"baud_rate\",\"accepted\":false}\n"); return; }
    printf("{\"baud\":%lu,\"confirm\":\"baud ok\",\"within_ms\":%u}\n", rate, (unsigned)(RD_BAUD_CONFIRM_US/1000));
    set_baud((uint32_t)rate);
    baud_pending = rate != RD_BAUD_DEFAULT;
    baud_deadline_us = (uint64_t)esp_timer_get_time() + RD_BAUD_CONFIRM_US;
#endif
}
static void baud_poll(void) {
#if !CONFIG_ESP_CONSOLE_USB_SERIAL_JTAG
    if (baud_pending && (uint64_t)esp_timer_get_time() > baud_deadline_us) {
        baud_pending = false;
        set_baud(RD_BAUD_DEFAULT);
        printf("{\"baud\":%u,\"reverted\":true}\n", (unsigned)RD_BAUD_DEFAULT);
    }
#endif
}
bool rd_app_extension(const char *line) {
    if (rd_ota_command(line)) return true;
    if (!strncmp(line, "baud ", 5)) { baud_command(line + 5); return true; }
    return false;
}
void app_main(void) {
    setvbuf(stdout, NULL, _IONBF, 0);
#if CONFIG_ESP_CONSOLE_USB_SERIAL_JTAG
    /* Boards without a USB-UART bridge: commands and replies use the chip's
       built-in USB-Serial/JTAG port (S3, C3, C6, H2, P4). */
    usb_serial_jtag_driver_config_t usb = USB_SERIAL_JTAG_DRIVER_CONFIG_DEFAULT();
    usb.rx_buffer_size = 2048;
    ESP_ERROR_CHECK(usb_serial_jtag_driver_install(&usb));
    usb_serial_jtag_vfs_use_driver();
#else
    /* UART0 115200 on the board's default TX/RX pins; USB-UART bridge.
       No Wi-Fi, PSRAM or external inference service required. */
    ESP_ERROR_CHECK(uart_driver_install(UART_NUM_0, 2048, 0, 0, NULL, 0));
    ESP_ERROR_CHECK(uart_set_baudrate(UART_NUM_0, 115200));
#endif
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
        /* Block for the first byte (waking every 50 ms so OTA idle timeouts
           still fire), then drain what is already buffered without waiting.
           Waiting for a full buffer, or delaying a tick per chunk, added up
           to ~60 ms of idle latency to every short command. */
#if CONFIG_ESP_CONSOLE_USB_SERIAL_JTAG
        int n = usb_serial_jtag_read_bytes(bytes, 1, pdMS_TO_TICKS(50));
        if (n > 0) n += usb_serial_jtag_read_bytes(bytes + 1, sizeof(bytes) - 1, 0);
#else
        int n = uart_read_bytes(UART_NUM_0, bytes, 1, pdMS_TO_TICKS(50));
        if (n > 0) {
            int more = uart_read_bytes(UART_NUM_0, bytes + 1, sizeof(bytes) - 1, 0);
            if (more > 0) n += more;
        }
#endif
        for (int i = 0; i < n; ++i) {
            if (rd_ota_receiving()) rd_ota_byte(bytes[i]);
            else rd_app_byte(bytes[i]);
        }
        rd_ota_poll();
        baud_poll();
    }
}
