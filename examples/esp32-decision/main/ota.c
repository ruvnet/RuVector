/* Serial OTA with A/B slots and self-test gated rollback.
 *
 * Protocol (one line each, replies are single JSON lines):
 *   ota status                   -> running/next slot, image state, app hash
 *   ota begin <bytes> <sha256>   -> {"ota":"ready","block":1024} then the host
 *                                   sends exactly <bytes> raw bytes, waiting for
 *                                   {"ota_ack":<offset>} after each block
 *   (end of stream)              -> {"ota":"done",...} and a restart, or an error
 *
 * The file digest binds the transfer to the exact release artifact; esp_ota_end
 * additionally verifies the image checksum, appended SHA-256 and chip id. The
 * new slot boots in PENDING_VERIFY and is kept only if the model self-test
 * passes (see rd_ota_boot_check); otherwise the bootloader returns to the
 * previous slot. Serial OTA grants nothing beyond the ROM serial bootloader
 * already reachable on the same port. */
#include "ota.h"
#include "sdkconfig.h"
#if CONFIG_RD_OTA
#include "app.h"
#include "esp_app_desc.h"
#include "esp_ota_ops.h"
#include "esp_system.h"
#include "esp_timer.h"
#include "freertos/FreeRTOS.h"
#include "freertos/task.h"
#include "mbedtls/sha256.h"
#include <errno.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define RD_OTA_BLOCK 1024
#define RD_OTA_IDLE_US 10000000ULL

static struct {
    bool active;
    esp_ota_handle_t handle;
    const esp_partition_t *target;
    mbedtls_sha256_context sha;
    uint8_t expected[32];
    uint8_t block[RD_OTA_BLOCK];
    size_t fill, received, total;
    uint64_t last_us;
} ota;

static void ota_error(const char *code, esp_err_t err) {
    printf("{\"error\":\"%s\",\"esp_err\":\"%s\",\"accepted\":false}\n", code, esp_err_to_name(err));
}
static const char *state_name(esp_ota_img_states_t s) {
    switch (s) {
    case ESP_OTA_IMG_NEW: return "new";
    case ESP_OTA_IMG_PENDING_VERIFY: return "pending_verify";
    case ESP_OTA_IMG_VALID: return "valid";
    case ESP_OTA_IMG_INVALID: return "invalid";
    case ESP_OTA_IMG_ABORTED: return "aborted";
    default: return "undefined";
    }
}
static void hex(const uint8_t *b, size_t n) { for (size_t i = 0; i < n; ++i) printf("%02x", b[i]); }
static bool parse_hex32(const char *s, uint8_t out[32]) {
    if (strlen(s) != 64) return false;
    for (size_t i = 0; i < 32; ++i) {
        unsigned v;
        if (sscanf(s + 2*i, "%2x", &v) != 1 || !strchr("0123456789abcdefABCDEF", s[2*i]) ||
            !strchr("0123456789abcdefABCDEF", s[2*i+1])) return false;
        out[i] = (uint8_t)v;
    }
    return true;
}
static void status(void) {
    const esp_partition_t *run = esp_ota_get_running_partition();
    const esp_partition_t *next = esp_ota_get_next_update_partition(NULL);
    esp_ota_img_states_t st = ESP_OTA_IMG_UNDEFINED;
    esp_ota_get_state_partition(run, &st);
    const esp_app_desc_t *d = esp_app_get_description();
    printf("{\"ota\":\"status\",\"running\":\"%s\",\"state\":\"%s\",\"next_update\":\"%s\","
           "\"slot_bytes\":%u,\"rollback\":%s,\"app_version\":\"%s\",\"idf\":\"%s\",\"app_elf_sha256\":\"",
           run ? run->label : "none", state_name(st), next ? next->label : "none",
           next ? (unsigned)next->size : 0u,
#if CONFIG_BOOTLOADER_APP_ROLLBACK_ENABLE
           "true",
#else
           "false",
#endif
           d->version, d->idf_ver);
    hex(d->app_elf_sha256, sizeof(d->app_elf_sha256));
    printf("\"}\n");
}
static void finish_abort(const char *code, esp_err_t err) {
    esp_ota_abort(ota.handle);
    mbedtls_sha256_free(&ota.sha);
    ota.active = false;
    ota_error(code, err);
}
static void begin(char *args) {
    char *end, *digest;
    errno = 0;
    unsigned long bytes = strtoul(args, &end, 10);
    if (errno || end == args || *end != ' ' || *args == '-' || bytes == 0) { ota_error("ota_args", ESP_ERR_INVALID_ARG); return; }
    digest = end + 1;
    if (!parse_hex32(digest, ota.expected)) { ota_error("ota_args", ESP_ERR_INVALID_ARG); return; }
    ota.target = esp_ota_get_next_update_partition(NULL);
    if (!ota.target) { ota_error("ota_no_slot", ESP_ERR_NOT_FOUND); return; }
    if (bytes > ota.target->size) { ota_error("ota_too_large", ESP_ERR_INVALID_SIZE); return; }
    esp_err_t err = esp_ota_begin(ota.target, OTA_WITH_SEQUENTIAL_WRITES, &ota.handle);
    if (err != ESP_OK) { ota_error("ota_begin", err); return; }
    mbedtls_sha256_init(&ota.sha);
    mbedtls_sha256_starts(&ota.sha, 0);
    ota.total = bytes; ota.received = 0; ota.fill = 0;
    ota.last_us = (uint64_t)esp_timer_get_time();
    ota.active = true;
    printf("{\"ota\":\"ready\",\"slot\":\"%s\",\"bytes\":%lu,\"block\":%d}\n", ota.target->label, bytes, RD_OTA_BLOCK);
}
static void flush_block(void) {
    esp_err_t err = esp_ota_write(ota.handle, ota.block, ota.fill);
    if (err != ESP_OK) { finish_abort("ota_write", err); return; }
    mbedtls_sha256_update(&ota.sha, ota.block, ota.fill);
    ota.received += ota.fill; ota.fill = 0;
    if (ota.received < ota.total) { printf("{\"ota_ack\":%u}\n", (unsigned)ota.received); return; }
    uint8_t got[32];
    mbedtls_sha256_finish(&ota.sha, got);
    mbedtls_sha256_free(&ota.sha);
    if (memcmp(got, ota.expected, sizeof(got))) {
        esp_ota_abort(ota.handle); ota.active = false;
        printf("{\"error\":\"ota_digest\",\"sha256\":\""); hex(got, 32); printf("\",\"accepted\":false}\n");
        return;
    }
    ota.active = false;
    err = esp_ota_end(ota.handle); /* verifies checksum, appended hash and chip id */
    if (err != ESP_OK) { ota_error("ota_verify", err); return; }
    err = esp_ota_set_boot_partition(ota.target);
    if (err != ESP_OK) { ota_error("ota_set_boot", err); return; }
    printf("{\"ota\":\"done\",\"slot\":\"%s\",\"bytes\":%u,\"sha256\":\"", ota.target->label, (unsigned)ota.total);
    hex(got, 32);
    printf("\",\"restart\":true}\n");
    fflush(stdout);
    vTaskDelay(pdMS_TO_TICKS(200));
    esp_restart();
}

bool rd_app_extension(const char *line) {
    if (!strcmp(line, "ota status")) { status(); return true; }
    if (!strncmp(line, "ota begin ", 10)) {
        char args[96];
        if (strlen(line + 10) >= sizeof(args)) { ota_error("ota_args", ESP_ERR_INVALID_ARG); return true; }
        strcpy(args, line + 10);
        begin(args);
        return true;
    }
    return false;
}
bool rd_ota_receiving(void) { return ota.active; }
void rd_ota_byte(unsigned char byte) {
    ota.block[ota.fill++] = byte;
    ota.last_us = (uint64_t)esp_timer_get_time();
    if (ota.fill == RD_OTA_BLOCK || ota.received + ota.fill == ota.total) flush_block();
}
void rd_ota_poll(void) {
    if (ota.active && (uint64_t)esp_timer_get_time() - ota.last_us > RD_OTA_IDLE_US)
        finish_abort("ota_timeout", ESP_ERR_TIMEOUT);
}
void rd_ota_boot_check(bool healthy) {
    const esp_partition_t *run = esp_ota_get_running_partition();
    esp_ota_img_states_t st;
    if (esp_ota_get_state_partition(run, &st) != ESP_OK || st != ESP_OTA_IMG_PENDING_VERIFY) return;
    if (healthy) {
        esp_ota_mark_app_valid_cancel_rollback();
        printf("{\"ota\":\"committed\",\"slot\":\"%s\"}\n", run->label);
    } else {
        printf("{\"ota\":\"rollback\",\"slot\":\"%s\",\"reason\":\"selftest\"}\n", run->label);
        fflush(stdout);
        esp_ota_mark_app_invalid_rollback_and_reboot();
    }
}
#else
bool rd_ota_receiving(void) { return false; }
void rd_ota_byte(unsigned char byte) { (void)byte; }
void rd_ota_poll(void) {}
void rd_ota_boot_check(bool healthy) { (void)healthy; }
#endif
