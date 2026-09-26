#ifndef RD_OTA_H
#define RD_OTA_H
#include <stdbool.h>
/* Serial OTA transport; see ota.c for the protocol. Without CONFIG_RD_OTA the
 * stubs keep the UART loop unchanged. */
/* Handles "ota status" / "ota begin"; returns false for other lines. */
bool rd_ota_command(const char *line);
bool rd_ota_receiving(void);
void rd_ota_byte(unsigned char byte);
void rd_ota_poll(void);
/* Commit a PENDING_VERIFY image when healthy, otherwise roll back and reboot. */
void rd_ota_boot_check(bool healthy);
#endif
