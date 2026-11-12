#ifndef RD_APP_H
#define RD_APP_H
#include <stdint.h>
#include <stddef.h>
#include <stdbool.h>
#include "rvdecision.h"
uint64_t rd_clock_us(void);
const char *rd_target(void);
size_t rd_free_heap(void);
void rd_yield(void);
uint32_t rd_clock_cycles(void);
unsigned rd_cpu_hz(void);
unsigned rd_core_id(void);
bool rd_dynamic_frequency(void);
bool rd_fixed_affinity(void);
void rd_marker(bool active);
int rd_marker_gpio(void);
/* Board integration overrides these weak defaults with the actual driver.
 * Fill exactly count raw measurements in the model's documented units/order. */
bool rd_sensor_read(float *values,size_t count);
const char *rd_sensor_name(void);
/* Optional board-level commands (e.g. OTA). Return true when handled. */
bool rd_app_extension(const char *line);
bool rd_app_init(void);
void rd_app_byte(unsigned char byte);
#endif
