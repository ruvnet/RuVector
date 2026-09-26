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
bool rd_app_init(void);
void rd_app_byte(unsigned char byte);
#endif
