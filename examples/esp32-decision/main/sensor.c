#include "app.h"
__attribute__((weak)) bool rd_sensor_read(float *values,size_t count) {
    (void)values;(void)count;return false;
}
__attribute__((weak)) const char *rd_sensor_name(void) { return "unconfigured"; }
