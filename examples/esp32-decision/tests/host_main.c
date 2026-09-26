#define _POSIX_C_SOURCE 200809L
#include "app.h"
#include <stdio.h>
#include <time.h>
uint64_t rd_clock_us(void) {
    struct timespec t; clock_gettime(CLOCK_MONOTONIC, &t);
    return (uint64_t)t.tv_sec*1000000 + (uint64_t)t.tv_nsec/1000;
}
const char *rd_target(void) { return "host"; }
size_t rd_free_heap(void) { return 0; }
void rd_yield(void) {}
int main(void) {
    setvbuf(stdout, NULL, _IONBF, 0);
    if (!rd_app_init()) return 1;
    int c; while ((c = getchar()) != EOF) rd_app_byte((unsigned char)c);
    return 0;
}
