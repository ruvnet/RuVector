#ifndef RD_PROFILE_H
#define RD_PROFILE_H
#include "app.h"
unsigned rd_profile_capacity(void);
size_t rd_profile_buffer_bytes(void);
bool rd_profile(const rd_context *ctx,rd_workspace *w,const float *inputs,
                size_t input_count,unsigned runs,bool energy);
#endif
