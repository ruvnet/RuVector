#include "profile.h"
#include <limits.h>
#include <stdio.h>
#include <stdlib.h>

#define RD_PROFILE_MAX 2048
static uint32_t times[RD_PROFILE_MAX],cycles[RD_PROFILE_MAX];
static int order(const void *a,const void *b) {
    uint32_t x=*(const uint32_t*)a,y=*(const uint32_t*)b;
    return (x>y)-(x<y);
}
static unsigned percentile(const uint32_t *v,unsigned n,unsigned p) {
    return (unsigned)v[((size_t)n*p+99)/100-1];
}
bool rd_profile(const rd_context *ctx,rd_workspace *w,const float *inputs,
                size_t input_count,unsigned runs,bool energy) {
    if(!runs || runs>(energy ? 256u : RD_PROFILE_MAX) || !input_count) return false;
    rd_result result;
    for(unsigned i=0;i<32;++i)
        if(rd_predict(ctx,inputs+(i%input_count)*ctx->model->dims,ctx->model->dims,w,&result)!=RD_OK) return false;
    uint32_t overhead_us=UINT32_MAX,overhead_cycles=UINT32_MAX;
    for(unsigned i=0;i<64;++i) {
        uint64_t start=rd_clock_us(); uint32_t c=rd_clock_cycles();
        uint32_t dc=rd_clock_cycles()-c,dt=(uint32_t)(rd_clock_us()-start);
        if(dt<overhead_us) overhead_us=dt;
        if(dc<overhead_cycles) overhead_cycles=dc;
    }
    size_t heap=rd_free_heap(); unsigned core=rd_core_id(),hz=rd_cpu_hz();
    uint64_t sum=0,csum=0;
    rd_marker(energy);
    uint64_t batch_start=rd_clock_us();
    for(unsigned i=0;i<runs;++i) {
        uint64_t start=energy ? 0 : rd_clock_us();
        uint32_t c=energy ? 0 : rd_clock_cycles();
        rd_status status=rd_predict(ctx,inputs+(i%input_count)*ctx->model->dims,ctx->model->dims,w,&result);
        if(status!=RD_OK) { rd_marker(false);return false; }
        if(!energy) {
            cycles[i]=rd_clock_cycles()-c; times[i]=(uint32_t)(rd_clock_us()-start);
            sum+=times[i];csum+=cycles[i];
            if(i%32==31) rd_yield();
        }
    }
    uint64_t batch_us=rd_clock_us()-batch_start;
    rd_marker(false);
    if(core!=rd_core_id() || hz!=rd_cpu_hz()) return false;
    if(energy) {
        printf("{\"energy_batch\":true,\"runs\":%u,\"batch_us\":%llu,\"marker_gpio\":%d,\"scope\":\"kernel_batch\"}\n",
               runs,(unsigned long long)batch_us,rd_marker_gpio());
        return true;
    }
    qsort(times,runs,sizeof(times[0]),order);qsort(cycles,runs,sizeof(cycles[0]),order);
    printf("{\"profile\":true,\"runs\":%u,\"input_count\":%u,\"mean_us\":%.6f,\"p50_us\":%u,\"p95_us\":%u,\"p99_us\":%u,\"max_us\":%u,"
           "\"mean_cycles\":%.3f,\"p50_cycles\":%u,\"p95_cycles\":%u,\"p99_cycles\":%u,\"max_cycles\":%u,"
           "\"timer_overhead_us\":%u,\"cycle_overhead\":%u,\"cpu_hz\":%u,\"core\":%u,\"heap_before\":%u,\"heap_after\":%u,\"sample_buffer_bytes\":%u}\n",
           runs,(unsigned)input_count,(double)sum/runs,percentile(times,runs,50),percentile(times,runs,95),percentile(times,runs,99),(unsigned)times[runs-1],
           (double)csum/runs,percentile(cycles,runs,50),percentile(cycles,runs,95),percentile(cycles,runs,99),(unsigned)cycles[runs-1],
           (unsigned)overhead_us,(unsigned)overhead_cycles,hz,core,(unsigned)heap,(unsigned)rd_free_heap(),(unsigned)(sizeof(times)+sizeof(cycles)));
    return true;
}
