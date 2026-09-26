#define _POSIX_C_SOURCE 200809L
#include "rvdecision.h"
#include <model.h>
#include <assert.h>
#include <stdio.h>
#include <time.h>

rd_status baseline_init(rd_context *, const rd_model *, float, float);
rd_status baseline_predict(const rd_context *, const float *, size_t, rd_workspace *, rd_result *);
static volatile float checksum;
static uint64_t ns(void) {
    struct timespec t; clock_gettime(CLOCK_MONOTONIC,&t);
    return (uint64_t)t.tv_sec*1000000000+t.tv_nsec;
}
static double timed(const rd_context *ctx,int baseline,unsigned runs) {
    rd_workspace workspace; rd_result r;
    uint64_t start=ns();
    for(unsigned i=0;i<runs;++i) {
        const float *x=rd_golden_inputs[i%RD_GOLDEN_COUNT];
        rd_status status=baseline ? baseline_predict(ctx,x,RD_MODEL_DIMS,&workspace,&r) :
                                   rd_predict(ctx,x,RD_MODEL_DIMS,&workspace,&r);
        assert(status==RD_OK); checksum+=r.confidence;
    }
    return (double)(ns()-start)/runs;
}
int main(void) {
    rd_context baseline,current;
    assert(baseline_init(&baseline,&rd_firmware_model,.6f,.4f)==RD_OK);
    assert(rd_init(&current,&rd_firmware_model,.6f,.4f)==RD_OK);
    timed(&baseline,1,256); timed(&current,0,256);
    printf("{\"dims\":%u,\"classes\":%u,\"bits\":%u,\"model_sha256\":\"%s\",\"runs_per_round\":2048,\"input_count\":%u,\"rounds\":[",
           rd_firmware_model.dims,rd_firmware_model.classes,rd_firmware_model.quant_bits,RD_MODEL_HASH,RD_GOLDEN_COUNT);
    for(unsigned i=0;i<11;++i) {
        double a,b;
        if(i%2) { b=timed(&current,0,2048); a=timed(&baseline,1,2048); }
        else { a=timed(&baseline,1,2048); b=timed(&current,0,2048); }
        printf("%s{\"baseline_ns\":%.3f,\"candidate_ns\":%.3f}",i?",":"",a,b);
    }
    puts("]}");
}
