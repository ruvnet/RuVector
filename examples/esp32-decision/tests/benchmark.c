#define _POSIX_C_SOURCE 200809L
#include "rvdecision.h"
#include <assert.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

rd_status baseline_init(rd_context *, const rd_model *, float, float);
rd_status baseline_predict(const rd_context *, const float *, size_t, rd_workspace *, rd_result *);
static int16_t p16[RD_MAX_CLASSES][RD_MAX_DIMS], w16[RD_MAX_CLASSES][RD_MAX_DIMS];
static int8_t p8[RD_MAX_CLASSES][RD_MAX_DIMS], w8[RD_MAX_CLASSES][RD_MAX_DIMS];
static float inputs[64][RD_MAX_DIMS], biases[RD_MAX_CLASSES];
static rd_row protos[RD_MAX_CLASSES], weights[RD_MAX_CLASSES], negatives[RD_MAX_CLASSES];
static volatile float checksum;
static unsigned seed=17;
static float random_value(void) { seed=seed*1664525u+1013904223u; return ((seed>>8)/8388608.0f)-1; }
static uint64_t ns(void) { struct timespec t; clock_gettime(CLOCK_MONOTONIC,&t); return (uint64_t)t.tv_sec*1000000000+t.tv_nsec; }
static double timed(const rd_context *ctx, int baseline, unsigned runs) {
    rd_workspace workspace; rd_result r;
    uint64_t start=ns();
    for(unsigned i=0;i<runs;++i) {
        rd_status s=baseline ? baseline_predict(ctx,inputs[i%64],ctx->model->dims,&workspace,&r) :
                              rd_predict(ctx,inputs[i%64],ctx->model->dims,&workspace,&r);
        assert(s==RD_OK); checksum+=r.confidence;
    }
    return (double)(ns()-start)/runs;
}
int main(int argc,char **argv) {
    assert(argc==6);
    unsigned d=(unsigned)atoi(argv[1]), k=(unsigned)atoi(argv[2]), bits=(unsigned)atoi(argv[3]);
    int head=atoi(argv[4]);
    unsigned negative=(unsigned)atoi(argv[5]);
    assert(negative<=2);
    assert(d>0 && d<=RD_MAX_DIMS && k>0 && k<=RD_MAX_CLASSES && (bits==8 || bits==16));
    for(unsigned c=0;c<k;++c) {
        for(unsigned i=0;i<d;++i) {
            p16[c][i]=(int16_t)(random_value()*32767); w16[c][i]=(int16_t)(random_value()*32767);
            p8[c][i]=(int8_t)(p16[c][i]/258); w8[c][i]=(int8_t)(w16[c][i]/258);
        }
        float qmax=bits==8 ? 127 : 32767;
        protos[c]=(rd_row){bits==8 ? (void*)p8[c] : (void*)p16[c],d,1.0f/(sqrtf((float)d)*qmax)};
        weights[c]=(rd_row){bits==8 ? (void*)w8[c] : (void*)w16[c],d,3.0f/qmax};
        negatives[c]=protos[negative==1 ? c : c%2];
    }
    for(unsigned n=0;n<64;++n) for(unsigned i=0;i<d;++i) inputs[n][i]=random_value();
    rd_model model={.version=1,.quant_bits=(uint8_t)bits,.dims=(uint16_t)d,.classes=(uint16_t)k,
        .kind=RD_CHOICE,.head=head ? RD_PROBE : RD_PROTOTYPE,.prototypes=protos,.prototype_count=k,
        .weights=weights,.weight_count=k,.bias=biases,.bias_count=k,.negatives=negatives,.negative_count=negative ? k : 0,
        .not_for_lambda=.5f,.abstain_tau=.35f,.abstain_scale=.5f,.logit_scale=5,.temperature=.5f,.platt_a=1};
    rd_context current,baseline; assert(rd_init(&current,&model,.6f,.4f)==RD_OK);
    assert(baseline_init(&baseline,&model,.6f,.4f)==RD_OK);
    unsigned runs=2048;
    timed(&baseline,1,256); timed(&current,0,256);
    printf("{\"dims\":%u,\"classes\":%u,\"bits\":%u,\"head\":\"%s\",\"negatives\":\"%s\",\"runs_per_round\":%u,\"rounds\":[",d,k,bits,head?"probe":"prototype",negative==0?"none":negative==1?"unique":"shared",runs);
    for(unsigned round=0;round<11;++round) {
        double a,b;
        if(round%2) {b=timed(&current,0,runs); a=timed(&baseline,1,runs);}
        else {a=timed(&baseline,1,runs);b=timed(&current,0,runs);}
        printf("%s{\"baseline_ns\":%.3f,\"candidate_ns\":%.3f}",round?",":"",a,b);
    }
    puts("]}");
}
