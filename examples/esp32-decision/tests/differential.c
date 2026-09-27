#include "rvdecision.h"
#include <assert.h>
#include <float.h>
#include <math.h>
#include <stdio.h>
#include <string.h>

rd_status baseline_init(rd_context *, const rd_model *, float, float);
rd_status baseline_predict(const rd_context *, const float *, size_t, rd_workspace *, rd_result *);
static int16_t p16[RD_MAX_CLASSES][RD_MAX_DIMS], w16[RD_MAX_CLASSES][RD_MAX_DIMS];
static int8_t p8[RD_MAX_CLASSES][RD_MAX_DIMS], w8[RD_MAX_CLASSES][RD_MAX_DIMS];
static float input[RD_MAX_DIMS], bias[RD_MAX_CLASSES];
static rd_row protos[RD_MAX_CLASSES], weights[RD_MAX_CLASSES], negatives[RD_MAX_CLASSES];
static uint32_t seed=9927;
static float random_value(void) { seed=seed*1664525u+1013904223u; return (seed>>8)/8388608.0f-1; }

static unsigned check(unsigned d,unsigned k,unsigned bits,unsigned mode,unsigned neg) {
    for(unsigned c=0;c<k;++c) {
        for(unsigned i=0;i<d;++i) {
            p16[c][i]=(int16_t)(random_value()*32767); w16[c][i]=(int16_t)(random_value()*32767);
            p8[c][i]=(int8_t)(p16[c][i]/258); w8[c][i]=(int8_t)(w16[c][i]/258);
        }
        float qmax=bits==8 ? 127 : 32767;
        protos[c]=(rd_row){bits==8 ? (void*)p8[c] : (void*)p16[c],d,1/(sqrtf((float)d)*qmax)};
        weights[c]=(rd_row){bits==8 ? (void*)w8[c] : (void*)w16[c],d,3/qmax};
        bias[c]=random_value();
        negatives[c]=protos[neg==1 ? c : 0];
        if(neg==3) negatives[c].scale*=c+1; /* Same pointer, distinct scale. */
        if(neg==4 && c%2==0) negatives[c]=(rd_row){NULL,0,1};
    }
    rd_model m={.version=1,.quant_bits=(uint8_t)bits,.dims=(uint16_t)d,.classes=(uint16_t)k,
        .kind=mode>=3 ? RD_NOUL : mode==2 ? RD_SCORE : RD_CHOICE,
        .head=mode==0 ? RD_PROTOTYPE : mode<3 ? RD_PROBE : mode==3 ? RD_LOGISTIC : RD_SIMILARITY,
        .prototypes=protos,.prototype_count=k,.weights=weights,.weight_count=k,.bias=bias,.bias_count=k,
        .negatives=negatives,.negative_count=neg ? k : 0,.not_for_lambda=.5f,
        .abstain_tau=.35f,.abstain_scale=.5f,.logit_scale=5,.temperature=.13f,.platt_a=1.1f,.platt_b=-.3f};
    rd_context a,b;
    assert(baseline_init(&a,&m,.6f,.4f)==RD_OK && rd_init(&b,&m,.6f,.4f)==RD_OK);
    rd_workspace wa,wb; rd_result ra,rb;
    for(unsigned n=0;n<128;++n) {
        for(unsigned i=0;i<d;++i) input[i]=random_value();
        if(n==0) { memset(input,0,d*sizeof(float)); input[0]=FLT_MAX; }
        if(n==1) { memset(input,0,d*sizeof(float)); input[0]=-FLT_MIN; }
        if(n==2) memset(input,0,d*sizeof(float));
        if(n==3) input[0]=NAN;
        if(n==4) input[0]=INFINITY;
        rd_status sa=baseline_predict(&a,input,d,&wa,&ra), sb=rd_predict(&b,input,d,&wb,&rb);
        assert(sa==sb);
        /* Both functions zero the whole result, including padding. */
        if(memcmp(&ra,&rb,sizeof(ra))) {
            fprintf(stderr,"parity failure: d=%u k=%u bits=%u mode=%u negatives=%u row=%u\n",d,k,bits,mode,neg,n);
            return 0;
        }
    }
    return 128;
}
int main(void) {
    unsigned count=0;
    const unsigned dims[]={1,7,32,128,384,768}, classes[]={1,3,16};
    for(unsigned b=8;b<=16;b+=8) for(unsigned d=0;d<6;++d)
        for(unsigned mode=0;mode<5;++mode) for(unsigned c=0;c<(mode>=3 ? 1u : 3u);++c)
            for(unsigned neg=0;neg<5;++neg) {
                unsigned n=check(dims[d],classes[c],b,mode,neg);
                if(!n) return 1;
                count+=n;
            }
    printf("{\"comparisons\":%u,\"bit_identical_results\":true,\"includes_invalid_inputs\":true}\n",count);
}
