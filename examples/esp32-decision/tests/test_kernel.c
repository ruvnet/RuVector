#include "rvdecision.h"
#include <assert.h>
#include <float.h>
#include <math.h>
#include <stdio.h>
#include <string.h>

static uint32_t rng = 719;
static int8_t random_i8(void) { rng = rng*1664525u+1013904223u; return (int8_t)(rng >> 24); }
int main(void) {
    int8_t a[RD_MAX_DIMS], b[RD_MAX_DIMS];
    for (size_t n = 1; n <= RD_MAX_DIMS; ++n) {
        int32_t expected = 0;
        for (size_t i = 0; i < n; ++i) { a[i]=random_i8(); b[i]=random_i8(); expected+=(int32_t)a[i]*b[i]; }
        assert(rd_dot_i8(a,b,n) == expected);
    }
    memset(a, -128, sizeof(a)); memset(b, -128, sizeof(b));
    assert(rd_dot_i8(a,b,RD_MAX_DIMS) == 16384*RD_MAX_DIMS);
    int16_t a16[RD_MAX_DIMS], b16[RD_MAX_DIMS];
    for (size_t i=0; i<RD_MAX_DIMS; ++i) a16[i]=b16[i]=-32768;
    assert(rd_dot_i16(a16,b16,RD_MAX_DIMS) == (int64_t)1073741824*RD_MAX_DIMS);
    const int8_t v0[] = {127,0,0}, v1[] = {0,127,0};
    rd_row rows[] = {{v0,3,1.0f/127},{v1,3,1.0f/127}};
    rd_model m = {.version=1,.quant_bits=8,.dims=3,.classes=2,.kind=RD_CHOICE,.head=RD_PROTOTYPE,
        .prototypes=rows,.prototype_count=2,.abstain_tau=.35f,.abstain_scale=.5f,
        .logit_scale=5,.temperature=1,.platt_a=1};
    rd_context ctx; rd_workspace w; rd_result r;
    assert(rd_init(&ctx,&m,.6f,.4f)==RD_OK);
    float x[] = {1,0,0};
    assert(rd_predict(&ctx,x,3,&w,&r)==RD_OK && r.index==0 && r.accepted);
    assert(fabsf(r.probabilities[0]+r.probabilities[1]-1)<1e-6f);
    x[0]=FLT_MAX;
    assert(rd_predict(&ctx,x,3,&w,&r)==RD_OK && r.index==0);
    x[0]=FLT_MIN;
    assert(rd_predict(&ctx,x,3,&w,&r)==RD_OK && r.index==0);
    x[0]=0; x[2]=1;
    assert(rd_predict(&ctx,x,3,&w,&r)==RD_OK && !r.accepted && r.abstain>.8f);
    x[2]=0;
    assert(rd_predict(&ctx,x,3,&w,&r)==RD_BAD_INPUT && !r.accepted && r.abstain==1);
    x[0]=NAN;
    assert(rd_predict(&ctx,x,3,&w,&r)==RD_BAD_INPUT && !r.accepted);
    x[0]=INFINITY;
    assert(rd_predict(&ctx,x,3,&w,&r)==RD_BAD_INPUT);
    assert(rd_predict(&ctx,x,2,&w,&r)==RD_BAD_INPUT);
    assert(rd_predict(&ctx,NULL,3,&w,&r)==RD_BAD_INPUT);
    assert(rd_predict(NULL,x,3,&w,&r)==RD_BAD_INPUT);
    assert(rd_init(&ctx,&m,NAN,.4f)==RD_BAD_MODEL && !ctx.model);
    assert(rd_init(&ctx,&m,.6f,2)==RD_BAD_MODEL);
    rd_model bad=m; bad.dims=RD_MAX_DIMS+1; assert(rd_init(&ctx,&bad,.6f,.4f)==RD_BAD_MODEL);
    bad=m; bad.prototype_count=1; assert(rd_init(&ctx,&bad,.6f,.4f)==RD_BAD_MODEL);
    bad=m; bad.kind=RD_NOUL; assert(rd_init(&ctx,&bad,.6f,.4f)==RD_BAD_MODEL);
    bad=m; bad.head=RD_PROBE; assert(rd_init(&ctx,&bad,.6f,.4f)==RD_BAD_MODEL);
    bad=m; bad.temperature=0; assert(rd_init(&ctx,&bad,.6f,.4f)==RD_BAD_MODEL);
    bad=m; bad.abstain_scale=NAN; assert(rd_init(&ctx,&bad,.6f,.4f)==RD_BAD_MODEL);
    rows[0].length=2; assert(rd_init(&ctx,&m,.6f,.4f)==RD_BAD_MODEL);
    puts("kernel validation, overflow bounds, 768 dot products and fail-closed tests passed");
}
