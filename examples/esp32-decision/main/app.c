#include "app.h"
#include "profile.h"
#include <model.h>
#include <errno.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#ifndef RD_KERNEL_SHA256
#define RD_KERNEL_SHA256 "unknown"
#endif
#ifndef RD_PREPROCESSING
#define RD_PREPROCESSING 0
#endif

static rd_context ctx;
static rd_workspace workspace;
static float features[RD_MODEL_DIMS];
static char line[RD_MODEL_DIMS * 24 + 32];
static size_t used;
static bool overflow, ready;

static void json_string(const char *s) {
    putchar('"');
    for (; *s; ++s) {
        unsigned char c = (unsigned char)*s;
        if (c == '"' || c == '\\') { putchar('\\'); putchar(c); }
        else if (c < 32) printf("\\u%04x", c);
        else putchar(c);
    }
    putchar('"');
}
__attribute__((weak)) bool rd_app_extension(const char *line) { (void)line; return false; }
static void error(const char *code) { printf("{\"error\":\"%s\",\"accepted\":false}\n", code); }
static bool selftest(void) {
#if RD_GOLDEN_COUNT == 0
    return false;
#else
    for (size_t i = 0; i < RD_GOLDEN_COUNT; ++i) {
        rd_result r;
        if (rd_predict(&ctx, rd_golden_inputs[i], ctx.model->dims, &workspace, &r) != RD_OK) return false;
        const rd_result *e = &rd_golden_expected[i];
        if (r.index != e->index || fabsf(r.confidence-e->confidence) > .025f ||
            fabsf(r.abstain-e->abstain) > .025f || fabsf(r.noul-e->noul) > .025f) return false;
        for (size_t j = 0; j < ctx.model->classes; ++j)
            if (fabsf(r.probabilities[j]-e->probabilities[j]) > .025f) return false;
    }
    return true;
#endif
}
static void meta(void) {
    printf("{\"event\":\"ready\",\"target\":\"%s\",\"kernel_sha256\":\"%s\",\"cpu_hz\":%u,\"core\":%u,\"dynamic_frequency\":%s,\"fixed_affinity\":%s,\"marker_gpio\":%d,\"profile_buffer_bytes\":%u,\"profile_capacity\":%u,\"sensor_pipeline\":%s,\"model_sha256\":\"%s\",\"model_id\":",
           rd_target(),RD_KERNEL_SHA256,rd_cpu_hz(),rd_core_id(),rd_dynamic_frequency()?"true":"false",rd_fixed_affinity()?"true":"false",rd_marker_gpio(),(unsigned)rd_profile_buffer_bytes(),rd_profile_capacity(),RD_PREPROCESSING?"true":"false",RD_MODEL_HASH);
    json_string(RD_MODEL_ID);
    printf(",\"capture_driver\":");json_string(rd_sensor_name());
    printf(",\"quant_bits\":%u,\"dims\":%u,\"classes\":%u,\"parameter_bytes\":%d,\"float_parameter_bytes\":%d,"
           "\"workspace_bytes\":%u,\"static_app_bytes\":%u,\"free_heap\":%u,\"source_calibrated\":%s,"
           "\"calibrated\":false,\"selftest_pass\":%s}\n",
           ctx.model->quant_bits, ctx.model->dims, ctx.model->classes, RD_PARAMETER_BYTES, RD_FLOAT_PARAMETER_BYTES,
           (unsigned)sizeof(workspace), (unsigned)(sizeof(workspace)+sizeof(features)+sizeof(line)+sizeof(ctx)),
           (unsigned)rd_free_heap(), ctx.model->source_calibrated ? "true" : "false", ready ? "true" : "false");
}
/* "format compact" replies carry each float as its 8-hex-digit binary32 bit
 * pattern and omit the label and timing fields; "format json" restores the
 * default. Decisions are identical; only the encoding changes. */
static bool compact;
static void hex_float(float v) {
    uint32_t bits; memcpy(&bits, &v, sizeof(bits));
    static const char digits[] = "0123456789abcdef";
    char out[11] = {'"'};
    for (int i = 0; i < 8; ++i) out[1+i] = digits[(bits >> (28 - 4*i)) & 0xf];
    out[9] = '"'; out[10] = 0;
    fputs(out, stdout);
}
static void decide_features(size_t n,bool sensor,uint64_t parse_us,int64_t capture_us) {
    uint64_t pre_start=rd_clock_us();
    if(sensor) {
#if RD_PREPROCESSING
        if(n!=RD_MODEL_DIMS) { error("dimension");return; }
        for(size_t i=0;i<n;++i) {
            features[i]=(features[i]-rd_feature_mean[i])/rd_feature_std[i];
            if(!isfinite(features[i])) { error("invalid_feature");return; }
        }
#else
        error("sensor_pipeline_unavailable");return;
#endif
    }
    uint64_t preprocess_us=rd_clock_us()-pre_start;
    rd_result r;
    uint64_t start = rd_clock_us();
    rd_status status = rd_predict(&ctx, features, n, &workspace, &r);
    uint64_t elapsed = rd_clock_us()-start;
    if (status != RD_OK) { error("invalid_input"); return; }
    if (compact) {
        /* Exact binary32 bit patterns; no float-to-decimal work on the MCU. */
        printf("{\"i\":%u,\"a\":%d,\"p\":[", r.index, r.accepted ? 1 : 0);
        for (size_t i = 0; i < ctx.model->classes; ++i) {
            if (i) putchar(',');
            hex_float(r.probabilities[i]);
        }
        printf("],\"c\":"); hex_float(r.confidence);
        printf(",\"b\":"); hex_float(r.abstain);
        printf(",\"n\":"); hex_float(r.noul);
        printf("}\n");
        return;
    }
    printf("{\"index\":%u,\"label\":", r.index);
    json_string(ctx.model->kind == RD_NOUL ? (r.index ? "yes" : "no") : rd_labels[r.index]);
    printf(",\"probabilities\":[");
    for (size_t i = 0; i < ctx.model->classes; ++i) printf("%s%.9g", i ? "," : "", (double)r.probabilities[i]);
    printf("],\"confidence\":%.9g,\"abstain\":%.9g,\"noul\":%.9g,\"accepted\":%s,\"inference_us\":%llu,\"parse_us\":%llu,\"preprocess_us\":%llu,\"capture_us\":",
           (double)r.confidence, (double)r.abstain, (double)r.noul, r.accepted ? "true" : "false", (unsigned long long)elapsed,
           (unsigned long long)parse_us,(unsigned long long)preprocess_us);
    if(capture_us<0) printf("null");else printf("%llu",(unsigned long long)capture_us);
    printf(",\"calibrated\":false}\n");
}
static void infer(char *input,bool sensor) {
    uint64_t request_start=rd_clock_us();
    size_t n=0;char *p=input;
    while(*p) {
        while(*p==' ' || *p=='\t')++p;
        if(!*p)break;
        if(n==ctx.model->dims) { error("dimension");return; }
        errno=0;char *end;float value=strtof(p,&end);
        if(p==end || errno==ERANGE || !isfinite(value) || (*end && *end!=' ' && *end!='\t')) {
            error("invalid_feature");return;
        }
        features[n++]=value;p=end;
    }
    decide_features(n,sensor,rd_clock_us()-request_start,-1);
}
/* Exact binary input: each value is the 8-hex-digit IEEE-754 binary32 bit
 * pattern (e.g. 3f800000 for 1.0). No decimal conversion runs on the MCU, and
 * the kernel receives exactly the host's float. Non-finite values are rejected
 * exactly as in the decimal path. */
static int hex_digit(char c) {
    if(c>='0' && c<='9') return c-'0';
    if(c>='a' && c<='f') return c-'a'+10;
    if(c>='A' && c<='F') return c-'A'+10;
    return -1;
}
static void infer_hex(const char *p,bool sensor) {
    uint64_t request_start=rd_clock_us();
    size_t n=0;
    while(*p) {
        while(*p==' ' || *p=='\t')++p;
        if(!*p)break;
        if(n==ctx.model->dims) { error("dimension");return; }
        uint32_t bits=0;
        for(int i=0;i<8;++i) {
            int d=hex_digit(p[i]);
            if(d<0) { error("invalid_feature");return; }
            bits=(bits<<4)|(uint32_t)d;
        }
        p+=8;
        if(*p && *p!=' ' && *p!='\t') { error("invalid_feature");return; }
        float value;memcpy(&value,&bits,sizeof(value));
        if(!isfinite(value)) { error("invalid_feature");return; }
        features[n++]=value;
    }
    decide_features(n,sensor,rd_clock_us()-request_start,-1);
}
static void sample(void) {
    if(!RD_PREPROCESSING) { error("sensor_pipeline_unavailable");return; }
    for(size_t i=0;i<ctx.model->dims;++i)features[i]=NAN;
    uint64_t start=rd_clock_us();
    if(!rd_sensor_read(features,ctx.model->dims)) { error("capture_unavailable");return; }
    decide_features(ctx.model->dims,true,0,(int64_t)(rd_clock_us()-start));
}
static void benchmark(void) {
    for (size_t i = 0; i < ctx.model->dims; ++i)
        features[i] = (ctx.model->quant_bits == 8 ? ((const int8_t *)ctx.model->prototypes[0].values)[i] :
                         ((const int16_t *)ctx.model->prototypes[0].values)[i]) * ctx.model->prototypes[0].scale;
    uint64_t total = 0, worst = 0; rd_result r;
    const unsigned runs = 1000;
    for (unsigned i = 0; i < runs; ++i) {
        uint64_t start = rd_clock_us();
        if (rd_predict(&ctx, features, ctx.model->dims, &workspace, &r) != RD_OK) { error("benchmark"); return; }
        uint64_t us = rd_clock_us()-start; total += us; if (us > worst) worst = us;
        if (i%50 == 0) rd_yield();
    }
    printf("{\"benchmark\":true,\"runs\":%u,\"mean_us\":%.3f,\"max_us\":%llu,\"free_heap\":%u}\n",
           runs, (double)total/runs, (unsigned long long)worst, (unsigned)rd_free_heap());
}
static void command(void) {
    if (!strcmp(line, "meta")) meta();
    else if (!strcmp(line, "selftest")) printf("{\"selftest_pass\":%s,\"vectors\":%d}\n", selftest() ? "true" : "false", RD_GOLDEN_COUNT);
    else if (!ready) error("not_ready");
    else if (!strcmp(line, "bench")) benchmark();
    else if (!strcmp(line,"sample")) sample();
    else if (!strncmp(line, "infer ", 6)) infer(line+6,false);
    else if (!strncmp(line, "sensor ", 7)) infer(line+7,true);
    else if (!strncmp(line, "inferx ", 7)) infer_hex(line+7,false);
    else if (!strncmp(line, "sensorx ", 8)) infer_hex(line+8,true);
    else if (!strncmp(line,"profile ",8) || !strncmp(line,"energy ",7)) {
        bool energy=line[0]=='e';
        if(!energy && !rd_profile_capacity()) { error("profile_disabled");return; }
        char *arg=line+(energy?7:8),*end;
        errno=0;unsigned long runs=strtoul(arg,&end,10);
        if(errno || end==arg || *end || *arg=='-' || runs>2048) { error("profile_runs");return; }
#if RD_GOLDEN_COUNT > 0
        if(!rd_profile(&ctx,&workspace,&rd_golden_inputs[0][0],RD_GOLDEN_COUNT,(unsigned)runs,energy)) error("profile_runs");
#else
        error("no_profile_inputs");
#endif
    }
    else if (!strcmp(line, "format compact") || !strcmp(line, "format json")) {
        compact = line[7] == 'c';
        printf("{\"format\":\"%s\"}\n", compact ? "compact" : "json");
    }
    else if (!rd_app_extension(line)) error("unknown_command");
}
bool rd_app_init(void) {
    if (rd_init(&ctx, &rd_firmware_model, .6f, .4f) != RD_OK) { error("model"); return false; }
    ready = selftest(); meta(); return ready;
}
void rd_app_byte(unsigned char c) {
    if (c == '\r') return;
    if (c == '\n') {
        if (overflow) error("line_too_long_or_control");
        else { line[used] = 0; command(); }
        used = 0; overflow = false; fflush(stdout); return;
    }
    if (c == 0 || (c < 32 && c != '\t') || used+1 >= sizeof(line)) overflow = true;
    if (!overflow) line[used++] = (char)c;
}
