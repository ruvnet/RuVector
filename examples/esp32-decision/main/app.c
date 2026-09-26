#include "app.h"
#include <model.h>
#include <errno.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

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
    printf("{\"event\":\"ready\",\"target\":\"%s\",\"model_sha256\":\"%s\",\"model_id\":", rd_target(), RD_MODEL_HASH);
    json_string(RD_MODEL_ID);
    printf(",\"quant_bits\":%u,\"dims\":%u,\"classes\":%u,\"parameter_bytes\":%d,\"float_parameter_bytes\":%d,"
           "\"workspace_bytes\":%u,\"static_app_bytes\":%u,\"free_heap\":%u,\"source_calibrated\":%s,"
           "\"calibrated\":false,\"selftest_pass\":%s}\n",
           ctx.model->quant_bits, ctx.model->dims, ctx.model->classes, RD_PARAMETER_BYTES, RD_FLOAT_PARAMETER_BYTES,
           (unsigned)sizeof(workspace), (unsigned)(sizeof(workspace)+sizeof(features)+sizeof(line)+sizeof(ctx)),
           (unsigned)rd_free_heap(), ctx.model->source_calibrated ? "true" : "false", ready ? "true" : "false");
}
static void infer(char *input) {
    size_t n = 0; char *p = input;
    while (*p) {
        while (*p == ' ' || *p == '\t') ++p;
        if (!*p) break;
        if (n == ctx.model->dims) { error("dimension"); return; }
        errno = 0; char *end;
        float value = strtof(p, &end);
        if (p == end || errno == ERANGE || !isfinite(value) || (*end && *end != ' ' && *end != '\t')) {
            error("invalid_feature"); return;
        }
        features[n++] = value; p = end;
    }
    rd_result r;
    uint64_t start = rd_clock_us();
    rd_status status = rd_predict(&ctx, features, n, &workspace, &r);
    uint64_t elapsed = rd_clock_us()-start;
    if (status != RD_OK) { error("invalid_input"); return; }
    printf("{\"index\":%u,\"label\":", r.index);
    json_string(ctx.model->kind == RD_NOUL ? (r.index ? "yes" : "no") : rd_labels[r.index]);
    printf(",\"probabilities\":[");
    for (size_t i = 0; i < ctx.model->classes; ++i) printf("%s%.9g", i ? "," : "", (double)r.probabilities[i]);
    printf("],\"confidence\":%.9g,\"abstain\":%.9g,\"noul\":%.9g,\"accepted\":%s,\"inference_us\":%llu,\"calibrated\":false}\n",
           (double)r.confidence, (double)r.abstain, (double)r.noul, r.accepted ? "true" : "false", (unsigned long long)elapsed);
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
    else if (!strncmp(line, "infer ", 6)) infer(line+6);
    else error("unknown_command");
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
