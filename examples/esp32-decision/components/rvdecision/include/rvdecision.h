#ifndef RVDECISION_H
#define RVDECISION_H
#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

#define RD_MAX_DIMS 768
#define RD_MAX_CLASSES 16
#define RD_VERSION 1
typedef enum { RD_CHOICE, RD_SCORE, RD_NOUL } rd_kind;
typedef enum { RD_PROTOTYPE, RD_PROBE, RD_LOGISTIC, RD_SIMILARITY } rd_head;
typedef enum { RD_OK, RD_BAD_MODEL, RD_BAD_INPUT } rd_status;
/* Arrays are immutable, compiler-owned flash data, never untrusted pointers. */
typedef struct {
    const void *values;
    size_t length;
    float scale;
} rd_row;
typedef struct {
    uint32_t version;
    uint8_t quant_bits;
    uint16_t dims, classes;
    rd_kind kind;
    rd_head head;
    const rd_row *prototypes, *negatives, *weights;
    const float *bias;
    size_t prototype_count, negative_count, weight_count, bias_count;
    float not_for_lambda, abstain_tau, abstain_scale;
    float logit_scale, temperature, platt_a, platt_b;
    bool source_calibrated;
} rd_model;
typedef struct {
    union { int8_t q8[RD_MAX_DIMS]; int16_t q16[RD_MAX_DIMS]; } input;
    float geometry[RD_MAX_CLASSES + 1];
    float logits[RD_MAX_CLASSES];
} rd_workspace;
typedef struct {
    uint16_t index;
    float probabilities[RD_MAX_CLASSES];
    float confidence, abstain, noul;
    bool accepted;
} rd_result;
typedef struct {
    const rd_model *model;
    float min_confidence, max_abstain;
} rd_context;

/* Validate once at boot. Do not mutate the model after successful init. */
rd_status rd_init(rd_context *ctx, const rd_model *model,
                  float min_confidence, float max_abstain);
/* Normalize finite, nonzero features and quantize activations dynamically.
 * The caller must use exactly the model's feature/embedding pipeline.
 * No heap allocation; one workspace per concurrent caller. On error the
 * output is cleared, rejected, and has abstain=1. */
rd_status rd_predict(const rd_context *ctx, const float *features, size_t count,
                     rd_workspace *workspace, rd_result *result);
int32_t rd_dot_i8(const int8_t *a, const int8_t *b, size_t count);
/* Both dot helpers require count <= RD_MAX_DIMS and valid typed arrays.
 * INT8 uses INT32 accumulation; INT16 uses INT64 to prevent overflow. */
int64_t rd_dot_i16(const int16_t *a, const int16_t *b, size_t count);
#endif
