#include "rvdecision.h"
#include <float.h>
#include <math.h>
#include <string.h>

static bool bounded(float x) { return isfinite(x) && fabsf(x) <= 1000000.0f; }
static bool row_valid(const rd_row *r, size_t dims, bool optional) {
    if (optional && !r->values) return r->length == 0;
    return r->values && r->length == dims && bounded(r->scale) && r->scale > 0;
}
rd_status rd_init(rd_context *ctx, const rd_model *m, float min_c, float max_a) {
    if (!ctx) return RD_BAD_MODEL;
    memset(ctx, 0, sizeof(*ctx));
    if (!m || m->version != RD_VERSION || (m->quant_bits != 8 && m->quant_bits != 16) || !m->dims || m->dims > RD_MAX_DIMS ||
        !m->classes || m->classes > RD_MAX_CLASSES ||
        !isfinite(min_c) || min_c < 0 || min_c > 1 ||
        !isfinite(max_a) || max_a < 0 || max_a > 1) return RD_BAD_MODEL;
    if (!bounded(m->not_for_lambda) || !bounded(m->abstain_tau) ||
        !bounded(m->abstain_scale) || fabsf(m->abstain_scale) < FLT_EPSILON ||
        !bounded(m->logit_scale) || m->logit_scale <= 0 ||
        !bounded(m->temperature) || m->temperature < 0.1f ||
        !bounded(m->platt_a) || !bounded(m->platt_b)) return RD_BAD_MODEL;
    if (m->kind == RD_NOUL) {
        if (m->classes != 1 || (m->head != RD_LOGISTIC && m->head != RD_SIMILARITY))
            return RD_BAD_MODEL;
    } else if ((m->kind != RD_CHOICE && m->kind != RD_SCORE) ||
               (m->head != RD_PROTOTYPE && m->head != RD_PROBE)) return RD_BAD_MODEL;
    if (!m->prototypes || m->prototype_count != m->classes) return RD_BAD_MODEL;
    if (m->negative_count && (!m->negatives || m->negative_count != m->classes))
        return RD_BAD_MODEL;
    bool trained = m->head == RD_PROBE || m->head == RD_LOGISTIC;
    if (trained && (!m->weights || !m->bias || m->weight_count != m->classes ||
                    m->bias_count != m->classes)) return RD_BAD_MODEL;
    for (size_t i = 0; i < m->classes; ++i) {
        if (!row_valid(&m->prototypes[i], m->dims, false)) return RD_BAD_MODEL;
        if (m->negative_count && !row_valid(&m->negatives[i], m->dims, true)) return RD_BAD_MODEL;
        if (trained && (!row_valid(&m->weights[i], m->dims, false) || !bounded(m->bias[i])))
            return RD_BAD_MODEL;
    }
    ctx->model = m; ctx->min_confidence = min_c; ctx->max_abstain = max_a;
    ctx->logit_factor = m->logit_scale / m->temperature;
    for (size_t i=0; i<m->classes; ++i) {
        ctx->negative_source[i]=(uint8_t)i;
        if (!m->negative_count || !m->negatives[i].values) continue;
        for (size_t j=0; j<i; ++j) {
            if (m->negatives[j].values == m->negatives[i].values &&
                m->negatives[j].scale == m->negatives[i].scale) {
                ctx->negative_source[i]=(uint8_t)j; break;
            }
        }
    }
    return RD_OK;
}

int32_t rd_dot_i8(const int8_t *a, const int8_t *b, size_t n) {
    int32_t s0 = 0;
    size_t i = 0;
#ifndef RD_SCALAR_DOT
    int32_t s1 = 0, s2 = 0, s3 = 0;
    for (; i + 3 < n; i += 4) {
        s0 += (int32_t)a[i] * b[i]; s1 += (int32_t)a[i+1] * b[i+1];
        s2 += (int32_t)a[i+2] * b[i+2]; s3 += (int32_t)a[i+3] * b[i+3];
    }
    s0 += s1 + s2 + s3;
#endif
    for (; i < n; ++i) s0 += (int32_t)a[i] * b[i];
    return s0;
}
int64_t rd_dot_i16(const int16_t *a, const int16_t *b, size_t n) {
    int64_t sum = 0;
    for (size_t i = 0; i < n; ++i) sum += (int32_t)a[i] * (int32_t)b[i];
    return sum;
}
/* Activations b are in [-32767,32767], so two products fit INT32 even
 * with full-range INT16 weights: 2*32768*32767 = 2147418112 < INT32_MAX.
 * Accumulate pairs in INT64. Keep the public full-range helper unchanged. */
#if defined(RD_PAIR_DOT) || defined(RD_TEST_HOOKS)
static int64_t dot_i16_quantized(const int16_t *a, const int16_t *b, size_t n) {
    int64_t sum=0;
    size_t i=0;
    for(; i+1<n; i+=2) {
        int32_t pair=(int32_t)a[i]*b[i]+(int32_t)a[i+1]*b[i+1];
        sum+=pair;
    }
    if(i<n) sum+=(int32_t)a[i]*b[i];
    return sum;
}
#ifdef RD_TEST_HOOKS
int64_t rd_test_dot_i16_quantized(const int16_t *a,const int16_t *b,size_t n) {
    return dot_i16_quantized(a,b,n);
}
#endif
#endif
static float dot(const rd_row *r, const rd_workspace *w, uint8_t bits, float scale) {
    float sum = bits == 8 ? (float)rd_dot_i8(r->values, w->input.q8, r->length) :
#ifdef RD_PAIR_DOT
                           (float)dot_i16_quantized(r->values, w->input.q16, r->length);
#else
                           (float)rd_dot_i16(r->values, w->input.q16, r->length);
#endif
    return sum * r->scale * scale;
}
/* exp(v - max) for the maximum element is expf(+0) == 1.0f exactly, so it is
 * returned without the call. Every other element, the summation order and the
 * final division are unchanged, so results are bit-identical. On targets
 * without an FPU each skipped expf is a large share of a small model's cost. */
static inline float exp_shifted(float v, float max) { return v == max ? 1.0f : expf(v - max); }
static void softmax(float *v, size_t n) {
    float max = v[0], sum = 0;
    for (size_t i = 1; i < n; ++i) if (v[i] > max) max = v[i];
    for (size_t i = 0; i < n; ++i) { v[i] = exp_shifted(v[i], max); sum += v[i]; }
    for (size_t i = 0; i < n; ++i) v[i] /= sum;
}
/* Only the abstain mass is consumed. Preserve the original summation order
 * and one final division, without normalizing the unused class masses. */
static float softmax_last(const float *v, size_t n) {
    float max=v[0], sum=0, last=0;
    for (size_t i=1; i<n; ++i) if(v[i]>max) max=v[i];
    for (size_t i=0; i<n; ++i) { last=exp_shifted(v[i],max); sum+=last; }
    return last/sum;
}
/* Exact half-away-from-zero for finite IEEE binary32 in [-32767,32767].
 * Every caller bounds the value before entry. Integer rounding avoids both
 * roundf and float-to-int helper calls on the C6's software float target. */
static int32_t round_quantized(float x) {
#ifdef RD_LIBM_ROUND
    return (int32_t)roundf(x);
#else
    _Static_assert(sizeof(float)==sizeof(uint32_t) && FLT_RADIX==2 && FLT_MANT_DIG==24,
                   "binary32 required");
    uint32_t bits; memcpy(&bits,&x,sizeof(bits));
    uint32_t magnitude=bits & UINT32_C(0x7fffffff);
    if(magnitude<UINT32_C(0x3f000000)) return 0;
    unsigned shift=150-(magnitude>>23); /* 9..24 over the documented interval */
    uint32_t mantissa=(magnitude & UINT32_C(0x7fffff)) | UINT32_C(0x800000);
    int32_t q=(int32_t)((mantissa+(UINT32_C(1)<<(shift-1)))>>shift);
    return bits>>31 ? -q : q;
#endif
}
#ifdef RD_TEST_HOOKS
int32_t rd_test_round_quantized(float x) { return round_quantized(x); }
#endif
rd_status rd_predict(const rd_context *ctx, const float *x, size_t n,
                     rd_workspace *w, rd_result *r) {
    if (!r) return RD_BAD_INPUT;
    memset(r, 0, sizeof(*r)); r->abstain = 1;
    if (!ctx || !ctx->model || !x || !w || n != ctx->model->dims) return RD_BAD_INPUT;
    const rd_model *m = ctx->model;
    float max = 0, norm2 = 0;
    size_t imax = 0;
    for (size_t i = 0; i < n; ++i) {
        if (!isfinite(x[i])) return RD_BAD_INPUT;
        if (fabsf(x[i]) > max) { max = fabsf(x[i]); imax = i; }
    }
    if (max < FLT_MIN) return RD_BAD_INPUT;
    /* Scaling before normalization avoids overflow and underflow in x*x. */
    float qmax = m->quant_bits == 8 ? 127.0f : 32767.0f;
    for (size_t i = 0; i < n; ++i) {
        /* x[imax]/max is exactly +-1: a sign copy replaces one software division. */
        float a = i == imax ? copysignf(1.0f, x[i]) : x[i] / max;
        norm2 += a * a;
        if (m->quant_bits == 8) w->input.q8[i] = (int8_t)round_quantized(a*qmax);
        else w->input.q16[i] = (int16_t)round_quantized(a*qmax);
    }
    float input_scale = 1.0f / (qmax * sqrtf(norm2));
    size_t k = m->classes;
    if (m->kind == RD_NOUL) {
        float p;
        if (m->head == RD_LOGISTIC) {
            float z = m->platt_a * (dot(m->weights, w, m->quant_bits, input_scale) + m->bias[0]) + m->platt_b;
            p = 1.0f / (1.0f + expf(-z));
        } else {
            p = fminf(1, fmaxf(0, (dot(m->prototypes, w, m->quant_bits, input_scale) + 1) * 0.5f));
        }
        r->noul = p; r->probabilities[0] = p; r->index = p >= 0.5f;
        r->confidence = fmaxf(p, 1-p); r->abstain = 0;
    } else {
        float max_sim = -FLT_MAX, best_neg = -FLT_MAX;
        float factor = ctx->logit_factor;
        for (size_t i = 0; i < k; ++i) {
            float sim = dot(&m->prototypes[i], w, m->quant_bits, input_scale), penalty = 0;
            max_sim = fmaxf(max_sim, sim);
            if (m->negative_count && m->negatives[i].values) {
                size_t source=ctx->negative_source[i];
                float ns = source==i ? dot(&m->negatives[i], w, m->quant_bits, input_scale) : w->negative_scores[source];
                w->negative_scores[i]=ns;
                best_neg = fmaxf(best_neg, ns); penalty = m->not_for_lambda * ns;
            }
            w->geometry[i] = (sim - penalty) * factor;
            w->logits[i] = m->head == RD_PROBE ?
                (dot(&m->weights[i], w, m->quant_bits, input_scale) + m->bias[i]) * factor : w->geometry[i];
        }
        w->geometry[k] = fmaxf(best_neg, (m->abstain_tau-max_sim)/m->abstain_scale) * factor;
        float abstain=softmax_last(w->geometry,k+1);
        softmax(w->logits, k);
        size_t best = 0; float expected = 0;
        for (size_t i = 0; i < k; ++i) {
            r->probabilities[i] = w->logits[i]; expected += (float)i * w->logits[i];
            if (w->logits[i] > w->logits[best]) best = i;
        }
        r->abstain = abstain;
        r->confidence = w->logits[best] * (1-r->abstain);
        r->index = m->kind == RD_SCORE ? (uint16_t)fminf((float)(k-1), floorf(expected+0.5f)) : (uint16_t)best;
    }
    r->accepted = r->confidence >= ctx->min_confidence && r->abstain <= ctx->max_abstain;
    return RD_OK;
}
