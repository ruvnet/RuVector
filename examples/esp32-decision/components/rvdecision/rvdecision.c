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
static float dot(const rd_row *r, const rd_workspace *w, uint8_t bits, float scale) {
    float sum = bits == 8 ? (float)rd_dot_i8(r->values, w->input.q8, r->length) :
                           (float)rd_dot_i16(r->values, w->input.q16, r->length);
    return sum * r->scale * scale;
}
static void softmax(float *v, size_t n) {
    float max = v[0], sum = 0;
    for (size_t i = 1; i < n; ++i) if (v[i] > max) max = v[i];
    for (size_t i = 0; i < n; ++i) { v[i] = expf(v[i] - max); sum += v[i]; }
    for (size_t i = 0; i < n; ++i) v[i] /= sum;
}
rd_status rd_predict(const rd_context *ctx, const float *x, size_t n,
                     rd_workspace *w, rd_result *r) {
    if (!r) return RD_BAD_INPUT;
    memset(r, 0, sizeof(*r)); r->abstain = 1;
    if (!ctx || !ctx->model || !x || !w || n != ctx->model->dims) return RD_BAD_INPUT;
    const rd_model *m = ctx->model;
    float max = 0, norm2 = 0;
    for (size_t i = 0; i < n; ++i) {
        if (!isfinite(x[i])) return RD_BAD_INPUT;
        if (fabsf(x[i]) > max) max = fabsf(x[i]);
    }
    if (max < FLT_MIN) return RD_BAD_INPUT;
    /* Scaling before normalization avoids overflow and underflow in x*x. */
    float qmax = m->quant_bits == 8 ? 127.0f : 32767.0f;
    for (size_t i = 0; i < n; ++i) {
        float a = x[i] / max;
        norm2 += a * a;
        if (m->quant_bits == 8) w->input.q8[i] = (int8_t)roundf(a*qmax);
        else w->input.q16[i] = (int16_t)roundf(a*qmax);
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
        float factor = m->logit_scale / m->temperature;
        for (size_t i = 0; i < k; ++i) {
            float sim = dot(&m->prototypes[i], w, m->quant_bits, input_scale), penalty = 0;
            max_sim = fmaxf(max_sim, sim);
            if (m->negative_count && m->negatives[i].values) {
                float ns = dot(&m->negatives[i], w, m->quant_bits, input_scale);
                best_neg = fmaxf(best_neg, ns); penalty = m->not_for_lambda * ns;
            }
            w->geometry[i] = (sim - penalty) * factor;
            w->logits[i] = m->head == RD_PROBE ?
                (dot(&m->weights[i], w, m->quant_bits, input_scale) + m->bias[i]) * factor : w->geometry[i];
        }
        w->geometry[k] = fmaxf(best_neg, (m->abstain_tau-max_sim)/m->abstain_scale) * factor;
        softmax(w->geometry, k+1); softmax(w->logits, k);
        size_t best = 0; float expected = 0;
        for (size_t i = 0; i < k; ++i) {
            r->probabilities[i] = w->logits[i]; expected += (float)i * w->logits[i];
            if (w->logits[i] > w->logits[best]) best = i;
        }
        r->abstain = w->geometry[k];
        r->confidence = w->logits[best] * (1-r->abstain);
        r->index = m->kind == RD_SCORE ? (uint16_t)fminf((float)(k-1), floorf(expected+0.5f)) : (uint16_t)best;
    }
    r->accepted = r->confidence >= ctx->min_confidence && r->abstain <= ctx->max_abstain;
    return RD_OK;
}
