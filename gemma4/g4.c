/* g4 engine core — M1: GGUF model load + f32 reference forward pass.
 *
 * The graph follows the Gemma 4 reference implementation
 * (gemma/gm/nn/gemma4/_modules.py, _moe.py, _layers.py) and the design in
 * gemma4-port/01-confronto-architetturale.md:
 *   - pre/post norms and skip_scale per block (_modules.py:593-664)
 *   - local sliding GQA + global K=V attention (_modules.py:201-419)
 *   - partial RoPE, half-split pairing (gm/math/_positional_embeddings.py)
 *   - MoE softmax->top-k->renorm + per_expert_scale (_moe.py:301-379)
 *   - parallel dense FFN branch, GeGLU, tied embeddings + softcap. */

#include "g4.h"
#include "g4_gguf.h"
#include "g4_quants.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define G4_K_MASK -2.3819763e38f /* _modules.py:26 */

typedef struct {
    /* Attention. */
    float *attn_norm;      /* [D] pre_attention_norm */
    float *post_attn_norm; /* [D] */
    float *wq;             /* [H*K, D] */
    float *wk;             /* [KV*K, D] */
    float *wv;             /* [KV*K, D], NULL on global layers (K=V) */
    float *wo;             /* [D, H*K] */
    float *q_norm;         /* [K] */
    float *k_norm;         /* [K] */
    /* MoE branch. */
    float *ffn_norm;           /* [D] pre_ffw_norm */
    float *post_ffn_norm_moe;  /* [D] post_ffw1_norm */
    float *router;             /* [E, D] */
    float *router_scale;       /* [D] */
    float *per_expert_scale;   /* [E] */
    float *gate_exps;          /* [E, Hexp, D] */
    float *up_exps;            /* [E, Hexp, D] */
    float *down_exps;          /* [E, D, Hexp] */
    /* Dense branch. */
    float *ffn_norm_shexp;      /* [D] pre_ffw2_norm */
    float *post_ffn_norm_shexp; /* [D] post_ffw2_norm */
    float *gate_shexp;          /* [H2, D] */
    float *up_shexp;            /* [H2, D] */
    float *down_shexp;          /* [D, H2] */
    /* Combined. */
    float *post_ffn_norm; /* [D] post_ffw_norm */
    float *skip_scale;    /* [1] */
    bool is_global;
} g4_layer;

struct g4_model {
    uint32_t n_layer, d_model, vocab, ctx_len;
    uint32_t n_heads, n_kv_local, key_local, n_kv_global, key_global;
    uint32_t sliding_window, pattern_period;
    uint32_t n_experts, top_k, expert_dim, dense_ffn;
    float softcap, rms_eps;
    float rope_local_base, rope_global_base, rope_global_prop;
    bool k_eq_v_global;
    float *token_embd;  /* [vocab, D] */
    float *output_norm; /* [D] */
    g4_layer *layer;
};

/* ------------------------------------------------------------------ */
/* Loading                                                             */
/* ------------------------------------------------------------------ */

static void seterr(char *err, size_t errlen, const char *fmt, const char *a) {
    if (err && errlen) snprintf(err, errlen, fmt, a);
}

static bool get_u32(const g4_gguf *g, const char *key, uint32_t *out) {
    const g4_gguf_kv *kv = g4_gguf_get(g, key);
    if (!kv) return false;
    *out = (uint32_t)kv->v.u64;
    return true;
}

static bool get_f32(const g4_gguf *g, const char *key, float *out) {
    const g4_gguf_kv *kv = g4_gguf_get(g, key);
    if (!kv) return false;
    *out = (float)kv->v.f64;
    return true;
}

/* Loads a tensor and dequantizes it row by row into a fresh f32 buffer. */
static float *load_f32(const g4_gguf *g, const char *name, uint64_t want_elems,
                       char *err, size_t errlen) {
    const g4_gguf_tensor *t = g4_gguf_tensor_by_name(g, name);
    if (!t) { seterr(err, errlen, "missing tensor %s", name); return NULL; }
    if (!g4q_can_dequantize((g4q_type)t->type)) {
        seterr(err, errlen, "cannot dequantize tensor %s", name);
        return NULL;
    }
    uint64_t rows = 1;
    for (uint32_t i = 1; i < t->n_dims; i++) rows *= t->dims[i];
    uint64_t elems = rows * t->dims[0];
    if (want_elems && elems != want_elems) {
        seterr(err, errlen, "tensor %s has unexpected shape", name);
        return NULL;
    }
    uint8_t *raw = malloc((size_t)t->nbytes);
    float *out = malloc(sizeof(float) * (size_t)elems);
    if (!raw || !out || g4_gguf_read_tensor_data(g, t, raw)) {
        free(raw); free(out);
        seterr(err, errlen, "cannot read tensor %s", name);
        return NULL;
    }
    size_t row_bytes = g4q_row_size((g4q_type)t->type, (int64_t)t->dims[0]);
    for (uint64_t r = 0; r < rows; r++)
        g4q_dequant_row((g4q_type)t->type, raw + r * row_bytes,
                        out + r * t->dims[0], (int64_t)t->dims[0]);
    free(raw);
    return out;
}

void g4_model_free(g4_model *m) {
    if (!m) return;
    free(m->token_embd);
    free(m->output_norm);
    if (m->layer) {
        for (uint32_t i = 0; i < m->n_layer; i++) {
            g4_layer *l = &m->layer[i];
            free(l->attn_norm); free(l->post_attn_norm);
            free(l->wq); free(l->wk); free(l->wv); free(l->wo);
            free(l->q_norm); free(l->k_norm);
            free(l->ffn_norm); free(l->post_ffn_norm_moe);
            free(l->router); free(l->router_scale); free(l->per_expert_scale);
            free(l->gate_exps); free(l->up_exps); free(l->down_exps);
            free(l->ffn_norm_shexp); free(l->post_ffn_norm_shexp);
            free(l->gate_shexp); free(l->up_shexp); free(l->down_shexp);
            free(l->post_ffn_norm); free(l->skip_scale);
        }
        free(m->layer);
    }
    free(m);
}

g4_model *g4_model_load(const char *path, char *err, size_t errlen) {
    g4_gguf g;
    if (g4_gguf_open(&g, path, err, errlen)) return NULL;

    g4_model *m = calloc(1, sizeof(*m));
    bool ok = get_u32(&g, "gemma4.block_count", &m->n_layer) &&
              get_u32(&g, "gemma4.embedding_length", &m->d_model) &&
              get_u32(&g, "gemma4.vocab_size", &m->vocab) &&
              get_u32(&g, "gemma4.context_length", &m->ctx_len) &&
              get_u32(&g, "gemma4.attention.head_count", &m->n_heads) &&
              get_u32(&g, "gemma4.attention.head_count_kv", &m->n_kv_local) &&
              get_u32(&g, "gemma4.attention.global_head_count_kv", &m->n_kv_global) &&
              get_u32(&g, "gemma4.attention.key_length", &m->key_local) &&
              get_u32(&g, "gemma4.attention.global_key_length", &m->key_global) &&
              get_u32(&g, "gemma4.attention.sliding_window", &m->sliding_window) &&
              get_u32(&g, "gemma4.attention.pattern_period", &m->pattern_period) &&
              get_f32(&g, "gemma4.attention.layer_norm_rms_epsilon", &m->rms_eps) &&
              get_f32(&g, "gemma4.rope.local.freq_base", &m->rope_local_base) &&
              get_f32(&g, "gemma4.rope.global.freq_base", &m->rope_global_base) &&
              get_f32(&g, "gemma4.rope.global.partial_factor", &m->rope_global_prop) &&
              get_u32(&g, "gemma4.expert_count", &m->n_experts) &&
              get_u32(&g, "gemma4.expert_used_count", &m->top_k) &&
              get_u32(&g, "gemma4.expert_ffn_length", &m->expert_dim) &&
              get_u32(&g, "gemma4.dense_ffn_length", &m->dense_ffn) &&
              get_f32(&g, "gemma4.final_logit_softcap", &m->softcap);
    const g4_gguf_kv *kv = g4_gguf_get(&g, "gemma4.attention.k_eq_v_global");
    m->k_eq_v_global = kv ? kv->v.b : false;
    if (!ok) {
        seterr(err, errlen, "missing gemma4.* metadata%s", "");
        g4_gguf_close(&g);
        g4_model_free(m);
        return NULL;
    }

    const uint32_t D = m->d_model;
    m->token_embd = load_f32(&g, "token_embd.weight",
                             (uint64_t)m->vocab * D, err, errlen);
    m->output_norm = load_f32(&g, "output_norm.weight", D, err, errlen);
    m->layer = calloc(m->n_layer, sizeof(g4_layer));
    if (!m->token_embd || !m->output_norm) goto fail;

    for (uint32_t i = 0; i < m->n_layer; i++) {
        g4_layer *l = &m->layer[i];
        l->is_global = (i % m->pattern_period) == m->pattern_period - 1;
        const uint32_t K = l->is_global ? m->key_global : m->key_local;
        const uint32_t KV = l->is_global ? m->n_kv_global : m->n_kv_local;
        char name[128];
#define LOAD(field, suffix, elems) \
        do { \
            snprintf(name, sizeof(name), "blk.%u." suffix, i); \
            l->field = load_f32(&g, name, (elems), err, errlen); \
            if (!l->field) goto fail; \
        } while (0)
        LOAD(attn_norm, "attn_norm.weight", D);
        LOAD(post_attn_norm, "post_attn_norm.weight", D);
        LOAD(wq, "attn_q.weight", (uint64_t)m->n_heads * K * D);
        LOAD(wk, "attn_k.weight", (uint64_t)KV * K * D);
        if (!l->is_global || !m->k_eq_v_global)
            LOAD(wv, "attn_v.weight", (uint64_t)KV * K * D);
        LOAD(wo, "attn_output.weight", (uint64_t)D * m->n_heads * K);
        LOAD(q_norm, "attn_q_norm.weight", K);
        LOAD(k_norm, "attn_k_norm.weight", K);
        LOAD(ffn_norm, "ffn_norm.weight", D);
        LOAD(post_ffn_norm_moe, "post_ffn_norm_moe.weight", D);
        LOAD(router, "ffn_gate_inp.weight", (uint64_t)m->n_experts * D);
        LOAD(router_scale, "router_scale", D);
        LOAD(per_expert_scale, "per_expert_scale", m->n_experts);
        LOAD(gate_exps, "ffn_gate_exps.weight",
             (uint64_t)m->n_experts * m->expert_dim * D);
        LOAD(up_exps, "ffn_up_exps.weight",
             (uint64_t)m->n_experts * m->expert_dim * D);
        LOAD(down_exps, "ffn_down_exps.weight",
             (uint64_t)m->n_experts * D * m->expert_dim);
        LOAD(ffn_norm_shexp, "ffn_norm_shexp.weight", D);
        LOAD(post_ffn_norm_shexp, "post_ffn_norm_shexp.weight", D);
        LOAD(gate_shexp, "ffn_gate_shexp.weight", (uint64_t)m->dense_ffn * D);
        LOAD(up_shexp, "ffn_up_shexp.weight", (uint64_t)m->dense_ffn * D);
        LOAD(down_shexp, "ffn_down_shexp.weight", (uint64_t)D * m->dense_ffn);
        LOAD(post_ffn_norm, "post_ffn_norm.weight", D);
        LOAD(skip_scale, "skip_scale", 1);
#undef LOAD
    }

    g4_gguf_close(&g);
    return m;

fail:
    g4_gguf_close(&g);
    g4_model_free(m);
    return NULL;
}

uint32_t g4_model_vocab(const g4_model *m) { return m->vocab; }
uint32_t g4_model_layers(const g4_model *m) { return m->n_layer; }

/* ------------------------------------------------------------------ */
/* Math primitives                                                     */
/* ------------------------------------------------------------------ */

static void rms_norm(const float *x, const float *scale, float *out,
                     uint32_t n, float eps) {
    double var = 0.0;
    for (uint32_t i = 0; i < n; i++) var += (double)x[i] * x[i];
    float inv = 1.0f / sqrtf((float)(var / n) + eps);
    for (uint32_t i = 0; i < n; i++)
        out[i] = x[i] * inv * (scale ? scale[i] : 1.0f);
}

static float dot(const float *a, const float *b, uint32_t n) {
    double acc = 0.0;
    for (uint32_t i = 0; i < n; i++) acc += (double)a[i] * b[i];
    return (float)acc;
}

/* out[r] = rows[r] . x for a [nrows, n] row-major matrix. */
static void matvec(const float *rows, const float *x, float *out,
                   uint32_t nrows, uint32_t n) {
    for (uint32_t r = 0; r < nrows; r++) out[r] = dot(rows + (size_t)r * n, x, n);
}

/* gelu(x1)*x2 with the tanh approximation (gelu_pytorch_tanh; jax nn.gelu
 * default approximate=True). */
static float gelu_tanh(float x) {
    const float c = 0.7978845608028654f; /* sqrt(2/pi) */
    return 0.5f * x * (1.0f + tanhf(c * (x + 0.044715f * x * x * x)));
}

/* Partial RoPE, half-split pairing (gm/math/_positional_embeddings.py):
 * pairs (i, i+K/2) rotate for i < rope_angles, the rest pass through. */
static void apply_rope(float *v, uint32_t head_dim, uint32_t pos,
                       float base, float proportion) {
    uint32_t half = head_dim / 2;
    uint32_t rope_angles = (uint32_t)floorf(proportion * (float)head_dim / 2.0f);
    for (uint32_t i = 0; i < rope_angles; i++) {
        float exponent = (2.0f / (float)head_dim) * (float)i;
        float timescale = powf(base, exponent);
        float angle = (float)pos / timescale;
        float s = sinf(angle), c = cosf(angle);
        float a = v[i], b = v[i + half];
        v[i] = a * c - b * s;
        v[i + half] = b * c + a * s;
    }
}

static void softmax_inplace(float *x, uint32_t n) {
    float mx = x[0];
    for (uint32_t i = 1; i < n; i++) if (x[i] > mx) mx = x[i];
    double sum = 0.0;
    for (uint32_t i = 0; i < n; i++) { x[i] = expf(x[i] - mx); sum += x[i]; }
    float inv = (float)(1.0 / sum);
    for (uint32_t i = 0; i < n; i++) x[i] *= inv;
}

/* ------------------------------------------------------------------ */
/* Forward pass                                                        */
/* ------------------------------------------------------------------ */

typedef struct {
    float *k; /* [T, KV*K] */
    float *v; /* [T, KV*K] */
} g4_kv_cache;

int g4_forward_prefill_f32(const g4_model *m, const int32_t *tokens,
                           uint32_t n_tokens, float *logits) {
    const uint32_t D = m->d_model, H = m->n_heads;
    const uint32_t Kmax = m->key_global > m->key_local ? m->key_global
                                                       : m->key_local;
    const uint32_t Hexp = m->expert_dim, H2 = m->dense_ffn, E = m->n_experts;

    g4_kv_cache *cache = calloc(m->n_layer, sizeof(*cache));
    for (uint32_t i = 0; i < m->n_layer; i++) {
        const g4_layer *l = &m->layer[i];
        uint32_t K = l->is_global ? m->key_global : m->key_local;
        uint32_t KV = l->is_global ? m->n_kv_global : m->n_kv_local;
        cache[i].k = malloc(sizeof(float) * (size_t)n_tokens * KV * K);
        cache[i].v = malloc(sizeof(float) * (size_t)n_tokens * KV * K);
    }

    float *x = malloc(sizeof(float) * D);
    float *xn = malloc(sizeof(float) * D);
    float *tmp = malloc(sizeof(float) * D);
    float *res = malloc(sizeof(float) * D);
    float *q = malloc(sizeof(float) * H * Kmax);
    float *kraw = malloc(sizeof(float) * H * Kmax);
    float *att = malloc(sizeof(float) * n_tokens);
    float *enc = malloc(sizeof(float) * H * Kmax);
    float *gate = malloc(sizeof(float) * (Hexp > H2 ? Hexp : H2));
    float *up = malloc(sizeof(float) * (Hexp > H2 ? Hexp : H2));
    float *ffn = malloc(sizeof(float) * D);
    float *moe_out = malloc(sizeof(float) * D);
    float *dense_out = malloc(sizeof(float) * D);
    float *rlogits = malloc(sizeof(float) * E);

    for (uint32_t pos = 0; pos < n_tokens; pos++) {
        /* Embedder.encode: table row * sqrt(D)  (_modules.py:112-125) */
        const float *row = m->token_embd + (size_t)tokens[pos] * D;
        float emb_scale = sqrtf((float)D);
        for (uint32_t d = 0; d < D; d++) x[d] = row[d] * emb_scale;

        for (uint32_t li = 0; li < m->n_layer; li++) {
            const g4_layer *l = &m->layer[li];
            const uint32_t K = l->is_global ? m->key_global : m->key_local;
            const uint32_t KV = l->is_global ? m->n_kv_global : m->n_kv_local;
            const uint32_t group = H / KV;
            const float base = l->is_global ? m->rope_global_base
                                            : m->rope_local_base;
            const float prop = l->is_global ? m->rope_global_prop : 1.0f;

            /* 1. Attention (_modules.py:619-634). */
            rms_norm(x, l->attn_norm, xn, D, m->rms_eps);

            matvec(l->wq, xn, q, H * K, D);
            for (uint32_t h = 0; h < H; h++) {
                rms_norm(q + h * K, l->q_norm, q + h * K, K, m->rms_eps);
                apply_rope(q + h * K, K, pos, base, prop);
            }

            matvec(l->wk, xn, kraw, KV * K, D);
            float *kdst = cache[li].k + (size_t)pos * KV * K;
            float *vdst = cache[li].v + (size_t)pos * KV * K;
            for (uint32_t h = 0; h < KV; h++) {
                if (l->is_global && m->k_eq_v_global) {
                    /* K and V share the projection; K gets key_norm(+scale)
                     * and RoPE, V gets value_norm (no scale), no RoPE
                     * (_modules.py:277-295). */
                    rms_norm(kraw + h * K, NULL, vdst + h * K, K, m->rms_eps);
                    rms_norm(kraw + h * K, l->k_norm, kdst + h * K, K, m->rms_eps);
                    apply_rope(kdst + h * K, K, pos, base, prop);
                } else {
                    rms_norm(kraw + h * K, l->k_norm, kdst + h * K, K, m->rms_eps);
                    apply_rope(kdst + h * K, K, pos, base, prop);
                }
            }
            if (!(l->is_global && m->k_eq_v_global)) {
                matvec(l->wv, xn, kraw, KV * K, D);
                for (uint32_t h = 0; h < KV; h++)
                    rms_norm(kraw + h * K, NULL, vdst + h * K, K, m->rms_eps);
            }

            /* Per-head attention over positions 0..pos.  No 1/sqrt(d)
             * scaling: the reference applies none (_modules.py:322-334). */
            for (uint32_t h = 0; h < H; h++) {
                const uint32_t kvh = h / group;
                uint32_t s0 = 0;
                if (!l->is_global && pos + 1 > m->sliding_window)
                    s0 = pos + 1 - m->sliding_window; /* s > pos - window */
                for (uint32_t s = s0; s <= pos; s++)
                    att[s - s0] = dot(q + h * K,
                                      cache[li].k + ((size_t)s * KV + kvh) * K, K);
                softmax_inplace(att, pos - s0 + 1);
                float *eh = enc + h * K;
                memset(eh, 0, sizeof(float) * K);
                for (uint32_t s = s0; s <= pos; s++) {
                    const float p = att[s - s0];
                    const float *vv = cache[li].v + ((size_t)s * KV + kvh) * K;
                    for (uint32_t d = 0; d < K; d++) eh[d] += p * vv[d];
                }
            }
            matvec(l->wo, enc, tmp, D, H * K);
            rms_norm(tmp, l->post_attn_norm, tmp, D, m->rms_eps);
            for (uint32_t d = 0; d < D; d++) res[d] = x[d] + tmp[d];

            /* 2. FFN: dense branch (_modules.py:674-680). */
            rms_norm(res, l->ffn_norm_shexp, xn, D, m->rms_eps);
            matvec(l->gate_shexp, xn, gate, H2, D);
            matvec(l->up_shexp, xn, up, H2, D);
            for (uint32_t hh = 0; hh < H2; hh++)
                gate[hh] = gelu_tanh(gate[hh]) * up[hh];
            matvec(l->down_shexp, gate, dense_out, D, H2);
            rms_norm(dense_out, l->post_ffn_norm_shexp, dense_out, D, m->rms_eps);

            /* 2b. MoE branch (_moe.py:381-407): the router reads the
             * UN-normalized residual through its own scale-less RMSNorm. */
            rms_norm(res, NULL, tmp, D, m->rms_eps);
            const float root = 1.0f / sqrtf((float)D);
            for (uint32_t d = 0; d < D; d++)
                tmp[d] = tmp[d] * root * l->router_scale[d];
            matvec(l->router, tmp, rlogits, E, D);
            softmax_inplace(rlogits, E); /* probs now */

            /* exact top-k by probability (== by logit) */
            uint32_t sel[64];
            float wsum = 0.0f;
            for (uint32_t j = 0; j < m->top_k; j++) {
                uint32_t best = 0;
                float bp = -1.0f;
                for (uint32_t e = 0; e < E; e++) {
                    bool taken = false;
                    for (uint32_t t2 = 0; t2 < j; t2++)
                        if (sel[t2] == e) { taken = true; break; }
                    if (!taken && rlogits[e] > bp) { bp = rlogits[e]; best = e; }
                }
                sel[j] = best;
                wsum += rlogits[best];
            }
            if (wsum <= 0.0f) wsum = 1.0f; /* _moe.py:32-35 */

            rms_norm(res, l->ffn_norm, xn, D, m->rms_eps);
            memset(moe_out, 0, sizeof(float) * D);
            for (uint32_t j = 0; j < m->top_k; j++) {
                const uint32_t e = sel[j];
                const float w = rlogits[e] / wsum;
                const float pes = l->per_expert_scale[e];
                matvec(l->gate_exps + (size_t)e * Hexp * D, xn, gate, Hexp, D);
                matvec(l->up_exps + (size_t)e * Hexp * D, xn, up, Hexp, D);
                for (uint32_t hh = 0; hh < Hexp; hh++)
                    gate[hh] = gelu_tanh(gate[hh]) * up[hh];
                matvec(l->down_exps + (size_t)e * D * Hexp, gate, ffn, D, Hexp);
                for (uint32_t d = 0; d < D; d++)
                    moe_out[d] += w * pes * ffn[d];
            }
            rms_norm(moe_out, l->post_ffn_norm_moe, moe_out, D, m->rms_eps);

            /* Combine + residual + skip_scale (_modules.py:636-663,686-693). */
            for (uint32_t d = 0; d < D; d++) ffn[d] = dense_out[d] + moe_out[d];
            rms_norm(ffn, l->post_ffn_norm, ffn, D, m->rms_eps);
            for (uint32_t d = 0; d < D; d++)
                x[d] = (res[d] + ffn[d]) * l->skip_scale[0];
        }

        /* Final norm + tied decode + softcap (_transformer.py:332-336). */
        rms_norm(x, m->output_norm, xn, D, m->rms_eps);
        float *lg = logits + (size_t)pos * m->vocab;
        matvec(m->token_embd, xn, lg, m->vocab, D);
        for (uint32_t v = 0; v < m->vocab; v++)
            lg[v] = tanhf(lg[v] / m->softcap) * m->softcap;
    }

    for (uint32_t i = 0; i < m->n_layer; i++) {
        free(cache[i].k);
        free(cache[i].v);
    }
    free(cache);
    free(x); free(xn); free(tmp); free(res); free(q); free(kraw);
    free(att); free(enc); free(gate); free(up); free(ffn);
    free(moe_out); free(dense_out); free(rlogits);
    return 0;
}
