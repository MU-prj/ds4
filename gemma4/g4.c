/* g4 engine core — model load, f32 forward pass, session API, G4SP payload.
 *
 * The graph follows the Gemma 4 reference implementation
 * (gemma/gm/nn/gemma4/_modules.py, _moe.py, _layers.py) and the design in
 * gemma4-port/01-confronto-architetturale.md:
 *   - pre/post norms and skip_scale per block (_modules.py:593-664)
 *   - local sliding GQA + global K=V attention (_modules.py:201-419)
 *   - partial RoPE, half-split pairing (gm/math/_positional_embeddings.py)
 *   - MoE softmax->top-k->renorm + per_expert_scale (_moe.py:301-379)
 *   - parallel dense FFN branch, GeGLU, tied embeddings + softcap.
 *
 * Sessions keep the KV state exactly as the disk design assumes
 * (gemma4-port/03 §3): local layers hold only the last `sliding_window`
 * rows in a ring, global layers hold the full history. */

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
    uint32_t down_row;    /* row length of down_exps (>= expert_dim: k-quant
                             block padding, gemma4-port/02 §2) */
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
        {
            snprintf(name, sizeof(name), "blk.%u.ffn_down_exps.weight", i);
            const g4_gguf_tensor *td = g4_gguf_tensor_by_name(&g, name);
            if (!td || td->dims[0] < m->expert_dim) {
                seterr(err, errlen, "bad down_exps tensor %s", name);
                goto fail;
            }
            l->down_row = (uint32_t)td->dims[0];
        }
        LOAD(down_exps, "ffn_down_exps.weight",
             (uint64_t)m->n_experts * D * l->down_row);
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
/* Sessions                                                            */
/* ------------------------------------------------------------------ */

struct g4_session {
    const g4_model *m;
    uint32_t ctx;    /* token capacity */
    uint32_t n_past; /* tokens evaluated so far */
    int32_t *tokens; /* [ctx] history */
    float *logits;   /* [vocab] next-token logits */
    /* Per-layer KV state: local layers keep a ring of `sliding_window`
     * rows (slot = pos % window), global layers keep [ctx] rows. */
    float **ck;
    float **cv;
    /* Scratch. */
    float *x, *xn, *tmp, *res, *q, *kraw, *att, *enc;
    float *gate, *up, *ffn, *moe_out, *dense_out, *rlogits;
};

static uint32_t layer_cache_rows(const g4_model *m, uint32_t li, uint32_t ctx) {
    return m->layer[li].is_global
               ? ctx
               : (ctx < m->sliding_window ? ctx : m->sliding_window);
}

static uint32_t layer_row_elems(const g4_model *m, uint32_t li) {
    const g4_layer *l = &m->layer[li];
    return (l->is_global ? m->n_kv_global : m->n_kv_local) *
           (l->is_global ? m->key_global : m->key_local);
}

g4_session *g4_session_create(const g4_model *m, uint32_t ctx) {
    if (!m || ctx == 0) return NULL;
    g4_session *s = calloc(1, sizeof(*s));
    s->m = m;
    s->ctx = ctx;
    s->tokens = malloc(sizeof(int32_t) * ctx);
    s->logits = calloc(m->vocab, sizeof(float));
    s->ck = calloc(m->n_layer, sizeof(float *));
    s->cv = calloc(m->n_layer, sizeof(float *));
    for (uint32_t i = 0; i < m->n_layer; i++) {
        size_t n = (size_t)layer_cache_rows(m, i, ctx) * layer_row_elems(m, i);
        s->ck[i] = malloc(sizeof(float) * n);
        s->cv[i] = malloc(sizeof(float) * n);
    }
    const uint32_t D = m->d_model;
    const uint32_t Kmax = m->key_global > m->key_local ? m->key_global
                                                       : m->key_local;
    const uint32_t Hbig = m->expert_dim > m->dense_ffn ? m->expert_dim
                                                       : m->dense_ffn;
    s->x = malloc(sizeof(float) * D);
    s->xn = malloc(sizeof(float) * D);
    s->tmp = malloc(sizeof(float) * D);
    s->res = malloc(sizeof(float) * D);
    s->q = malloc(sizeof(float) * m->n_heads * Kmax);
    s->kraw = malloc(sizeof(float) * m->n_heads * Kmax);
    s->att = malloc(sizeof(float) * ctx);
    s->enc = malloc(sizeof(float) * m->n_heads * Kmax);
    s->gate = malloc(sizeof(float) * Hbig);
    s->up = malloc(sizeof(float) * Hbig);
    s->ffn = malloc(sizeof(float) * D);
    s->moe_out = malloc(sizeof(float) * D);
    s->dense_out = malloc(sizeof(float) * D);
    s->rlogits = malloc(sizeof(float) * m->n_experts);
    return s;
}

void g4_session_free(g4_session *s) {
    if (!s) return;
    for (uint32_t i = 0; i < s->m->n_layer; i++) {
        free(s->ck[i]);
        free(s->cv[i]);
    }
    free(s->ck); free(s->cv);
    free(s->tokens); free(s->logits);
    free(s->x); free(s->xn); free(s->tmp); free(s->res); free(s->q);
    free(s->kraw); free(s->att); free(s->enc); free(s->gate); free(s->up);
    free(s->ffn); free(s->moe_out); free(s->dense_out); free(s->rlogits);
    free(s);
}

uint32_t g4_session_pos(const g4_session *s) { return s->n_past; }
const int32_t *g4_session_tokens(const g4_session *s) { return s->tokens; }
const float *g4_session_logits(const g4_session *s) { return s->logits; }

int g4_session_eval(g4_session *s, int32_t token) {
    const g4_model *m = s->m;
    const uint32_t D = m->d_model, H = m->n_heads, W = m->sliding_window;
    const uint32_t Hexp = m->expert_dim, H2 = m->dense_ffn, E = m->n_experts;
    const uint32_t pos = s->n_past;
    if (pos >= s->ctx || token < 0 || (uint32_t)token >= m->vocab) return -1;
    s->tokens[pos] = token;

    float *x = s->x, *xn = s->xn, *tmp = s->tmp, *res = s->res;

    /* Embedder.encode: table row * sqrt(D)  (_modules.py:112-125) */
    const float *row = m->token_embd + (size_t)token * D;
    const float emb_scale = sqrtf((float)D);
    for (uint32_t d = 0; d < D; d++) x[d] = row[d] * emb_scale;

    for (uint32_t li = 0; li < m->n_layer; li++) {
        const g4_layer *l = &m->layer[li];
        const uint32_t K = l->is_global ? m->key_global : m->key_local;
        const uint32_t KV = l->is_global ? m->n_kv_global : m->n_kv_local;
        const uint32_t group = H / KV;
        const float base = l->is_global ? m->rope_global_base
                                        : m->rope_local_base;
        const float prop = l->is_global ? m->rope_global_prop : 1.0f;
        const uint32_t rows = layer_cache_rows(m, li, s->ctx);

        /* 1. Attention (_modules.py:619-634). */
        rms_norm(x, l->attn_norm, xn, D, m->rms_eps);

        matvec(l->wq, xn, s->q, H * K, D);
        for (uint32_t h = 0; h < H; h++) {
            rms_norm(s->q + h * K, l->q_norm, s->q + h * K, K, m->rms_eps);
            apply_rope(s->q + h * K, K, pos, base, prop);
        }

        const uint32_t slot = l->is_global ? pos : pos % rows;
        matvec(l->wk, xn, s->kraw, KV * K, D);
        float *kdst = s->ck[li] + (size_t)slot * KV * K;
        float *vdst = s->cv[li] + (size_t)slot * KV * K;
        for (uint32_t h = 0; h < KV; h++) {
            if (l->is_global && m->k_eq_v_global) {
                /* K and V share the projection; K gets key_norm(+scale) and
                 * RoPE, V gets value_norm (no scale), no RoPE
                 * (_modules.py:277-295). */
                rms_norm(s->kraw + h * K, NULL, vdst + h * K, K, m->rms_eps);
                rms_norm(s->kraw + h * K, l->k_norm, kdst + h * K, K, m->rms_eps);
                apply_rope(kdst + h * K, K, pos, base, prop);
            } else {
                rms_norm(s->kraw + h * K, l->k_norm, kdst + h * K, K, m->rms_eps);
                apply_rope(kdst + h * K, K, pos, base, prop);
            }
        }
        if (!(l->is_global && m->k_eq_v_global)) {
            matvec(l->wv, xn, s->kraw, KV * K, D);
            for (uint32_t h = 0; h < KV; h++)
                rms_norm(s->kraw + h * K, NULL, vdst + h * K, K, m->rms_eps);
        }

        /* Visible span: global layers see 0..pos, local layers the last
         * `window` positions (sliding mask, _modules.py:38-52).  With the
         * ring these are exactly the live rows. */
        uint32_t n_att = pos + 1;
        if (!l->is_global && n_att > W) n_att = W;
        const uint32_t q0 = pos + 1 - n_att;

        /* Per-head attention.  No 1/sqrt(d) scaling: the reference applies
         * none (_modules.py:322-334). */
        for (uint32_t h = 0; h < H; h++) {
            const uint32_t kvh = h / group;
            for (uint32_t j = 0; j < n_att; j++) {
                const uint32_t qpos = q0 + j;
                const uint32_t sl = l->is_global ? qpos : qpos % rows;
                s->att[j] = dot(s->q + h * K,
                                s->ck[li] + ((size_t)sl * KV + kvh) * K, K);
            }
            softmax_inplace(s->att, n_att);
            float *eh = s->enc + h * K;
            memset(eh, 0, sizeof(float) * K);
            for (uint32_t j = 0; j < n_att; j++) {
                const uint32_t qpos = q0 + j;
                const uint32_t sl = l->is_global ? qpos : qpos % rows;
                const float p = s->att[j];
                const float *vv = s->cv[li] + ((size_t)sl * KV + kvh) * K;
                for (uint32_t d = 0; d < K; d++) eh[d] += p * vv[d];
            }
        }
        matvec(l->wo, s->enc, tmp, D, H * K);
        rms_norm(tmp, l->post_attn_norm, tmp, D, m->rms_eps);
        for (uint32_t d = 0; d < D; d++) res[d] = x[d] + tmp[d];

        /* 2. FFN: dense branch (_modules.py:674-680). */
        rms_norm(res, l->ffn_norm_shexp, xn, D, m->rms_eps);
        matvec(l->gate_shexp, xn, s->gate, H2, D);
        matvec(l->up_shexp, xn, s->up, H2, D);
        for (uint32_t hh = 0; hh < H2; hh++)
            s->gate[hh] = gelu_tanh(s->gate[hh]) * s->up[hh];
        matvec(l->down_shexp, s->gate, s->dense_out, D, H2);
        rms_norm(s->dense_out, l->post_ffn_norm_shexp, s->dense_out, D,
                 m->rms_eps);

        /* 2b. MoE branch (_moe.py:381-407): the router reads the
         * UN-normalized residual through its own scale-less RMSNorm. */
        rms_norm(res, NULL, tmp, D, m->rms_eps);
        const float root = 1.0f / sqrtf((float)D);
        for (uint32_t d = 0; d < D; d++)
            tmp[d] = tmp[d] * root * l->router_scale[d];
        matvec(l->router, tmp, s->rlogits, E, D);
        softmax_inplace(s->rlogits, E); /* probs now */

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
                if (!taken && s->rlogits[e] > bp) { bp = s->rlogits[e]; best = e; }
            }
            sel[j] = best;
            wsum += s->rlogits[best];
        }
        if (wsum <= 0.0f) wsum = 1.0f; /* _moe.py:32-35 */

        rms_norm(res, l->ffn_norm, xn, D, m->rms_eps);
        memset(s->moe_out, 0, sizeof(float) * D);
        for (uint32_t j = 0; j < m->top_k; j++) {
            const uint32_t e = sel[j];
            const float w = s->rlogits[e] / wsum;
            const float pes = l->per_expert_scale[e];
            matvec(l->gate_exps + (size_t)e * Hexp * D, xn, s->gate, Hexp, D);
            matvec(l->up_exps + (size_t)e * Hexp * D, xn, s->up, Hexp, D);
            for (uint32_t hh = 0; hh < Hexp; hh++)
                s->gate[hh] = gelu_tanh(s->gate[hh]) * s->up[hh];
            const float *wd = l->down_exps + (size_t)e * D * l->down_row;
            for (uint32_t d = 0; d < D; d++)
                s->ffn[d] = dot(wd + (size_t)d * l->down_row, s->gate, Hexp);
            for (uint32_t d = 0; d < D; d++)
                s->moe_out[d] += w * pes * s->ffn[d];
        }
        rms_norm(s->moe_out, l->post_ffn_norm_moe, s->moe_out, D, m->rms_eps);

        /* Combine + residual + skip_scale (_modules.py:636-663,686-693). */
        for (uint32_t d = 0; d < D; d++)
            s->ffn[d] = s->dense_out[d] + s->moe_out[d];
        rms_norm(s->ffn, l->post_ffn_norm, s->ffn, D, m->rms_eps);
        for (uint32_t d = 0; d < D; d++)
            x[d] = (res[d] + s->ffn[d]) * l->skip_scale[0];
    }

    /* Final norm + tied decode + softcap (_transformer.py:332-336). */
    rms_norm(x, m->output_norm, xn, D, m->rms_eps);
    matvec(m->token_embd, xn, s->logits, m->vocab, D);
    for (uint32_t v = 0; v < m->vocab; v++)
        s->logits[v] = tanhf(s->logits[v] / m->softcap) * m->softcap;

    s->n_past = pos + 1;
    return 0;
}

int g4_forward_prefill_f32(const g4_model *m, const int32_t *tokens,
                           uint32_t n_tokens, float *logits) {
    g4_session *s = g4_session_create(m, n_tokens);
    if (!s) return -1;
    for (uint32_t p = 0; p < n_tokens; p++) {
        if (g4_session_eval(s, tokens[p])) {
            g4_session_free(s);
            return -1;
        }
        memcpy(logits + (size_t)p * m->vocab, s->logits,
               sizeof(float) * m->vocab);
    }
    g4_session_free(s);
    return 0;
}

/* ------------------------------------------------------------------ */
/* G4SP payload (gemma4-port/03-disk-kv-cache.md §5)                   */
/* ------------------------------------------------------------------ */

uint64_t g4_session_payload_bytes(const g4_session *s) {
    const g4_model *m = s->m;
    const uint32_t T = s->n_past;
    uint64_t bytes = 4u * G4_SESSION_PAYLOAD_U32_FIELDS;
    bytes += 4ull * T;              /* tokens */
    bytes += 4ull * m->vocab;       /* logits */
    for (uint32_t li = 0; li < m->n_layer; li++) {
        uint32_t R = m->layer[li].is_global
                         ? T
                         : (T < m->sliding_window ? T : m->sliding_window);
        bytes += 4 + 2ull * 4ull * R * layer_row_elems(m, li);
    }
    return bytes;
}

static int payload_header(const g4_session *s, uint32_t h[G4_SESSION_PAYLOAD_U32_FIELDS]) {
    const g4_model *m = s->m;
    memcpy(&h[0], "G4SP", 4);
    h[1] = G4_SESSION_PAYLOAD_VERSION;
    h[2] = s->ctx;
    h[3] = 0; /* prefill chunk: single-token reference path */
    h[4] = m->n_layer;
    h[5] = m->pattern_period;
    h[6] = m->sliding_window;
    h[7] = m->n_kv_local;
    h[8] = m->key_local;
    h[9] = m->n_kv_global;
    h[10] = m->key_global;
    h[11] = m->vocab;
    h[12] = s->n_past;
    h[13] = 1; /* kv dtype: f32 */
    h[14] = 1; /* flags: logits present */
    h[15] = 0;
    return 0;
}

int g4_session_save_payload(const g4_session *s, FILE *fp,
                            char *err, size_t errlen) {
    const g4_model *m = s->m;
    const uint32_t T = s->n_past;
    uint32_t h[G4_SESSION_PAYLOAD_U32_FIELDS];
    payload_header(s, h);
    if (fwrite(h, 4, G4_SESSION_PAYLOAD_U32_FIELDS, fp) !=
        G4_SESSION_PAYLOAD_U32_FIELDS)
        goto werr;
    if (T && fwrite(s->tokens, 4, T, fp) != T) goto werr;
    if (fwrite(s->logits, 4, m->vocab, fp) != m->vocab) goto werr;

    for (uint32_t li = 0; li < m->n_layer; li++) {
        const bool g = m->layer[li].is_global;
        const uint32_t rows_cap = layer_cache_rows(m, li, s->ctx);
        const uint32_t elems = layer_row_elems(m, li);
        const uint32_t R = g ? T : (T < m->sliding_window ? T : m->sliding_window);
        if (fwrite(&R, 4, 1, fp) != 1) goto werr;
        /* Rows in logical position order T-R..T-1 (doc 03 §3). */
        for (int pass = 0; pass < 2; pass++) {
            const float *cache = pass == 0 ? s->ck[li] : s->cv[li];
            for (uint32_t j = 0; j < R; j++) {
                const uint32_t qpos = T - R + j;
                const uint32_t sl = g ? qpos : qpos % rows_cap;
                if (fwrite(cache + (size_t)sl * elems, 4, elems, fp) != elems)
                    goto werr;
            }
        }
    }
    return 0;
werr:
    seterr(err, errlen, "payload write failed%s", "");
    return -1;
}

int g4_session_load_payload(g4_session *s, FILE *fp, char *err, size_t errlen) {
    const g4_model *m = s->m;
    uint32_t h[G4_SESSION_PAYLOAD_U32_FIELDS];
    if (fread(h, 4, G4_SESSION_PAYLOAD_U32_FIELDS, fp) !=
        G4_SESSION_PAYLOAD_U32_FIELDS)
    {
        seterr(err, errlen, "payload header read failed%s", "");
        return -1;
    }
    if (memcmp(&h[0], "G4SP", 4) || h[1] != G4_SESSION_PAYLOAD_VERSION ||
        h[4] != m->n_layer || h[5] != m->pattern_period ||
        h[6] != m->sliding_window || h[7] != m->n_kv_local ||
        h[8] != m->key_local || h[9] != m->n_kv_global ||
        h[10] != m->key_global || h[11] != m->vocab || h[13] != 1)
    {
        seterr(err, errlen, "payload does not match the model shape%s", "");
        return -1;
    }
    const uint32_t T = h[12];
    if (T > s->ctx) {
        seterr(err, errlen, "payload longer than session context%s", "");
        return -1;
    }
    if (T && fread(s->tokens, 4, T, fp) != T) goto rerr;
    if (fread(s->logits, 4, m->vocab, fp) != m->vocab) goto rerr;

    for (uint32_t li = 0; li < m->n_layer; li++) {
        const bool g = m->layer[li].is_global;
        const uint32_t rows_cap = layer_cache_rows(m, li, s->ctx);
        const uint32_t elems = layer_row_elems(m, li);
        uint32_t R;
        if (fread(&R, 4, 1, fp) != 1) goto rerr;
        const uint32_t expect =
            g ? T : (T < m->sliding_window ? T : m->sliding_window);
        if (R != expect) {
            seterr(err, errlen, "payload row count mismatch%s", "");
            return -1;
        }
        for (int pass = 0; pass < 2; pass++) {
            float *cache = pass == 0 ? s->ck[li] : s->cv[li];
            for (uint32_t j = 0; j < R; j++) {
                const uint32_t qpos = T - R + j;
                const uint32_t sl = g ? qpos : qpos % rows_cap;
                if (fread(cache + (size_t)sl * elems, 4, elems, fp) != elems)
                    goto rerr;
            }
        }
    }
    s->n_past = T;
    return 0;
rerr:
    seterr(err, errlen, "payload read failed%s", "");
    return -1;
}
