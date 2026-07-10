/* g4 M0 test runner: quant round-trips, GGUF round-trip, KVG header. */

#include "../g4_gguf.h"
#include "../g4_kvstore.h"
#include "../g4_quants.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static int g_failures;

#define CHECK(cond, ...) do { \
    if (!(cond)) { \
        g_failures++; \
        fprintf(stderr, "FAIL %s:%d: ", __FILE__, __LINE__); \
        fprintf(stderr, __VA_ARGS__); \
        fprintf(stderr, "\n"); \
    } \
} while (0)

static uint64_t rng_state = 0x9e3779b97f4a7c15ull;

static uint64_t rng_next(void) {
    rng_state ^= rng_state << 13;
    rng_state ^= rng_state >> 7;
    rng_state ^= rng_state << 17;
    return rng_state;
}

/* Approximately gaussian via sum of uniforms. */
static float rng_gauss(void) {
    float acc = 0.0f;
    for (int i = 0; i < 12; i++)
        acc += (float)(rng_next() >> 40) / (float)(1 << 24);
    return acc - 6.0f;
}

static double rel_rms_err(const float *ref, const float *got, int64_t n) {
    double num = 0.0, den = 0.0;
    for (int64_t i = 0; i < n; i++) {
        double d = (double)got[i] - (double)ref[i];
        num += d * d;
        den += (double)ref[i] * (double)ref[i];
    }
    return den > 0.0 ? sqrt(num / den) : 0.0;
}

static void test_f16_bf16(void) {
    const float vals[] = { 0.0f, 1.0f, -1.0f, 0.5f, 1024.0f, -3.140625f };
    for (size_t i = 0; i < sizeof(vals) / sizeof(vals[0]); i++) {
        uint16_t h;
        g4q_f32_to_f16_row(&vals[i], &h, 1);
        CHECK(g4q_f16_to_f32(h) == vals[i], "f16 round-trip of %g", vals[i]);
        uint16_t b;
        g4q_f32_to_bf16_row(&vals[i], &b, 1);
        CHECK(g4q_bf16_to_f32(b) == vals[i], "bf16 round-trip of %g", vals[i]);
    }
}

static void test_quant_roundtrip(g4q_type type, int64_t ncols, int64_t nrows,
                                 bool with_imatrix, double max_rel_err) {
    const int64_t n = ncols * nrows;
    float *src = malloc(sizeof(float) * (size_t)n);
    float *out = malloc(sizeof(float) * (size_t)n);
    float *imatrix = NULL;
    for (int64_t i = 0; i < n; i++) src[i] = rng_gauss();
    if (with_imatrix || g4q_requires_imatrix(type)) {
        imatrix = malloc(sizeof(float) * (size_t)ncols);
        for (int64_t i = 0; i < ncols; i++)
            imatrix[i] = 0.5f + (float)(rng_next() & 0xff) / 255.0f;
    }

    size_t row_size = g4q_row_size(type, ncols);
    CHECK(row_size > 0, "%s: row size for %lld cols", g4q_type_name(type),
          (long long)ncols);
    uint8_t *q = malloc(row_size * (size_t)nrows);
    g4q_quantize_init(type);
    size_t written = g4q_quantize_chunk(type, src, q, 0, nrows, ncols, imatrix);
    CHECK(written == row_size * (size_t)nrows,
          "%s: quantize_chunk wrote %zu, want %zu", g4q_type_name(type),
          written, row_size * (size_t)nrows);

    for (int64_t r = 0; r < nrows; r++)
        g4q_dequant_row(type, q + (size_t)r * row_size, out + r * ncols, ncols);

    double err = rel_rms_err(src, out, n);
    CHECK(err < max_rel_err, "%s: rel RMS err %.4f >= %.4f",
          g4q_type_name(type), err, max_rel_err);
    printf("  %-8s %4lld cols x %lld rows  rel-rms-err %.4f (max %.3f)\n",
           g4q_type_name(type), (long long)ncols, (long long)nrows, err,
           max_rel_err);

    free(src);
    free(out);
    free(q);
    free(imatrix);
}

/* Down-projection padding case (doc 02 §2): rows of 704 real values padded
 * with 64 zeros to 768.  The zero tail must decode to exactly 0.0 and must
 * not disturb the real values beyond the normal Q2_K error. */
static void test_q2k_down_padding(void) {
    const int64_t real_cols = 704, pad_cols = 768, nrows = 8;
    float *src = calloc((size_t)(pad_cols * nrows), sizeof(float));
    float *out = malloc(sizeof(float) * (size_t)(pad_cols * nrows));
    float *imatrix = calloc((size_t)pad_cols, sizeof(float));
    for (int64_t r = 0; r < nrows; r++)
        for (int64_t i = 0; i < real_cols; i++)
            src[r * pad_cols + i] = rng_gauss();
    for (int64_t i = 0; i < real_cols; i++) imatrix[i] = 1.0f;
    /* padding columns get zero importance, as the converter will do */

    size_t row_size = g4q_row_size(G4Q_TYPE_Q2_K, pad_cols);
    uint8_t *q = malloc(row_size * (size_t)nrows);
    g4q_quantize_init(G4Q_TYPE_Q2_K);
    g4q_quantize_chunk(G4Q_TYPE_Q2_K, src, q, 0, nrows, pad_cols, imatrix);
    for (int64_t r = 0; r < nrows; r++)
        g4q_dequant_row(G4Q_TYPE_Q2_K, q + (size_t)r * row_size,
                        out + r * pad_cols, pad_cols);

    int nonzero_pad = 0;
    for (int64_t r = 0; r < nrows; r++)
        for (int64_t i = real_cols; i < pad_cols; i++)
            if (out[r * pad_cols + i] != 0.0f) nonzero_pad++;
    CHECK(nonzero_pad == 0, "q2_k padding: %d non-zero padded values", nonzero_pad);

    double err = 0.0, den = 0.0;
    for (int64_t r = 0; r < nrows; r++)
        for (int64_t i = 0; i < real_cols; i++) {
            double d = (double)out[r * pad_cols + i] - (double)src[r * pad_cols + i];
            err += d * d;
            den += (double)src[r * pad_cols + i] * (double)src[r * pad_cols + i];
        }
    err = sqrt(err / den);
    CHECK(err < 0.35, "q2_k padding: real-value rel err %.4f", err);
    printf("  q2_K down-pad 704->768: zero tail exact, real rel-rms-err %.4f\n", err);

    free(src);
    free(out);
    free(q);
    free(imatrix);
}

static void test_gguf_roundtrip(const char *path) {
    g4_gguf_writer *w = g4_gguf_writer_new();
    g4_gguf_writer_kv_u32(w, "gemma4.block_count", 30);
    g4_gguf_writer_kv_u64(w, "gemma4.context_length", 262144);
    g4_gguf_writer_kv_f32(w, "gemma4.final_logit_softcap", 30.0f);
    g4_gguf_writer_kv_bool(w, "gemma4.attention.k_eq_v_global", true);
    g4_gguf_writer_kv_str(w, "general.architecture", "gemma4");
    const int32_t eos[] = { 1, 106 };
    g4_gguf_writer_kv_arr_i32(w, "gemma4.eos_token_ids", eos, 2);
    const float rope[] = { 10000.0f, 1000000.0f };
    g4_gguf_writer_kv_arr_f32(w, "gemma4.rope.freq_bases", rope, 2);
    const char *toks[] = { "<pad>", "<eos>", "<bos>", "<|turn>" };
    g4_gguf_writer_kv_arr_str(w, "tokenizer.g4.tokens", toks, 4);

    float t0[64 * 3];
    for (int i = 0; i < 64 * 3; i++) t0[i] = (float)i * 0.25f;
    uint64_t d0[2] = { 64, 3 };
    CHECK(g4_gguf_writer_tensor(w, "test.f32", G4Q_TYPE_F32, 2, d0, t0,
                                sizeof(t0)) == 0, "add f32 tensor");

    float q8src[128];
    for (int i = 0; i < 128; i++) q8src[i] = rng_gauss();
    size_t q8row = g4q_row_size(G4Q_TYPE_Q8_0, 64);
    uint8_t *q8 = malloc(q8row * 2);
    g4q_quantize_chunk(G4Q_TYPE_Q8_0, q8src, q8, 0, 2, 64, NULL);
    uint64_t d1[2] = { 64, 2 };
    CHECK(g4_gguf_writer_tensor(w, "test.q8_0", G4Q_TYPE_Q8_0, 2, d1, q8,
                                q8row * 2) == 0, "add q8_0 tensor");

    char err[256] = "";
    CHECK(g4_gguf_writer_finish(w, path, err, sizeof(err)) == 0,
          "writer finish: %s", err);
    g4_gguf_writer_free(w);

    g4_gguf g;
    CHECK(g4_gguf_open(&g, path, err, sizeof(err)) == 0, "open: %s", err);
    CHECK(g.n_kv == 8, "kv count %llu", (unsigned long long)g.n_kv);
    CHECK(g.n_tensor == 2, "tensor count %llu", (unsigned long long)g.n_tensor);

    const g4_gguf_kv *kv = g4_gguf_get(&g, "gemma4.block_count");
    CHECK(kv && kv->v.u64 == 30, "block_count");
    kv = g4_gguf_get(&g, "gemma4.context_length");
    CHECK(kv && kv->v.u64 == 262144, "context_length");
    kv = g4_gguf_get(&g, "gemma4.final_logit_softcap");
    CHECK(kv && kv->v.f64 == 30.0, "softcap");
    kv = g4_gguf_get(&g, "gemma4.attention.k_eq_v_global");
    CHECK(kv && kv->v.b, "k_eq_v");
    kv = g4_gguf_get(&g, "general.architecture");
    CHECK(kv && kv->str && !strcmp(kv->str, "gemma4"), "architecture");
    kv = g4_gguf_get(&g, "gemma4.eos_token_ids");
    CHECK(kv && kv->count == 2 && ((const int32_t *)kv->arr)[1] == 106, "eos ids");
    kv = g4_gguf_get(&g, "gemma4.rope.freq_bases");
    CHECK(kv && kv->count == 2 && ((const float *)kv->arr)[1] == 1000000.0f, "rope");
    kv = g4_gguf_get(&g, "tokenizer.g4.tokens");
    CHECK(kv && kv->count == 4 &&
          !strcmp(((char **)kv->arr)[3], "<|turn>"), "token array");

    const g4_gguf_tensor *t = g4_gguf_tensor_by_name(&g, "test.f32");
    CHECK(t && t->nbytes == sizeof(t0) && t->dims[0] == 64, "f32 tensor info");
    if (t) {
        float *back = malloc((size_t)t->nbytes);
        CHECK(g4_gguf_read_tensor_data(&g, t, back) == 0, "read f32 tensor");
        CHECK(memcmp(back, t0, sizeof(t0)) == 0, "f32 tensor bytes");
        free(back);
    }
    t = g4_gguf_tensor_by_name(&g, "test.q8_0");
    CHECK(t && t->nbytes == q8row * 2, "q8_0 tensor info");
    CHECK(t && t->offset % G4_GGUF_ALIGN == 0, "q8_0 tensor alignment");
    if (t) {
        uint8_t *back = malloc((size_t)t->nbytes);
        CHECK(g4_gguf_read_tensor_data(&g, t, back) == 0, "read q8_0 tensor");
        CHECK(memcmp(back, q8, q8row * 2) == 0, "q8_0 tensor bytes");
        free(back);
    }

    g4_gguf_close(&g);
    free(q8);
    remove(path);
    printf("  gguf round-trip: 8 kv + 2 tensors OK\n");
}

static void test_kvstore(void) {
    char sha[41];
    g4_kvstore_sha1_hex("abc", 3, sha);
    CHECK(!strcmp(sha, "a9993e364706816aba3e25717850c26c9cd0d89d"),
          "sha1(\"abc\") = %s", sha);

    g4_kvstore_entry e = {0};
    e.quant_bits = 2;
    e.reason = G4_KVSTORE_REASON_COLD;
    e.ext_flags = G4_KVSTORE_EXT_TOOL_MAP;
    e.model_id = G4_KVSTORE_MODEL_ID_GEMMA4_26B_A4B;
    e.tokens = 30000;
    e.hits = 7;
    e.ctx_size = 262144;
    e.created_at = 1700000000ull;
    e.last_used = 1700003600ull;
    e.payload_bytes = 800ull << 20;

    uint8_t h[G4_KVSTORE_FIXED_HEADER];
    g4_kvstore_fill_header(h, &e);
    g4_kvstore_entry back;
    CHECK(g4_kvstore_parse_header(h, &back), "parse header");
    CHECK(back.quant_bits == 2 && back.reason == G4_KVSTORE_REASON_COLD &&
          back.ext_flags == G4_KVSTORE_EXT_TOOL_MAP && back.model_id == 1 &&
          back.tokens == 30000 && back.hits == 7 &&
          back.ctx_size == 262144 && back.created_at == e.created_at &&
          back.last_used == e.last_used &&
          back.payload_bytes == e.payload_bytes,
          "header field round-trip");

    uint8_t bad[G4_KVSTORE_FIXED_HEADER];
    memcpy(bad, h, sizeof(bad));
    bad[0] = 'X';
    CHECK(!g4_kvstore_parse_header(bad, &back), "reject bad magic");
    memcpy(bad, h, sizeof(bad));
    bad[7] = 0;
    CHECK(!g4_kvstore_parse_header(bad, &back), "reject model_id 0");

    /* Eviction score ordering. */
    g4_kvstore_entry a = e, b = e;
    a.file_size = 800ull << 20;
    b.file_size = 800ull << 20;
    uint64_t now = e.last_used + 60;
    b.hits = 0;
    CHECK(g4_kvstore_eviction_score(&a, now) > g4_kvstore_eviction_score(&b, now),
          "more hits score higher");
    b = a;
    b.last_used = e.last_used - 48 * 3600;
    b.created_at = b.last_used;
    CHECK(g4_kvstore_eviction_score(&a, now) > g4_kvstore_eviction_score(&b, now),
          "stale hits decay");
    b = a;
    b.reason = G4_KVSTORE_REASON_CONTINUED;
    CHECK(g4_kvstore_eviction_score(&a, now) > g4_kvstore_eviction_score(&b, now),
          "anchor reasons boosted");
    printf("  kvstore: sha1, header round-trip, eviction ordering OK\n");
}

int main(void) {
    printf("g4_test (M0)\n");

    printf("f16/bf16 conversions\n");
    test_f16_bf16();

    printf("quant round-trips (gaussian data)\n");
    /* 2816 = gate/up contraction; 768 = padded down contraction.
     * Tolerances are calibrated for pure gaussian data, which quantizes
     * worse than real LLM weight rows (no heavy tails, flat importance):
     * expected rel-rms is ~0.005 q8_0, ~0.07 q4_K, ~0.29 q2_K, ~0.34
     * iq2_xxs. */
    test_quant_roundtrip(G4Q_TYPE_Q8_0, 2816, 8, false, 0.01);
    test_quant_roundtrip(G4Q_TYPE_Q4_K, 2816, 8, false, 0.10);
    test_quant_roundtrip(G4Q_TYPE_Q4_K, 2816, 8, true, 0.10);
    test_quant_roundtrip(G4Q_TYPE_Q2_K, 2816, 8, false, 0.35);
    test_quant_roundtrip(G4Q_TYPE_Q2_K, 768, 8, true, 0.35);
    test_quant_roundtrip(G4Q_TYPE_IQ2_XXS, 2816, 8, true, 0.55);
    test_q2k_down_padding();

    printf("gguf container\n");
    test_gguf_roundtrip("/tmp/g4_test_roundtrip.gguf");

    printf("kvstore container\n");
    test_kvstore();

    if (g_failures) {
        printf("FAILED: %d check(s)\n", g_failures);
        return 1;
    }
    printf("OK: all checks passed\n");
    return 0;
}
