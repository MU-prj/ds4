/* Session + G4SP payload validation on the toy model:
 *  1. incremental eval must equal the batch prefill (ring buffer path);
 *  2. save at an arbitrary mid-point -> load into a fresh session ->
 *     continue: final logits must be bit-exact vs the uninterrupted run;
 *  3. corrupted/mismatched payload headers must be rejected. */

#include "../g4.h"

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

int main(int argc, char **argv) {
    const char *path = argc > 1 ? argv[1] : "tests/vectors/toy.gguf";
    const char *payload_path = "/tmp/g4_session_test.g4sp";
    char err[256] = "";

    g4_model *m = g4_model_load(path, err, sizeof(err));
    if (!m) {
        fprintf(stderr, "model load failed: %s\n", err);
        return 1;
    }
    const uint32_t vocab = g4_model_vocab(m);

    /* The toy reference token sequence lives in the GGUF; reuse the prefill
     * wrapper (already validated against JAX) as the ground truth. */
    int32_t tokens[64];
    uint32_t n_tokens = 21;
    for (uint32_t i = 0; i < n_tokens; i++) tokens[i] = (int32_t)(i * 7 % 97);

    float *ref = malloc(sizeof(float) * n_tokens * vocab);
    CHECK(g4_forward_prefill_f32(m, tokens, n_tokens, ref) == 0, "prefill");

    /* 1. Incremental session equals batch prefill. */
    g4_session *a = g4_session_create(m, n_tokens);
    for (uint32_t p = 0; p < n_tokens; p++) {
        CHECK(g4_session_eval(a, tokens[p]) == 0, "eval pos %u", p);
        CHECK(!memcmp(g4_session_logits(a), ref + (size_t)p * vocab,
                      sizeof(float) * vocab),
              "incremental logits differ at pos %u", p);
    }
    CHECK(g4_session_pos(a) == n_tokens, "session pos");

    /* 2. Save at cut, resume in a fresh session, continue. The cut is past
     * the sliding window (8) so the ring has already wrapped. */
    const uint32_t cut = 12;
    g4_session *b = g4_session_create(m, n_tokens);
    for (uint32_t p = 0; p < cut; p++) CHECK(g4_session_eval(b, tokens[p]) == 0, "b eval");
    FILE *fp = fopen(payload_path, "wb");
    CHECK(fp != NULL, "open payload for write");
    uint64_t want_bytes = g4_session_payload_bytes(b);
    CHECK(g4_session_save_payload(b, fp, err, sizeof(err)) == 0, "save: %s", err);
    long wrote = ftell(fp);
    CHECK(wrote > 0 && (uint64_t)wrote == want_bytes,
          "payload size %ld != declared %llu", wrote,
          (unsigned long long)want_bytes);
    fclose(fp);
    g4_session_free(b);

    g4_session *c = g4_session_create(m, n_tokens);
    fp = fopen(payload_path, "rb");
    CHECK(fp != NULL, "open payload for read");
    CHECK(g4_session_load_payload(c, fp, err, sizeof(err)) == 0, "load: %s", err);
    fclose(fp);
    CHECK(g4_session_pos(c) == cut, "resumed pos %u", g4_session_pos(c));
    CHECK(!memcmp(g4_session_tokens(c), tokens, sizeof(int32_t) * cut),
          "resumed token history");
    CHECK(!memcmp(g4_session_logits(c), ref + (size_t)(cut - 1) * vocab,
                  sizeof(float) * vocab),
          "resumed logits differ (saved next-token distribution)");
    for (uint32_t p = cut; p < n_tokens; p++) {
        CHECK(g4_session_eval(c, tokens[p]) == 0, "c eval pos %u", p);
        CHECK(!memcmp(g4_session_logits(c), ref + (size_t)p * vocab,
                      sizeof(float) * vocab),
              "post-resume logits differ at pos %u (bit-exact expected)", p);
    }
    g4_session_free(c);

    /* 3. Rejection paths. */
    fp = fopen(payload_path, "r+b");
    uint32_t bad = 999;
    fseek(fp, 6 * 4, SEEK_SET); /* sliding_window field */
    CHECK(fwrite(&bad, 4, 1, fp) == 1, "corrupt payload");
    fclose(fp);
    g4_session *d = g4_session_create(m, n_tokens);
    fp = fopen(payload_path, "rb");
    CHECK(g4_session_load_payload(d, fp, err, sizeof(err)) != 0,
          "mismatched shape must be rejected");
    fclose(fp);
    /* Payload longer than the target session context. */
    g4_session *e = g4_session_create(m, 4);
    fp = fopen(payload_path, "rb");
    CHECK(g4_session_load_payload(e, fp, err, sizeof(err)) != 0,
          "payload longer than ctx must be rejected");
    fclose(fp);
    g4_session_free(d);
    g4_session_free(e);
    g4_session_free(a);

    remove(payload_path);
    g4_model_free(m);
    free(ref);

    if (g_failures) {
        printf("FAILED: %d check(s)\n", g_failures);
        return 1;
    }
    printf("session: incremental==batch, save@%u/resume bit-exact, "
           "rejects OK\n", cut);
    printf("OK: session and G4SP payload behave per gemma4-port/03\n");
    return 0;
}
