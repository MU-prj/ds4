/* g4-run — minimal generation CLI for the g4 engine.
 *
 * Loads a g4 GGUF (frugal mmap mode by default, so the real 26B model runs
 * on a machine that cannot hold it in RAM) and the G4TK tokenizer blob,
 * encodes a prompt, and greedily decodes tokens, printing detokenized text.
 * This is the end-to-end smoke test for a converted model: coherent output
 * means the conversion + graph are correct.
 *
 * Usage:
 *   tools/g4-run -m MODEL.gguf -t tok_table.bin -p "prompt" [-n N]
 *                [--ram] [--temp T] [--top-k K] [--seed S]
 */

#include "../g4.h"
#include "../g4_tokenizer.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

static int argmax(const float *v, uint32_t n) {
    int best = 0;
    for (uint32_t i = 1; i < n; i++) if (v[i] > v[best]) best = (int)i;
    return best;
}

/* Temperature + top-k sampling (xorshift RNG). */
static int sample(const float *logits, uint32_t n, float temp, int top_k,
                  uint64_t *rng) {
    if (temp <= 0.0f) return argmax(logits, n);
    /* crude top-k: find k best by repeated scan (k is small). */
    if (top_k <= 0 || (uint32_t)top_k > n) top_k = (int)n;
    int *idx = malloc(sizeof(int) * top_k);
    float *val = malloc(sizeof(float) * top_k);
    for (int j = 0; j < top_k; j++) {
        int best = -1;
        float bv = -1e30f;
        for (uint32_t i = 0; i < n; i++) {
            bool taken = false;
            for (int t = 0; t < j; t++) if (idx[t] == (int)i) { taken = true; break; }
            if (!taken && logits[i] > bv) { bv = logits[i]; best = (int)i; }
        }
        idx[j] = best;
        val[j] = bv;
    }
    double sum = 0.0;
    for (int j = 0; j < top_k; j++) { val[j] = expf(val[j] / temp); sum += val[j]; }
    *rng ^= *rng << 13; *rng ^= *rng >> 7; *rng ^= *rng << 17;
    double r = ((double)(*rng >> 11) / 9007199254740992.0) * sum;
    int pick = idx[top_k - 1];
    double acc = 0.0;
    for (int j = 0; j < top_k; j++) { acc += val[j]; if (r <= acc) { pick = idx[j]; break; } }
    free(idx); free(val);
    return pick;
}

int main(int argc, char **argv) {
    const char *model_path = NULL, *tok_path = NULL, *prompt = "Ciao";
    int n_predict = 64, top_k = 40;
    float temp = 0.0f;
    bool ram = false;
    uint64_t seed = 1234;

    for (int i = 1; i < argc; i++) {
        if (!strcmp(argv[i], "-m") && i + 1 < argc) model_path = argv[++i];
        else if (!strcmp(argv[i], "-t") && i + 1 < argc) tok_path = argv[++i];
        else if (!strcmp(argv[i], "-p") && i + 1 < argc) prompt = argv[++i];
        else if (!strcmp(argv[i], "-n") && i + 1 < argc) n_predict = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--temp") && i + 1 < argc) temp = atof(argv[++i]);
        else if (!strcmp(argv[i], "--top-k") && i + 1 < argc) top_k = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--seed") && i + 1 < argc) seed = strtoull(argv[++i], NULL, 10);
        else if (!strcmp(argv[i], "--ram")) ram = true;
        else { fprintf(stderr, "unknown arg %s\n", argv[i]); return 1; }
    }
    if (!model_path || !tok_path) {
        fprintf(stderr, "usage: g4-run -m MODEL.gguf -t tok_table.bin "
                        "-p \"prompt\" [-n N] [--ram] [--temp T] [--top-k K]\n");
        return 1;
    }

    char err[256] = "";
    fprintf(stderr, "loading model (%s)...\n", ram ? "RAM" : "frugal mmap");
    double t0 = (double)clock() / CLOCKS_PER_SEC;
    g4_model *m = ram ? g4_model_load(model_path, err, sizeof(err))
                      : g4_model_load_mmap(model_path, err, sizeof(err));
    if (!m) { fprintf(stderr, "model load failed: %s\n", err); return 1; }

    g4_tokenizer *tok = g4_tokenizer_load_blob(tok_path, err, sizeof(err));
    if (!tok) { fprintf(stderr, "tokenizer load failed: %s\n", err); return 1; }
    fprintf(stderr, "loaded in %.1fs. vocab %u, %u layers.\n",
            (double)clock() / CLOCKS_PER_SEC - t0, g4_model_vocab(m),
            g4_model_layers(m));

    /* BOS (2) + prompt tokens. */
    int32_t ids[4096];
    ids[0] = 2;
    uint32_t n_prompt = 1 + g4_tokenizer_encode(tok, prompt, strlen(prompt),
                                                ids + 1, 4095);
    if (n_prompt > 4096) n_prompt = 4096;

    uint32_t ctx = n_prompt + (uint32_t)n_predict + 8;
    g4_session *s = g4_session_create(m, ctx);

    fprintf(stderr, "prompt: %u tokens. prefill (skips logits until last)...\n",
            n_prompt);
    printf("%s", prompt);
    fflush(stdout);

    double tpre = (double)clock() / CLOCKS_PER_SEC;
    for (uint32_t p = 0; p < n_prompt; p++) {
        /* Only the last prompt token needs logits to seed generation. */
        bool want = (p + 1 == n_prompt);
        if (g4_session_eval_ex(s, ids[p], want)) {
            fprintf(stderr, "\neval failed\n"); return 1;
        }
        fprintf(stderr, "\r  prefill %u/%u (%.1fs)", p + 1, n_prompt,
                (double)clock() / CLOCKS_PER_SEC - tpre);
    }
    fprintf(stderr, "\ngenerating (a dot per token):\n");

    uint64_t rng = seed;
    int eos_a = 1, eos_b = 106;
    int generated = 0;
    const char *stop = "reached -n limit";
    double tgen = (double)clock() / CLOCKS_PER_SEC;
    for (int i = 0; i < n_predict; i++) {
        int next = sample(g4_session_logits(s), g4_model_vocab(m), temp, top_k, &rng);
        if (next == eos_a || next == eos_b) { stop = "EOS"; break; }
        uint32_t len;
        const char *txt = g4_tokenizer_token_text(tok, (uint32_t)next, &len);
        if (txt) {
            for (uint32_t c = 0; c < len; c++) {
                if (c + 2 < len && (uint8_t)txt[c] == 0xe2 &&
                    (uint8_t)txt[c+1] == 0x96 && (uint8_t)txt[c+2] == 0x81) {
                    putchar(' '); c += 2;
                } else putchar(txt[c]);
            }
            fflush(stdout);
        }
        fputc('.', stderr);  /* heartbeat: one dot per generated token */
        generated++;
        if (g4_session_eval(s, next)) { fprintf(stderr, "\neval failed\n"); return 1; }
    }
    double dt = (double)clock() / CLOCKS_PER_SEC - tgen;
    printf("\n\n");
    fprintf(stderr, "\ndone: %d tokens, stop=%s, %.1fs (%.2f tok/s CPU)\n",
            generated, stop, dt, generated / (dt > 0 ? dt : 1));

    g4_session_free(s);
    g4_tokenizer_free(tok);
    g4_model_free(m);
    return 0;
}
