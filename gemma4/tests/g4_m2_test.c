/* M2 gate: run the converted model and compare against transformers vectors.
 *
 * Reads tests/vectors/real_vectors.bin (produced by scripts/gen_logit_vectors.py
 * from the real bf16 model) and, for each prompt, runs the g4 engine prefill
 * and checks: (1) greedy argmax matches at every position; (2) the top-K
 * logprobs are within tolerance.  A q8 build should match closely; looser
 * tolerances apply to more aggressive quants.
 *
 * Usage: g4_m2_test MODEL.gguf [vectors.bin] [max_logprob_delta]
 */

#include "../g4.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

int main(int argc, char **argv) {
    if (argc < 2) {
        fprintf(stderr, "usage: g4_m2_test MODEL.gguf [vectors.bin] [max_delta]\n");
        return 2;
    }
    const char *model_path = argv[1];
    const char *vec_path = argc > 2 ? argv[2] : "tests/vectors/real_vectors.bin";
    double max_delta = argc > 3 ? atof(argv[3]) : 0.1;

    FILE *fp = fopen(vec_path, "rb");
    if (!fp) {
        fprintf(stderr, "cannot open %s (run scripts/gen_logit_vectors.py first)\n",
                vec_path);
        return 2;
    }
    char magic[4];
    uint32_t version, n_prompts, top_k;
    if (fread(magic, 1, 4, fp) != 4 || memcmp(magic, "G4RV", 4) ||
        fread(&version, 4, 1, fp) != 1 || fread(&n_prompts, 4, 1, fp) != 1 ||
        fread(&top_k, 4, 1, fp) != 1)
    {
        fprintf(stderr, "bad vectors file\n");
        return 2;
    }

    char err[256] = "";
    g4_model *m = g4_model_load_mmap(model_path, err, sizeof(err));
    if (!m) { fprintf(stderr, "model load: %s\n", err); return 2; }
    const uint32_t vocab = g4_model_vocab(m);

    uint32_t total_pos = 0, argmax_hits = 0;
    double sum_delta = 0.0, worst_delta = 0.0;
    int failed = 0;

    for (uint32_t pi = 0; pi < n_prompts; pi++) {
        uint32_t n_tok;
        if (fread(&n_tok, 4, 1, fp) != 1) { fprintf(stderr, "truncated\n"); return 2; }
        int32_t *ids = malloc(sizeof(int32_t) * n_tok);
        if (fread(ids, 4, n_tok, fp) != n_tok) { fprintf(stderr, "truncated\n"); return 2; }

        float *logits = malloc(sizeof(float) * (size_t)n_tok * vocab);
        if (g4_forward_prefill_f32(m, ids, n_tok, logits)) {
            fprintf(stderr, "forward failed\n");
            return 2;
        }

        uint32_t prompt_hits = 0;
        for (uint32_t t = 0; t < n_tok; t++) {
            /* our logprobs at this position */
            const float *lg = logits + (size_t)t * vocab;
            double mx = lg[0];
            for (uint32_t v = 1; v < vocab; v++) if (lg[v] > mx) mx = lg[v];
            double se = 0.0;
            for (uint32_t v = 0; v < vocab; v++) se += exp((double)lg[v] - mx);
            double lse = mx + log(se);
            uint32_t our_argmax = 0;
            for (uint32_t v = 1; v < vocab; v++) if (lg[v] > lg[our_argmax]) our_argmax = v;

            uint32_t ref_argmax = 0;
            for (uint32_t k = 0; k < top_k; k++) {
                int32_t rid;
                float rlp;
                if (fread(&rid, 4, 1, fp) != 1 || fread(&rlp, 4, 1, fp) != 1)
                    { fprintf(stderr, "truncated\n"); return 2; }
                if (k == 0) ref_argmax = (uint32_t)rid;
                if (rid >= 0 && (uint32_t)rid < vocab) {
                    double our_lp = (double)lg[rid] - lse;
                    double d = fabs(our_lp - (double)rlp);
                    sum_delta += d;
                    if (d > worst_delta) worst_delta = d;
                }
            }
            total_pos++;
            if (our_argmax == ref_argmax) { argmax_hits++; prompt_hits++; }
        }
        printf("  prompt %u: %u/%u argmax match\n", pi, prompt_hits, n_tok);
        free(ids);
        free(logits);
    }
    fclose(fp);

    double mean_delta = total_pos ? sum_delta / ((double)total_pos * top_k) : 0.0;
    printf("M2: argmax %u/%u, mean top-%u logprob delta %.4f (worst %.4f, max %.4f)\n",
           argmax_hits, total_pos, top_k, mean_delta, worst_delta, max_delta);
    if (argmax_hits != total_pos || mean_delta > max_delta) {
        printf("FAILED: engine diverges from transformers reference\n");
        failed = 1;
    } else {
        printf("OK: converted model matches the transformers reference\n");
    }
    g4_model_free(m);
    return failed;
}
