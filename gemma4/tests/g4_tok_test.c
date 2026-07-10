/* M3 validation: encode the committed vectors with the C tokenizer and
 * compare with the HF `tokenizers` reference output. */

#include "../g4_tokenizer.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

int main(int argc, char **argv) {
    const char *table = argc > 1 ? argv[1] : "tests/vectors/tok_table.bin";
    const char *vecs = argc > 2 ? argv[2] : "tests/vectors/tok_vectors.bin";
    char err[256] = "";

    g4_tokenizer *t = g4_tokenizer_load_blob(table, err, sizeof(err));
    if (!t) { fprintf(stderr, "load: %s\n", err); return 1; }

    FILE *fp = fopen(vecs, "rb");
    if (!fp) { fprintf(stderr, "cannot open %s\n", vecs); return 1; }
    uint32_t n_cases;
    if (fread(&n_cases, 4, 1, fp) != 1) return 1;

    int failed = 0;
    for (uint32_t c = 0; c < n_cases; c++) {
        uint32_t tlen, nids;
        if (fread(&tlen, 4, 1, fp) != 1) return 1;
        char *text = malloc(tlen + 1);
        if (tlen && fread(text, 1, tlen, fp) != tlen) return 1;
        text[tlen] = 0;
        if (fread(&nids, 4, 1, fp) != 1) return 1;
        int32_t *want = malloc(sizeof(int32_t) * (nids ? nids : 1));
        if (nids && fread(want, 4, nids, fp) != nids) return 1;

        int32_t got[512];
        uint32_t ng = g4_tokenizer_encode(t, text, tlen, got, 512);
        bool ok = ng == nids && (ng == 0 || !memcmp(got, want, 4 * ng));
        if (!ok) {
            failed++;
            fprintf(stderr, "FAIL case %u: %.60s\n  want (%u):", c, text, nids);
            for (uint32_t i = 0; i < nids && i < 16; i++)
                fprintf(stderr, " %d", want[i]);
            fprintf(stderr, "\n  got  (%u):", ng);
            for (uint32_t i = 0; i < ng && i < 16; i++)
                fprintf(stderr, " %d", got[i]);
            fprintf(stderr, "\n");
        }
        free(text);
        free(want);
    }
    fclose(fp);
    printf("tokenizer: %u/%u cases match\n", n_cases - failed, n_cases);
    g4_tokenizer_free(t);
    if (failed) { printf("FAILED\n"); return 1; }
    printf("OK: tokenizer matches the HF reference\n");
    return 0;
}
