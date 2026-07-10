#ifndef G4_H
#define G4_H

/* g4 engine public API.  M1 scope: model load from a g4 GGUF and an f32
 * reference forward pass (prefill) that returns per-position logits.
 * The quantized fast paths and the session/KV API land with M2+. */

#include <stdbool.h>
#include <stdint.h>
#include <stddef.h>

typedef struct g4_model g4_model;

g4_model *g4_model_load(const char *path, char *err, size_t errlen);
void g4_model_free(g4_model *m);

uint32_t g4_model_vocab(const g4_model *m);
uint32_t g4_model_layers(const g4_model *m);

/* Runs a full prefill over `tokens[0..n_tokens)` and writes the logits of
 * every position into `logits` (n_tokens * vocab floats, position-major).
 * Returns 0 on success. */
int g4_forward_prefill_f32(const g4_model *m, const int32_t *tokens,
                           uint32_t n_tokens, float *logits);

#endif
