#ifndef G4_H
#define G4_H

/* g4 engine public API.
 *
 * M1: model load from a g4 GGUF + f32 reference forward.
 * Session API: incremental decode with persistent KV state (ring buffer of
 * `sliding_window` rows on local layers, full history on global layers) and
 * the G4SP disk payload (design: gemma4-port/03-disk-kv-cache.md §5).
 * Quantized fast paths land with M2+. */

#include <stdbool.h>
#include <stdint.h>
#include <stddef.h>
#include <stdio.h>

typedef struct g4_model g4_model;
typedef struct g4_session g4_session;

g4_model *g4_model_load(const char *path, char *err, size_t errlen);
void g4_model_free(g4_model *m);

uint32_t g4_model_vocab(const g4_model *m);
uint32_t g4_model_layers(const g4_model *m);

/* Sessions.  `ctx` bounds the token history (global-layer KV rows). */
g4_session *g4_session_create(const g4_model *m, uint32_t ctx);
void g4_session_free(g4_session *s);
uint32_t g4_session_pos(const g4_session *s);
const int32_t *g4_session_tokens(const g4_session *s);
/* Next-token logits after the last eval (vocab floats). */
const float *g4_session_logits(const g4_session *s);
/* Evaluates one token at the current position.  Returns 0 on success. */
int g4_session_eval(g4_session *s, int32_t token);

/* G4SP payload (gemma4-port/03 §5).  kv dtype: 0=bf16, 1=f32; this f32
 * reference engine writes and expects dtype 1, so save->load->continue is
 * bit-exact.  Load fails if the payload shape does not match the model. */
#define G4_SESSION_PAYLOAD_VERSION 1u
#define G4_SESSION_PAYLOAD_U32_FIELDS 16u
uint64_t g4_session_payload_bytes(const g4_session *s);
int g4_session_save_payload(const g4_session *s, FILE *fp,
                            char *err, size_t errlen);
int g4_session_load_payload(g4_session *s, FILE *fp, char *err, size_t errlen);

/* Runs a full prefill over `tokens[0..n_tokens)` and writes the logits of
 * every position into `logits` (n_tokens * vocab floats, position-major).
 * Implemented on top of the session API, so it exercises the same ring
 * buffer path as incremental decode.  Returns 0 on success. */
int g4_forward_prefill_f32(const g4_model *m, const int32_t *tokens,
                           uint32_t n_tokens, float *logits);

#endif
