#ifndef G4_TOKENIZER_H
#define G4_TOKENIZER_H

/* Gemma 4 tokenizer: SentencePiece-style BPE (" "->U+2581 normalizer,
 * rank-ordered merges, byte fallback, added/special tokens matched before
 * BPE).  M3 scope: encode; tables come from the G4TK blob built by
 * scripts/gen_tokenizer_vectors.py (later: embedded in the GGUF). */

#include <stdbool.h>
#include <stdint.h>
#include <stddef.h>

typedef struct g4_tokenizer g4_tokenizer;

g4_tokenizer *g4_tokenizer_load_blob(const char *path, char *err, size_t errlen);
void g4_tokenizer_free(g4_tokenizer *t);

uint32_t g4_tokenizer_vocab_size(const g4_tokenizer *t);
/* Token text (UTF-8, U+2581 for spaces); NULL for out-of-range ids. */
const char *g4_tokenizer_token_text(const g4_tokenizer *t, uint32_t id,
                                    uint32_t *len);

/* Encodes UTF-8 text without adding BOS/EOS.  Returns the token count and
 * writes up to `cap` ids; call with cap=0 to size.  Added/special tokens in
 * the text are matched greedily before BPE, like the reference tokenizer. */
uint32_t g4_tokenizer_encode(const g4_tokenizer *t, const char *text,
                             size_t text_len, int32_t *ids, uint32_t cap);

#endif
