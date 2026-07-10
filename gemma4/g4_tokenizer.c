#include "g4_tokenizer.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/* ------------------------------------------------------------------ */
/* Structures                                                          */
/* ------------------------------------------------------------------ */

typedef struct {
    uint32_t off;
    uint32_t len;
} g4_tok_span;

struct g4_tokenizer {
    uint32_t vocab_size, n_merges, n_added;
    char *strings;          /* concatenated token texts */
    g4_tok_span *tok;       /* [vocab_size] */
    uint32_t *added;        /* [n_added] token ids */
    int32_t byte_id[256];   /* <0xNN> ids, -1 if absent */
    /* vocab hash: string -> id (open addressing, power of two). */
    int32_t *vhash;
    uint32_t vhash_size;
    /* merge hash: (left,right) -> rank,result. */
    uint64_t *mkey;
    uint32_t *mrank;
    uint32_t *mres;
    uint32_t mhash_size;
};

static uint64_t fnv1a(const char *s, size_t n) {
    uint64_t h = 1469598103934665603ull;
    for (size_t i = 0; i < n; i++) {
        h ^= (uint8_t)s[i];
        h *= 1099511628211ull;
    }
    return h;
}

static int32_t vocab_lookup(const g4_tokenizer *t, const char *s, size_t n) {
    uint64_t h = fnv1a(s, n) & (t->vhash_size - 1);
    while (t->vhash[h] >= 0) {
        const g4_tok_span *sp = &t->tok[t->vhash[h]];
        if (sp->len == n && !memcmp(t->strings + sp->off, s, n))
            return t->vhash[h];
        h = (h + 1) & (t->vhash_size - 1);
    }
    return -1;
}

static bool merge_lookup(const g4_tokenizer *t, uint32_t l, uint32_t r,
                         uint32_t *rank, uint32_t *res) {
    uint64_t key = ((uint64_t)l << 32) | r;
    uint64_t h = (key * 0x9e3779b97f4a7c15ull) & (t->mhash_size - 1);
    while (t->mkey[h] != UINT64_MAX) {
        if (t->mkey[h] == key) {
            *rank = t->mrank[h];
            *res = t->mres[h];
            return true;
        }
        h = (h + 1) & (t->mhash_size - 1);
    }
    return false;
}

/* ------------------------------------------------------------------ */
/* Loading                                                             */
/* ------------------------------------------------------------------ */

static uint32_t next_pow2(uint32_t v) {
    uint32_t p = 1;
    while (p < v) p <<= 1;
    return p;
}

void g4_tokenizer_free(g4_tokenizer *t) {
    if (!t) return;
    free(t->strings); free(t->tok); free(t->added);
    free(t->vhash); free(t->mkey); free(t->mrank); free(t->mres);
    free(t);
}

g4_tokenizer *g4_tokenizer_load_blob(const char *path, char *err, size_t errlen) {
#define FAIL(msg) do { \
        if (err && errlen) snprintf(err, errlen, "%s", msg); \
        if (fp) fclose(fp); \
        g4_tokenizer_free(t); \
        free(buf); \
        return NULL; \
    } while (0)
    g4_tokenizer *t = NULL;
    uint8_t *buf = NULL;
    FILE *fp = fopen(path, "rb");
    if (!fp) { FILE *fp2 = NULL; fp = fp2; FAIL("cannot open tokenizer blob"); }
    fseek(fp, 0, SEEK_END);
    long fsz = ftell(fp);
    fseek(fp, 0, SEEK_SET);
    if (fsz < 20) FAIL("blob too small");
    buf = malloc((size_t)fsz);
    if (!buf || fread(buf, 1, (size_t)fsz, fp) != (size_t)fsz)
        FAIL("cannot read blob");
    fclose(fp); fp = NULL;
    if (memcmp(buf, "G4TK", 4)) FAIL("bad blob magic");

    t = calloc(1, sizeof(*t));
    uint32_t version;
    memcpy(&version, buf + 4, 4);
    memcpy(&t->vocab_size, buf + 8, 4);
    memcpy(&t->n_merges, buf + 12, 4);
    memcpy(&t->n_added, buf + 16, 4);
    if (version != 1) FAIL("bad blob version");

    size_t p = 20;
    t->tok = malloc(sizeof(g4_tok_span) * t->vocab_size);
    t->strings = malloc((size_t)fsz); /* upper bound */
    uint32_t soff = 0;
    for (uint32_t i = 0; i < t->vocab_size; i++) {
        if (p + 2 > (size_t)fsz) FAIL("truncated vocab");
        uint16_t len;
        memcpy(&len, buf + p, 2);
        p += 2;
        if (p + len > (size_t)fsz) FAIL("truncated vocab");
        memcpy(t->strings + soff, buf + p, len);
        t->tok[i].off = soff;
        t->tok[i].len = len;
        soff += len;
        p += len;
    }

    t->vhash_size = next_pow2(t->vocab_size * 2);
    t->vhash = malloc(sizeof(int32_t) * t->vhash_size);
    memset(t->vhash, 0xff, sizeof(int32_t) * t->vhash_size);
    for (uint32_t i = 0; i < t->vocab_size; i++) {
        if (t->tok[i].len == 0) continue;
        uint64_t h = fnv1a(t->strings + t->tok[i].off, t->tok[i].len) &
                     (t->vhash_size - 1);
        while (t->vhash[h] >= 0) h = (h + 1) & (t->vhash_size - 1);
        t->vhash[h] = (int32_t)i;
    }

    t->mhash_size = next_pow2(t->n_merges * 2);
    t->mkey = malloc(sizeof(uint64_t) * t->mhash_size);
    t->mrank = malloc(sizeof(uint32_t) * t->mhash_size);
    t->mres = malloc(sizeof(uint32_t) * t->mhash_size);
    memset(t->mkey, 0xff, sizeof(uint64_t) * t->mhash_size);
    for (uint32_t i = 0; i < t->n_merges; i++) {
        if (p + 12 > (size_t)fsz) FAIL("truncated merges");
        uint32_t l, r, res;
        memcpy(&l, buf + p, 4);
        memcpy(&r, buf + p + 4, 4);
        memcpy(&res, buf + p + 8, 4);
        p += 12;
        uint64_t key = ((uint64_t)l << 32) | r;
        uint64_t h = (key * 0x9e3779b97f4a7c15ull) & (t->mhash_size - 1);
        while (t->mkey[h] != UINT64_MAX && t->mkey[h] != key)
            h = (h + 1) & (t->mhash_size - 1);
        if (t->mkey[h] == UINT64_MAX) { /* first rank wins on duplicates */
            t->mkey[h] = key;
            t->mrank[h] = i;
            t->mres[h] = res;
        }
    }

    t->added = malloc(sizeof(uint32_t) * (t->n_added ? t->n_added : 1));
    for (uint32_t i = 0; i < t->n_added; i++) {
        if (p + 4 > (size_t)fsz) FAIL("truncated added tokens");
        memcpy(&t->added[i], buf + p, 4);
        p += 4;
    }

    for (int b = 0; b < 256; b++) {
        char name[8];
        snprintf(name, sizeof(name), "<0x%02X>", b);
        t->byte_id[b] = vocab_lookup(t, name, 6);
    }

    free(buf);
    return t;
#undef FAIL
}

uint32_t g4_tokenizer_vocab_size(const g4_tokenizer *t) { return t->vocab_size; }

const char *g4_tokenizer_token_text(const g4_tokenizer *t, uint32_t id,
                                    uint32_t *len) {
    if (id >= t->vocab_size) return NULL;
    if (len) *len = t->tok[id].len;
    return t->strings + t->tok[id].off;
}

/* ------------------------------------------------------------------ */
/* Encoding                                                            */
/* ------------------------------------------------------------------ */

static size_t utf8_char_len(uint8_t c) {
    if (c < 0x80) return 1;
    if ((c & 0xe0) == 0xc0) return 2;
    if ((c & 0xf0) == 0xe0) return 3;
    if ((c & 0xf8) == 0xf0) return 4;
    return 1; /* invalid byte: treat as single symbol, byte-fallbacked */
}

typedef struct {
    uint32_t off, len; /* span into the normalized buffer */
    int32_t id;        /* vocab id or -1 (unknown, byte fallback at flush) */
    int32_t prev, next;
} g4_sym;

static uint32_t emit(int32_t *ids, uint32_t cap, uint32_t n, int32_t id) {
    if (n < cap) ids[n] = id;
    return n + 1;
}

/* BPE over one normalized segment (no added tokens inside). */
static uint32_t bpe_segment(const g4_tokenizer *t, const char *seg, size_t n,
                            int32_t *ids, uint32_t cap, uint32_t count) {
    if (n == 0) return count;
    uint32_t max_syms = (uint32_t)n;
    g4_sym *sym = malloc(sizeof(g4_sym) * max_syms);
    uint32_t ns = 0;
    for (size_t i = 0; i < n;) {
        size_t cl = utf8_char_len((uint8_t)seg[i]);
        if (i + cl > n) cl = 1;
        sym[ns].off = (uint32_t)i;
        sym[ns].len = (uint32_t)cl;
        sym[ns].id = vocab_lookup(t, seg + i, cl);
        sym[ns].prev = (int32_t)ns - 1;
        sym[ns].next = (i + cl < n) ? (int32_t)ns + 1 : -1;
        ns++;
        i += cl;
    }

    /* Greedy lowest-rank merging.  O(n^2) rescans; fine for the M3 tests,
     * a heap-based version replaces this when long-prompt encode lands. */
    for (;;) {
        uint32_t best_rank = UINT32_MAX, best_res = 0;
        int32_t best = -1;
        for (int32_t i = 0; i >= 0 && sym[i].next >= 0; i = sym[i].next) {
            int32_t j = sym[i].next;
            if (sym[i].id < 0 || sym[j].id < 0) continue;
            uint32_t rank, res;
            if (merge_lookup(t, (uint32_t)sym[i].id, (uint32_t)sym[j].id,
                             &rank, &res) && rank < best_rank)
            {
                best_rank = rank;
                best_res = res;
                best = i;
            }
        }
        if (best < 0) break;
        int32_t j = sym[best].next;
        sym[best].len += sym[j].len;
        sym[best].id = (int32_t)best_res;
        sym[best].next = sym[j].next;
        if (sym[j].next >= 0) sym[sym[j].next].prev = best;
    }

    for (int32_t i = 0; i >= 0; i = sym[i].next) {
        if (sym[i].id >= 0) {
            count = emit(ids, cap, count, sym[i].id);
        } else {
            /* byte fallback (model.byte_fallback = true) */
            for (uint32_t b = 0; b < sym[i].len; b++) {
                uint8_t byte = (uint8_t)seg[sym[i].off + b];
                count = emit(ids, cap, count,
                             t->byte_id[byte] >= 0 ? t->byte_id[byte] : 3);
            }
        }
    }
    free(sym);
    return count;
}

uint32_t g4_tokenizer_encode(const g4_tokenizer *t, const char *text,
                             size_t text_len, int32_t *ids, uint32_t cap) {
    /* Normalizer: " " -> U+2581 (0xE2 0x96 0x81). */
    char *norm = malloc(text_len * 3 + 1);
    size_t nn = 0;
    for (size_t i = 0; i < text_len; i++) {
        if (text[i] == ' ') {
            norm[nn++] = (char)0xe2;
            norm[nn++] = (char)0x96;
            norm[nn++] = (char)0x81;
        } else {
            norm[nn++] = text[i];
        }
    }

    /* Added/special tokens are matched greedily on the normalized text
     * (their strings contain no spaces, so normalization is a no-op for
     * them), splitting the text into plain BPE segments. */
    uint32_t count = 0;
    size_t seg_start = 0;
    for (size_t i = 0; i < nn;) {
        int32_t hit = -1;
        uint32_t hit_len = 0;
        for (uint32_t a = 0; a < t->n_added; a++) {
            const g4_tok_span *sp = &t->tok[t->added[a]];
            if (sp->len > hit_len && sp->len <= nn - i &&
                !memcmp(norm + i, t->strings + sp->off, sp->len))
            {
                hit = (int32_t)t->added[a];
                hit_len = sp->len;
            }
        }
        if (hit >= 0) {
            count = bpe_segment(t, norm + seg_start, i - seg_start, ids, cap,
                                count);
            count = emit(ids, cap, count, hit);
            i += hit_len;
            seg_start = i;
        } else {
            i++;
        }
    }
    count = bpe_segment(t, norm + seg_start, nn - seg_start, ids, cap, count);

    free(norm);
    return count;
}
