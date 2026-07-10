#include "g4_kvstore.h"

#include <math.h>
#include <string.h>

/* SHA1 implementation copied verbatim from ds4_kvstore.c. */

typedef struct {
    uint32_t h[5];
    uint64_t bytes;
    uint8_t block[64];
    size_t used;
} sha1_ctx;

static uint32_t rol32(uint32_t v, int n) {
    return (v << n) | (v >> (32 - n));
}

static void sha1_transform(sha1_ctx *c, const uint8_t block[64]) {
    uint32_t w[80];
    for (int i = 0; i < 16; i++) {
        w[i] = ((uint32_t)block[i * 4] << 24) |
               ((uint32_t)block[i * 4 + 1] << 16) |
               ((uint32_t)block[i * 4 + 2] << 8) |
               (uint32_t)block[i * 4 + 3];
    }
    for (int i = 16; i < 80; i++)
        w[i] = rol32(w[i - 3] ^ w[i - 8] ^ w[i - 14] ^ w[i - 16], 1);

    uint32_t a = c->h[0], b = c->h[1], d = c->h[3], e = c->h[4];
    uint32_t cc = c->h[2];
    for (int i = 0; i < 80; i++) {
        uint32_t f, k;
        if (i < 20) {
            f = (b & cc) | ((~b) & d);
            k = 0x5a827999u;
        } else if (i < 40) {
            f = b ^ cc ^ d;
            k = 0x6ed9eba1u;
        } else if (i < 60) {
            f = (b & cc) | (b & d) | (cc & d);
            k = 0x8f1bbcdcu;
        } else {
            f = b ^ cc ^ d;
            k = 0xca62c1d6u;
        }
        uint32_t tmp = rol32(a, 5) + f + e + k + w[i];
        e = d;
        d = cc;
        cc = rol32(b, 30);
        b = a;
        a = tmp;
    }
    c->h[0] += a;
    c->h[1] += b;
    c->h[2] += cc;
    c->h[3] += d;
    c->h[4] += e;
}

static void sha1_init(sha1_ctx *c) {
    c->h[0] = 0x67452301u;
    c->h[1] = 0xefcdab89u;
    c->h[2] = 0x98badcfeu;
    c->h[3] = 0x10325476u;
    c->h[4] = 0xc3d2e1f0u;
    c->bytes = 0;
    c->used = 0;
}

static void sha1_update(sha1_ctx *c, const void *ptr, size_t len) {
    const uint8_t *p = ptr;
    c->bytes += len;
    while (len != 0) {
        size_t n = 64 - c->used;
        if (n > len) n = len;
        memcpy(c->block + c->used, p, n);
        c->used += n;
        p += n;
        len -= n;
        if (c->used == 64) {
            sha1_transform(c, c->block);
            c->used = 0;
        }
    }
}

static void sha1_final(sha1_ctx *c, uint8_t out[20]) {
    uint64_t bits = c->bytes * 8;
    uint8_t one = 0x80;
    uint8_t zero = 0;
    sha1_update(c, &one, 1);
    while (c->used != 56) sha1_update(c, &zero, 1);
    uint8_t len[8];
    for (int i = 0; i < 8; i++) len[7 - i] = (uint8_t)(bits >> (8 * i));
    sha1_update(c, len, sizeof(len));
    for (int i = 0; i < 5; i++) {
        out[i * 4] = (uint8_t)(c->h[i] >> 24);
        out[i * 4 + 1] = (uint8_t)(c->h[i] >> 16);
        out[i * 4 + 2] = (uint8_t)(c->h[i] >> 8);
        out[i * 4 + 3] = (uint8_t)c->h[i];
    }
}

static void hex20(const uint8_t in[20], char out[41]) {
    static const char hex[] = "0123456789abcdef";
    for (int i = 0; i < 20; i++) {
        out[i * 2] = hex[in[i] >> 4];
        out[i * 2 + 1] = hex[in[i] & 15];
    }
    out[40] = '\0';
}


void g4_kvstore_sha1_hex(const void *ptr, size_t len, char out[41]) {
    sha1_ctx c;
    uint8_t digest[20];
    sha1_init(&c);
    sha1_update(&c, ptr, len);
    sha1_final(&c, digest);
    hex20(digest, out);
}

static void le_put32(uint8_t *p, uint32_t v) {
    p[0] = (uint8_t)(v & 0xff);
    p[1] = (uint8_t)((v >> 8) & 0xff);
    p[2] = (uint8_t)((v >> 16) & 0xff);
    p[3] = (uint8_t)((v >> 24) & 0xff);
}

static void le_put64(uint8_t *p, uint64_t v) {
    le_put32(p, (uint32_t)(v & 0xffffffffu));
    le_put32(p + 4, (uint32_t)(v >> 32));
}

static uint32_t le_get32(const uint8_t *p) {
    return (uint32_t)p[0] | ((uint32_t)p[1] << 8) |
           ((uint32_t)p[2] << 16) | ((uint32_t)p[3] << 24);
}

static uint64_t le_get64(const uint8_t *p) {
    return (uint64_t)le_get32(p) | ((uint64_t)le_get32(p + 4) << 32);
}

/* KVG header, 48 bytes little-endian (gemma4-port/03-disk-kv-cache.md §4). */
void g4_kvstore_fill_header(uint8_t h[G4_KVSTORE_FIXED_HEADER],
                            const g4_kvstore_entry *e) {
    memset(h, 0, G4_KVSTORE_FIXED_HEADER);
    h[0] = 'K'; h[1] = 'V'; h[2] = 'G';
    h[3] = 1; /* version */
    h[4] = e->quant_bits;
    h[5] = e->reason;
    h[6] = e->ext_flags;
    h[7] = e->model_id;
    le_put32(h + 8, e->tokens);
    le_put32(h + 12, e->hits);
    le_put32(h + 16, e->ctx_size);
    le_put64(h + 24, e->created_at);
    le_put64(h + 32, e->last_used);
    le_put64(h + 40, e->payload_bytes);
}

bool g4_kvstore_parse_header(const uint8_t h[G4_KVSTORE_FIXED_HEADER],
                             g4_kvstore_entry *e) {
    if (h[0] != 'K' || h[1] != 'V' || h[2] != 'G') return false;
    if (h[3] != 1) return false;
    if (h[7] == 0) return false; /* model_id 0 is reserved/invalid */
    memset(e, 0, sizeof(*e));
    e->quant_bits = h[4];
    e->reason = h[5];
    e->ext_flags = h[6];
    e->model_id = h[7];
    e->tokens = le_get32(h + 8);
    e->hits = le_get32(h + 12);
    e->ctx_size = le_get32(h + 16);
    e->created_at = le_get64(h + 24);
    e->last_used = le_get64(h + 32);
    e->payload_bytes = le_get64(h + 40);
    return true;
}

#define G4_KVSTORE_MIN_EFFECTIVE_HITS 0.01
#define G4_KVSTORE_ANCHOR_SCORE_FACTOR 4.0

double g4_kvstore_eviction_score(const g4_kvstore_entry *e, uint64_t now) {
    if (!e || e->file_size == 0) return 0.0;
    double effective_hits = (double)e->hits;
    uint64_t used_at = e->last_used ? e->last_used : e->created_at;
    if (used_at == 0) {
        effective_hits = 0.0;
    } else if (now > used_at) {
        double elapsed = (double)(now - used_at);
        effective_hits *= exp2(-elapsed / (double)G4_KVSTORE_HIT_HALF_LIFE_SECONDS);
        if (effective_hits < G4_KVSTORE_MIN_EFFECTIVE_HITS) effective_hits = 0.0;
    }
    double score = (effective_hits + 1.0) *
                   (double)e->tokens / (double)e->file_size;
    if (e->reason == G4_KVSTORE_REASON_COLD ||
        e->reason == G4_KVSTORE_REASON_EVICT ||
        e->reason == G4_KVSTORE_REASON_SHUTDOWN)
    {
        score *= G4_KVSTORE_ANCHOR_SCORE_FACTOR;
    }
    return score;
}
