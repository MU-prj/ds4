#ifndef G4_KVSTORE_H
#define G4_KVSTORE_H

/* Disk KV cache container for the g4 engine (design: gemma4-port/03).
 *
 * M0 scope: the on-disk fixed header (KVG, 48 bytes, same layout as the ds4
 * KVC header), the SHA1 identity of the rendered byte prefix, and the
 * eviction score.  Store scanning, save frontiers and the G4SP payload land
 * with M6. */

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>
#include <stdio.h>

#define G4_KVSTORE_FIXED_HEADER 48u
#define G4_KVSTORE_MODEL_ID_GEMMA4_26B_A4B 1u
#define G4_KVSTORE_HIT_HALF_LIFE_SECONDS (6ull * 60ull * 60ull)

#define G4_KVSTORE_EXT_TOOL_MAP      (1u << 0)
#define G4_KVSTORE_EXT_SESSION_TITLE (1u << 1)

typedef enum {
    G4_KVSTORE_REASON_UNKNOWN   = 0,
    G4_KVSTORE_REASON_COLD      = 1,
    G4_KVSTORE_REASON_CONTINUED = 2,
    G4_KVSTORE_REASON_EVICT     = 3,
    G4_KVSTORE_REASON_SHUTDOWN  = 4,
} g4_kvstore_reason;

typedef struct {
    char sha[41];
    uint8_t quant_bits;
    uint8_t reason;
    uint8_t ext_flags;
    uint8_t model_id;
    uint32_t tokens;
    uint32_t hits;
    uint32_t ctx_size;
    uint64_t created_at;
    uint64_t last_used;
    uint64_t payload_bytes;
    uint64_t file_size;   /* filled by the store scan, 0 otherwise */
    uint32_t text_bytes;  /* filled by the store scan, 0 otherwise */
} g4_kvstore_entry;

void g4_kvstore_sha1_hex(const void *ptr, size_t len, char out[41]);

void g4_kvstore_fill_header(uint8_t h[G4_KVSTORE_FIXED_HEADER],
                            const g4_kvstore_entry *e);
/* Returns false when magic/version/model layout is not a valid KVG header. */
bool g4_kvstore_parse_header(const uint8_t h[G4_KVSTORE_FIXED_HEADER],
                             g4_kvstore_entry *e);

/* Value-density eviction score, ported from ds4_kvstore_entry_eviction_score:
 * higher is more worth keeping. */
double g4_kvstore_eviction_score(const g4_kvstore_entry *e, uint64_t now);

#endif
