#ifndef G4_GGUF_H
#define G4_GGUF_H

/* Minimal GGUF v3 reader/writer for the g4 engine.
 *
 * Deliberately narrow: little-endian hosts only, alignment fixed at 32,
 * the value types the g4 tooling actually emits.  The reader keeps tensor
 * data on disk and loads it on request, so opening a 9 GiB model does not
 * read tensor bytes. */

#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>

#include "g4_quants.h"

#define G4_GGUF_MAGIC   0x46554747u /* "GGUF" */
#define G4_GGUF_VERSION 3u
#define G4_GGUF_ALIGN   32u
#define G4_GGUF_MAX_DIMS 4

typedef enum {
    G4_GGUF_U8 = 0, G4_GGUF_I8 = 1, G4_GGUF_U16 = 2, G4_GGUF_I16 = 3,
    G4_GGUF_U32 = 4, G4_GGUF_I32 = 5, G4_GGUF_F32 = 6, G4_GGUF_BOOL = 7,
    G4_GGUF_STRING = 8, G4_GGUF_ARRAY = 9, G4_GGUF_U64 = 10,
    G4_GGUF_I64 = 11, G4_GGUF_F64 = 12,
} g4_gguf_value_type;

typedef struct {
    char *key;
    uint32_t type;       /* g4_gguf_value_type */
    uint32_t elem_type;  /* for arrays */
    uint64_t count;      /* for arrays; strings use str_len */
    union {
        uint64_t u64;
        int64_t i64;
        double f64;
        bool b;
    } v;
    char *str;           /* G4_GGUF_STRING */
    uint64_t str_len;
    void *arr;           /* packed elems; array-of-string = char*[] */
} g4_gguf_kv;

typedef struct {
    char *name;
    uint32_t n_dims;
    uint64_t dims[G4_GGUF_MAX_DIMS]; /* dims[0] = row length (contraction) */
    uint32_t type;                   /* g4q_type */
    uint64_t offset;                 /* relative to data section, aligned */
    uint64_t nbytes;
} g4_gguf_tensor;

typedef struct {
    FILE *fp;
    uint64_t data_offset; /* absolute file offset of the data section */
    g4_gguf_kv *kv;
    uint64_t n_kv;
    g4_gguf_tensor *tensor;
    uint64_t n_tensor;
    /* Optional read-only mmap of the whole file, for zero-copy tensor
     * access on machines that cannot hold the model in RAM. */
    void *map_base;
    uint64_t map_size;
} g4_gguf;

int g4_gguf_open(g4_gguf *g, const char *path, char *err, size_t errlen);
/* Like g4_gguf_open, but also mmaps the file read-only so tensor bytes can
 * be reached with g4_gguf_tensor_ptr() without copying into RAM. */
int g4_gguf_open_mmap(g4_gguf *g, const char *path, char *err, size_t errlen);
void g4_gguf_close(g4_gguf *g);
const g4_gguf_kv *g4_gguf_get(const g4_gguf *g, const char *key);
const g4_gguf_tensor *g4_gguf_tensor_by_name(const g4_gguf *g, const char *name);
/* dst must hold t->nbytes bytes. */
int g4_gguf_read_tensor_data(const g4_gguf *g, const g4_gguf_tensor *t, void *dst);
/* Zero-copy pointer to a tensor's bytes inside the mmap; NULL if the file was
 * not opened with g4_gguf_open_mmap(). */
const void *g4_gguf_tensor_ptr(const g4_gguf *g, const g4_gguf_tensor *t);

/* Writer: buffers metadata and tensor descriptors, streams tensor bytes to a
 * temp layout on finish.  Tensor data pointers must stay valid until
 * g4_gguf_writer_finish(). */
typedef struct g4_gguf_writer g4_gguf_writer;

g4_gguf_writer *g4_gguf_writer_new(void);
void g4_gguf_writer_free(g4_gguf_writer *w);
void g4_gguf_writer_kv_u32(g4_gguf_writer *w, const char *key, uint32_t v);
void g4_gguf_writer_kv_u64(g4_gguf_writer *w, const char *key, uint64_t v);
void g4_gguf_writer_kv_f32(g4_gguf_writer *w, const char *key, float v);
void g4_gguf_writer_kv_bool(g4_gguf_writer *w, const char *key, bool v);
void g4_gguf_writer_kv_str(g4_gguf_writer *w, const char *key, const char *v);
void g4_gguf_writer_kv_arr_i32(g4_gguf_writer *w, const char *key,
                               const int32_t *v, uint64_t n);
void g4_gguf_writer_kv_arr_f32(g4_gguf_writer *w, const char *key,
                               const float *v, uint64_t n);
void g4_gguf_writer_kv_arr_str(g4_gguf_writer *w, const char *key,
                               const char *const *v, uint64_t n);
/* dims[0] is the row length; data is the already-encoded tensor payload. */
int g4_gguf_writer_tensor(g4_gguf_writer *w, const char *name, g4q_type type,
                          uint32_t n_dims, const uint64_t *dims,
                          const void *data, uint64_t nbytes);
int g4_gguf_writer_finish(g4_gguf_writer *w, const char *path,
                          char *err, size_t errlen);

/* Streaming mode, for files larger than RAM: declare every tensor with
 * g4_gguf_writer_tensor_info() (same order as the data will come), then
 * g4_gguf_writer_begin() writes header+metadata+infos, and the data is
 * appended per tensor with g4_gguf_writer_put() (exactly nbytes per tensor,
 * in declaration order; padding is handled internally).  Finish with
 * g4_gguf_writer_end().  Do not mix with g4_gguf_writer_finish(). */
int g4_gguf_writer_tensor_info(g4_gguf_writer *w, const char *name,
                               g4q_type type, uint32_t n_dims,
                               const uint64_t *dims);
int g4_gguf_writer_begin(g4_gguf_writer *w, const char *path,
                         char *err, size_t errlen);
int g4_gguf_writer_put(g4_gguf_writer *w, const void *data, uint64_t nbytes);
int g4_gguf_writer_end(g4_gguf_writer *w, char *err, size_t errlen);

#endif
