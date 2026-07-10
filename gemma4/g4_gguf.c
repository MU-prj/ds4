#include "g4_gguf.h"

#include <inttypes.h>
#include <stdlib.h>
#include <string.h>

static void g4_gguf_seterr(char *err, size_t errlen, const char *msg) {
    if (err && errlen) snprintf(err, errlen, "%s", msg);
}

static void *g4_xmalloc(size_t n) {
    void *p = malloc(n ? n : 1);
    if (!p) { fprintf(stderr, "g4_gguf: out of memory\n"); exit(1); }
    return p;
}

static void *g4_xrealloc(void *p, size_t n) {
    void *q = realloc(p, n ? n : 1);
    if (!q) { fprintf(stderr, "g4_gguf: out of memory\n"); exit(1); }
    return q;
}

static char *g4_xstrdup(const char *s) {
    char *d = g4_xmalloc(strlen(s) + 1);
    strcpy(d, s);
    return d;
}

static uint64_t g4_align_up(uint64_t x, uint64_t a) {
    return (x + a - 1) / a * a;
}

static uint64_t g4_tensor_nbytes(g4q_type type, uint32_t n_dims, const uint64_t *dims) {
    size_t row = g4q_row_size(type, (int64_t)dims[0]);
    if (row == 0) return 0;
    uint64_t n = row;
    for (uint32_t i = 1; i < n_dims; i++) n *= dims[i];
    return n;
}

/* ------------------------------------------------------------------ */
/* Reader                                                              */
/* ------------------------------------------------------------------ */

static int rd(FILE *fp, void *p, size_t n) {
    return fread(p, 1, n, fp) == n ? 0 : -1;
}

static int rd_str(FILE *fp, char **out, uint64_t *len_out) {
    uint64_t len;
    if (rd(fp, &len, 8)) return -1;
    if (len > (1ull << 31)) return -1;
    char *s = g4_xmalloc((size_t)len + 1);
    if (len && rd(fp, s, (size_t)len)) { free(s); return -1; }
    s[len] = 0;
    *out = s;
    if (len_out) *len_out = len;
    return 0;
}

static size_t g4_gguf_scalar_size(uint32_t type) {
    switch (type) {
        case G4_GGUF_U8: case G4_GGUF_I8: case G4_GGUF_BOOL: return 1;
        case G4_GGUF_U16: case G4_GGUF_I16: return 2;
        case G4_GGUF_U32: case G4_GGUF_I32: case G4_GGUF_F32: return 4;
        case G4_GGUF_U64: case G4_GGUF_I64: case G4_GGUF_F64: return 8;
        default: return 0;
    }
}

static int rd_kv_value(FILE *fp, g4_gguf_kv *kv, uint32_t type) {
    switch (type) {
        case G4_GGUF_U8: case G4_GGUF_U16: case G4_GGUF_U32: case G4_GGUF_U64: {
            uint64_t v = 0;
            if (rd(fp, &v, g4_gguf_scalar_size(type))) return -1;
            kv->v.u64 = v;
            return 0;
        }
        case G4_GGUF_I8: case G4_GGUF_I16: case G4_GGUF_I32: case G4_GGUF_I64: {
            int64_t v = 0;
            size_t sz = g4_gguf_scalar_size(type);
            uint64_t raw = 0;
            if (rd(fp, &raw, sz)) return -1;
            /* sign extend */
            if (sz < 8 && (raw >> (8 * sz - 1)) & 1) raw |= ~((1ull << (8 * sz)) - 1);
            memcpy(&v, &raw, 8);
            kv->v.i64 = v;
            return 0;
        }
        case G4_GGUF_F32: {
            float f;
            if (rd(fp, &f, 4)) return -1;
            kv->v.f64 = f;
            return 0;
        }
        case G4_GGUF_F64:
            return rd(fp, &kv->v.f64, 8);
        case G4_GGUF_BOOL: {
            uint8_t b;
            if (rd(fp, &b, 1)) return -1;
            kv->v.b = b != 0;
            return 0;
        }
        case G4_GGUF_STRING:
            return rd_str(fp, &kv->str, &kv->str_len);
        case G4_GGUF_ARRAY: {
            uint32_t et;
            uint64_t count;
            if (rd(fp, &et, 4) || rd(fp, &count, 8)) return -1;
            kv->elem_type = et;
            kv->count = count;
            if (et == G4_GGUF_STRING) {
                char **arr = g4_xmalloc(sizeof(char *) * (size_t)count);
                for (uint64_t i = 0; i < count; i++) arr[i] = NULL;
                for (uint64_t i = 0; i < count; i++) {
                    if (rd_str(fp, &arr[i], NULL)) {
                        for (uint64_t j = 0; j < count; j++) free(arr[j]);
                        free(arr);
                        return -1;
                    }
                }
                kv->arr = arr;
                return 0;
            }
            size_t esz = g4_gguf_scalar_size(et);
            if (!esz) return -1;
            kv->arr = g4_xmalloc(esz * (size_t)count);
            return rd(fp, kv->arr, esz * (size_t)count);
        }
        default:
            return -1;
    }
}

int g4_gguf_open(g4_gguf *g, const char *path, char *err, size_t errlen) {
    memset(g, 0, sizeof(*g));
    FILE *fp = fopen(path, "rb");
    if (!fp) { g4_gguf_seterr(err, errlen, "cannot open file"); return -1; }
    g->fp = fp;

    uint32_t magic = 0, version = 0;
    uint64_t n_tensor = 0, n_kv = 0;
    if (rd(fp, &magic, 4) || rd(fp, &version, 4) ||
        rd(fp, &n_tensor, 8) || rd(fp, &n_kv, 8) ||
        magic != G4_GGUF_MAGIC || version != G4_GGUF_VERSION)
    {
        g4_gguf_seterr(err, errlen, "bad GGUF header");
        g4_gguf_close(g);
        return -1;
    }

    g->kv = g4_xmalloc(sizeof(g4_gguf_kv) * (size_t)(n_kv ? n_kv : 1));
    memset(g->kv, 0, sizeof(g4_gguf_kv) * (size_t)(n_kv ? n_kv : 1));
    for (uint64_t i = 0; i < n_kv; i++) {
        g4_gguf_kv *kv = &g->kv[i];
        uint32_t type;
        if (rd_str(fp, &kv->key, NULL) || rd(fp, &type, 4) ||
            rd_kv_value(fp, kv, type))
        {
            g4_gguf_seterr(err, errlen, "bad GGUF metadata");
            g->n_kv = i + 1;
            g4_gguf_close(g);
            return -1;
        }
        kv->type = type;
        g->n_kv = i + 1;
    }

    g->tensor = g4_xmalloc(sizeof(g4_gguf_tensor) * (size_t)(n_tensor ? n_tensor : 1));
    memset(g->tensor, 0, sizeof(g4_gguf_tensor) * (size_t)(n_tensor ? n_tensor : 1));
    for (uint64_t i = 0; i < n_tensor; i++) {
        g4_gguf_tensor *t = &g->tensor[i];
        uint32_t n_dims = 0, type = 0;
        if (rd_str(fp, &t->name, NULL) || rd(fp, &n_dims, 4) ||
            n_dims == 0 || n_dims > G4_GGUF_MAX_DIMS)
        {
            g4_gguf_seterr(err, errlen, "bad GGUF tensor info");
            g->n_tensor = i + 1;
            g4_gguf_close(g);
            return -1;
        }
        t->n_dims = n_dims;
        for (uint32_t d = 0; d < n_dims; d++) {
            if (rd(fp, &t->dims[d], 8)) {
                g4_gguf_seterr(err, errlen, "bad GGUF tensor dims");
                g->n_tensor = i + 1;
                g4_gguf_close(g);
                return -1;
            }
        }
        if (rd(fp, &type, 4) || rd(fp, &t->offset, 8)) {
            g4_gguf_seterr(err, errlen, "bad GGUF tensor info");
            g->n_tensor = i + 1;
            g4_gguf_close(g);
            return -1;
        }
        t->type = type;
        t->nbytes = g4_tensor_nbytes((g4q_type)type, n_dims, t->dims);
        if (t->nbytes == 0) {
            g4_gguf_seterr(err, errlen, "unsupported tensor type/shape");
            g->n_tensor = i + 1;
            g4_gguf_close(g);
            return -1;
        }
        g->n_tensor = i + 1;
    }

    long pos = ftell(fp);
    if (pos < 0) { g4_gguf_close(g); return -1; }
    g->data_offset = g4_align_up((uint64_t)pos, G4_GGUF_ALIGN);
    return 0;
}

void g4_gguf_close(g4_gguf *g) {
    if (g->fp) fclose(g->fp);
    for (uint64_t i = 0; i < g->n_kv; i++) {
        g4_gguf_kv *kv = &g->kv[i];
        free(kv->key);
        free(kv->str);
        if (kv->type == G4_GGUF_ARRAY && kv->elem_type == G4_GGUF_STRING && kv->arr) {
            char **arr = (char **)kv->arr;
            for (uint64_t j = 0; j < kv->count; j++) free(arr[j]);
        }
        free(kv->arr);
    }
    free(g->kv);
    for (uint64_t i = 0; i < g->n_tensor; i++) free(g->tensor[i].name);
    free(g->tensor);
    memset(g, 0, sizeof(*g));
}

const g4_gguf_kv *g4_gguf_get(const g4_gguf *g, const char *key) {
    for (uint64_t i = 0; i < g->n_kv; i++)
        if (!strcmp(g->kv[i].key, key)) return &g->kv[i];
    return NULL;
}

const g4_gguf_tensor *g4_gguf_tensor_by_name(const g4_gguf *g, const char *name) {
    for (uint64_t i = 0; i < g->n_tensor; i++)
        if (!strcmp(g->tensor[i].name, name)) return &g->tensor[i];
    return NULL;
}

int g4_gguf_read_tensor_data(const g4_gguf *g, const g4_gguf_tensor *t, void *dst) {
    if (fseek(g->fp, (long)(g->data_offset + t->offset), SEEK_SET)) return -1;
    return rd(g->fp, dst, (size_t)t->nbytes);
}

/* ------------------------------------------------------------------ */
/* Writer                                                              */
/* ------------------------------------------------------------------ */

typedef struct {
    uint8_t *p;
    size_t len, cap;
} g4_buf;

static void g4_buf_put(g4_buf *b, const void *p, size_t n) {
    if (b->len + n > b->cap) {
        b->cap = (b->len + n) * 2 + 64;
        b->p = g4_xrealloc(b->p, b->cap);
    }
    memcpy(b->p + b->len, p, n);
    b->len += n;
}

static void g4_buf_put_str(g4_buf *b, const char *s) {
    uint64_t len = strlen(s);
    g4_buf_put(b, &len, 8);
    g4_buf_put(b, s, (size_t)len);
}

typedef struct {
    char *name;
    uint32_t type;
    uint32_t n_dims;
    uint64_t dims[G4_GGUF_MAX_DIMS];
    const void *data;
    uint64_t nbytes;
    uint64_t offset;
} g4_gguf_wtensor;

struct g4_gguf_writer {
    g4_buf kvbuf;
    uint64_t n_kv;
    g4_gguf_wtensor *tensor;
    uint64_t n_tensor, cap_tensor;
};

g4_gguf_writer *g4_gguf_writer_new(void) {
    g4_gguf_writer *w = g4_xmalloc(sizeof(*w));
    memset(w, 0, sizeof(*w));
    return w;
}

void g4_gguf_writer_free(g4_gguf_writer *w) {
    if (!w) return;
    free(w->kvbuf.p);
    for (uint64_t i = 0; i < w->n_tensor; i++) free(w->tensor[i].name);
    free(w->tensor);
    free(w);
}

static void wkv_head(g4_gguf_writer *w, const char *key, uint32_t type) {
    g4_buf_put_str(&w->kvbuf, key);
    g4_buf_put(&w->kvbuf, &type, 4);
    w->n_kv++;
}

void g4_gguf_writer_kv_u32(g4_gguf_writer *w, const char *key, uint32_t v) {
    wkv_head(w, key, G4_GGUF_U32);
    g4_buf_put(&w->kvbuf, &v, 4);
}

void g4_gguf_writer_kv_u64(g4_gguf_writer *w, const char *key, uint64_t v) {
    wkv_head(w, key, G4_GGUF_U64);
    g4_buf_put(&w->kvbuf, &v, 8);
}

void g4_gguf_writer_kv_f32(g4_gguf_writer *w, const char *key, float v) {
    wkv_head(w, key, G4_GGUF_F32);
    g4_buf_put(&w->kvbuf, &v, 4);
}

void g4_gguf_writer_kv_bool(g4_gguf_writer *w, const char *key, bool v) {
    uint8_t b = v ? 1 : 0;
    wkv_head(w, key, G4_GGUF_BOOL);
    g4_buf_put(&w->kvbuf, &b, 1);
}

void g4_gguf_writer_kv_str(g4_gguf_writer *w, const char *key, const char *v) {
    wkv_head(w, key, G4_GGUF_STRING);
    g4_buf_put_str(&w->kvbuf, v);
}

static void wkv_arr_head(g4_gguf_writer *w, const char *key, uint32_t et, uint64_t n) {
    wkv_head(w, key, G4_GGUF_ARRAY);
    g4_buf_put(&w->kvbuf, &et, 4);
    g4_buf_put(&w->kvbuf, &n, 8);
}

void g4_gguf_writer_kv_arr_i32(g4_gguf_writer *w, const char *key,
                               const int32_t *v, uint64_t n) {
    wkv_arr_head(w, key, G4_GGUF_I32, n);
    g4_buf_put(&w->kvbuf, v, 4 * (size_t)n);
}

void g4_gguf_writer_kv_arr_f32(g4_gguf_writer *w, const char *key,
                               const float *v, uint64_t n) {
    wkv_arr_head(w, key, G4_GGUF_F32, n);
    g4_buf_put(&w->kvbuf, v, 4 * (size_t)n);
}

void g4_gguf_writer_kv_arr_str(g4_gguf_writer *w, const char *key,
                               const char *const *v, uint64_t n) {
    wkv_arr_head(w, key, G4_GGUF_STRING, n);
    for (uint64_t i = 0; i < n; i++) g4_buf_put_str(&w->kvbuf, v[i]);
}

int g4_gguf_writer_tensor(g4_gguf_writer *w, const char *name, g4q_type type,
                          uint32_t n_dims, const uint64_t *dims,
                          const void *data, uint64_t nbytes) {
    if (n_dims == 0 || n_dims > G4_GGUF_MAX_DIMS) return -1;
    uint64_t expect = g4_tensor_nbytes(type, n_dims, dims);
    if (expect == 0 || expect != nbytes) return -1;
    if (w->n_tensor == w->cap_tensor) {
        w->cap_tensor = w->cap_tensor ? w->cap_tensor * 2 : 16;
        w->tensor = g4_xrealloc(w->tensor, sizeof(*w->tensor) * (size_t)w->cap_tensor);
    }
    g4_gguf_wtensor *t = &w->tensor[w->n_tensor++];
    memset(t, 0, sizeof(*t));
    t->name = g4_xstrdup(name);
    t->type = (uint32_t)type;
    t->n_dims = n_dims;
    memcpy(t->dims, dims, sizeof(uint64_t) * n_dims);
    t->data = data;
    t->nbytes = nbytes;
    return 0;
}

int g4_gguf_writer_finish(g4_gguf_writer *w, const char *path,
                          char *err, size_t errlen) {
    uint64_t off = 0;
    for (uint64_t i = 0; i < w->n_tensor; i++) {
        w->tensor[i].offset = off;
        off = g4_align_up(off + w->tensor[i].nbytes, G4_GGUF_ALIGN);
    }

    FILE *fp = fopen(path, "wb");
    if (!fp) { g4_gguf_seterr(err, errlen, "cannot create file"); return -1; }

    uint32_t magic = G4_GGUF_MAGIC, version = G4_GGUF_VERSION;
    if (fwrite(&magic, 4, 1, fp) != 1 || fwrite(&version, 4, 1, fp) != 1 ||
        fwrite(&w->n_tensor, 8, 1, fp) != 1 || fwrite(&w->n_kv, 8, 1, fp) != 1 ||
        (w->kvbuf.len && fwrite(w->kvbuf.p, 1, w->kvbuf.len, fp) != w->kvbuf.len))
    {
        goto werr;
    }

    for (uint64_t i = 0; i < w->n_tensor; i++) {
        const g4_gguf_wtensor *t = &w->tensor[i];
        uint64_t name_len = strlen(t->name);
        if (fwrite(&name_len, 8, 1, fp) != 1 ||
            fwrite(t->name, 1, (size_t)name_len, fp) != name_len ||
            fwrite(&t->n_dims, 4, 1, fp) != 1 ||
            fwrite(t->dims, 8, t->n_dims, fp) != t->n_dims ||
            fwrite(&t->type, 4, 1, fp) != 1 ||
            fwrite(&t->offset, 8, 1, fp) != 1)
        {
            goto werr;
        }
    }

    long pos = ftell(fp);
    if (pos < 0) goto werr;
    uint64_t pad = g4_align_up((uint64_t)pos, G4_GGUF_ALIGN) - (uint64_t)pos;
    static const uint8_t zeros[G4_GGUF_ALIGN] = {0};
    if (pad && fwrite(zeros, 1, (size_t)pad, fp) != pad) goto werr;

    uint64_t written = 0;
    for (uint64_t i = 0; i < w->n_tensor; i++) {
        const g4_gguf_wtensor *t = &w->tensor[i];
        uint64_t gap = t->offset - written;
        if (gap && fwrite(zeros, 1, (size_t)gap, fp) != gap) goto werr;
        if (fwrite(t->data, 1, (size_t)t->nbytes, fp) != t->nbytes) goto werr;
        written = t->offset + t->nbytes;
    }

    if (fclose(fp)) { g4_gguf_seterr(err, errlen, "close failed"); return -1; }
    return 0;

werr:
    fclose(fp);
    g4_gguf_seterr(err, errlen, "write failed");
    return -1;
}
