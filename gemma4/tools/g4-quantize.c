/* g4-quantize — HF safetensors -> g4 GGUF converter for Gemma 4 26B-A4B.
 *
 * Template-free (gemma4-port/04 §2): hyperparameters come from config.json,
 * tensor shapes/dtypes from the safetensors shard headers, the recipe from
 * the CLI.  Tensor naming/orientation contract: gemma4-port/02 §6 and
 * scripts/gen_synthetic_hf.py (the executable spec, validated end-to-end
 * against the JAX reference logits).
 *
 * Orientation is verified from shapes: d_model differs from every other
 * dimension in this model, so a transposed source tensor is detected and
 * rejected instead of silently reoriented.
 *
 * v1 limits (documented): single-threaded; no imatrix yet (Q8_0/Q4_K
 * bring-up profiles; the IQ2_XXS/Q2_K imatrix build lands with M5); the
 * tokenizer is not embedded in the GGUF yet (the engine uses the G4TK blob).
 *
 * Usage:
 *   tools/g4-quantize --hf DIR --out FILE [--profile q8|q4|f32]
 *       [--experts TYPE] [--routed-down TYPE] [--dense TYPE]
 *       [--attn TYPE] [--embed TYPE] [--dry-run] [--compare-tensor NAME]
 */

#include "../g4_gguf.h"
#include "../g4_quants.h"

#include <dirent.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static void die(const char *fmt, const char *a) {
    fprintf(stderr, "g4-quantize: ");
    fprintf(stderr, fmt, a);
    fprintf(stderr, "\n");
    exit(1);
}

static void *xmalloc(size_t n) {
    void *p = malloc(n ? n : 1);
    if (!p) die("out of memory%s", "");
    return p;
}

/* ------------------------------------------------------------------ */
/* Restricted JSON scanning (safetensors headers, config.json)         */
/* ------------------------------------------------------------------ */

/* Finds `"key"` at object level inside buf[0..len) and returns a pointer
 * just past the following ':'.  NULL if absent.  Good enough for the two
 * fixed JSON producers we consume (HF metadata, safetensors headers). */
static const char *json_after_key(const char *buf, size_t len, const char *key) {
    char pat[128];
    int n = snprintf(pat, sizeof(pat), "\"%s\"", key);
    for (size_t i = 0; i + n < len; i++) {
        if (memcmp(buf + i, pat, (size_t)n)) continue;
        const char *p = buf + i + n;
        while (*p == ' ' || *p == '\n' || *p == '\t') p++;
        if (*p == ':') return p + 1;
    }
    return NULL;
}

static bool json_number(const char *buf, size_t len, const char *key, double *out) {
    const char *p = json_after_key(buf, len, key);
    if (!p) return false;
    *out = strtod(p, NULL);
    return true;
}

static bool json_bool(const char *buf, size_t len, const char *key, bool *out) {
    const char *p = json_after_key(buf, len, key);
    if (!p) return false;
    while (*p == ' ') p++;
    *out = strncmp(p, "true", 4) == 0;
    return true;
}

/* ------------------------------------------------------------------ */
/* safetensors shards                                                  */
/* ------------------------------------------------------------------ */

typedef struct {
    char name[192];
    int shard;
    int dtype;          /* 0 = BF16, 1 = F32, 2 = F16 */
    uint32_t n_dims;
    uint64_t dims[4];   /* numpy order: dims[0] slowest */
    uint64_t begin, end;
} st_tensor;

typedef struct {
    char **shard_path;
    uint64_t *shard_data0;
    int n_shards;
    st_tensor *t;
    int n, cap;
} st_index;

static void st_parse_header(st_index *ix, int shard, const char *hdr, size_t n) {
    /* {"name":{"dtype":"BF16","shape":[..],"data_offsets":[a,b]}, ...} */
    size_t i = 0;
    while (i < n && hdr[i] != '{') i++;
    i++;
    while (i < n) {
        while (i < n && hdr[i] != '"') i++;
        if (i >= n) break;
        size_t s = ++i;
        while (i < n && hdr[i] != '"') i++;
        size_t name_len = i - s;
        i++;
        while (i < n && hdr[i] != '{') i++;
        size_t obj_start = i;
        int depth = 0;
        while (i < n) {
            if (hdr[i] == '{') depth++;
            if (hdr[i] == '}') { depth--; if (!depth) break; }
            i++;
        }
        size_t obj_len = i - obj_start + 1;
        i++;
        if (name_len >= 8 && !memcmp(hdr + s, "__metadata__", name_len < 12 ? name_len : 12))
            continue;
        if (ix->n == ix->cap) {
            ix->cap = ix->cap ? ix->cap * 2 : 256;
            ix->t = realloc(ix->t, sizeof(st_tensor) * (size_t)ix->cap);
        }
        st_tensor *t = &ix->t[ix->n];
        memset(t, 0, sizeof(*t));
        if (name_len >= sizeof(t->name)) die("tensor name too long%s", "");
        memcpy(t->name, hdr + s, name_len);
        t->shard = shard;
        const char *obj = hdr + obj_start;
        const char *d = json_after_key(obj, obj_len, "dtype");
        if (!d) die("no dtype for %s", t->name);
        while (*d == ' ' || *d == '"') d++;
        if (!strncmp(d, "BF16", 4)) t->dtype = 0;
        else if (!strncmp(d, "F32", 3)) t->dtype = 1;
        else if (!strncmp(d, "F16", 3)) t->dtype = 2;
        else die("unsupported dtype for %s", t->name);
        const char *sh = json_after_key(obj, obj_len, "shape");
        if (!sh) die("no shape for %s", t->name);
        while (*sh && *sh != '[') sh++;
        sh++;
        while (*sh && *sh != ']') {
            if (t->n_dims >= 4) die("too many dims for %s", t->name);
            t->dims[t->n_dims++] = strtoull(sh, (char **)&sh, 10);
            while (*sh == ',' || *sh == ' ') sh++;
        }
        const char *off = json_after_key(obj, obj_len, "data_offsets");
        if (!off) die("no data_offsets for %s", t->name);
        while (*off && *off != '[') off++;
        off++;
        t->begin = strtoull(off, (char **)&off, 10);
        while (*off == ',' || *off == ' ') off++;
        t->end = strtoull(off, NULL, 10);
        ix->n++;
    }
}

static void st_open_dir(st_index *ix, const char *dir) {
    memset(ix, 0, sizeof(*ix));
    DIR *d = opendir(dir);
    if (!d) die("cannot open --hf dir %s", dir);
    struct dirent *e;
    char **names = NULL;
    int nn = 0;
    while ((e = readdir(d))) {
        size_t l = strlen(e->d_name);
        if (l > 12 && !strcmp(e->d_name + l - 12, ".safetensors")) {
            names = realloc(names, sizeof(char *) * (size_t)(nn + 1));
            names[nn] = xmalloc(strlen(dir) + l + 2);
            sprintf(names[nn], "%s/%s", dir, e->d_name);
            nn++;
        }
    }
    closedir(d);
    if (!nn) die("no .safetensors files in %s", dir);
    ix->shard_path = names;
    ix->shard_data0 = xmalloc(sizeof(uint64_t) * (size_t)nn);
    ix->n_shards = nn;
    for (int s = 0; s < nn; s++) {
        FILE *fp = fopen(names[s], "rb");
        if (!fp) die("cannot open shard %s", names[s]);
        uint64_t hlen;
        if (fread(&hlen, 8, 1, fp) != 1 || hlen > (1ull << 28))
            die("bad safetensors header in %s", names[s]);
        char *hdr = xmalloc((size_t)hlen);
        if (fread(hdr, 1, (size_t)hlen, fp) != hlen)
            die("bad safetensors header in %s", names[s]);
        ix->shard_data0[s] = 8 + hlen;
        st_parse_header(ix, s, hdr, (size_t)hlen);
        free(hdr);
        fclose(fp);
    }
}

static const st_tensor *st_find(const st_index *ix, const char *name) {
    for (int i = 0; i < ix->n; i++)
        if (!strcmp(ix->t[i].name, name)) return &ix->t[i];
    return NULL;
}

/* Loads a tensor fully, converted to f32 (numpy layout preserved). */
static float *st_load_f32(const st_index *ix, const st_tensor *t, uint64_t *n_out) {
    uint64_t n = 1;
    for (uint32_t i = 0; i < t->n_dims; i++) n *= t->dims[i];
    size_t esz = t->dtype == 1 ? 4 : 2;
    if (t->end - t->begin != n * esz) die("size mismatch for %s", t->name);
    FILE *fp = fopen(ix->shard_path[t->shard], "rb");
    if (!fp) die("cannot open shard for %s", t->name);
    if (fseek(fp, (long)(ix->shard_data0[t->shard] + t->begin), SEEK_SET))
        die("seek failed for %s", t->name);
    float *out = xmalloc(sizeof(float) * (size_t)n);
    if (t->dtype == 1) {
        if (fread(out, 4, (size_t)n, fp) != n) die("read failed for %s", t->name);
    } else {
        uint16_t *tmp = xmalloc(2 * (size_t)n);
        if (fread(tmp, 2, (size_t)n, fp) != n) die("read failed for %s", t->name);
        for (uint64_t i = 0; i < n; i++)
            out[i] = t->dtype == 0 ? g4q_bf16_to_f32(tmp[i]) : g4q_f16_to_f32(tmp[i]);
        free(tmp);
    }
    fclose(fp);
    if (n_out) *n_out = n;
    return out;
}

/* ------------------------------------------------------------------ */
/* Model config (from config.json, text_config section)                */
/* ------------------------------------------------------------------ */

typedef struct {
    uint32_t layers, d_model, vocab, ctx, heads, kv_local, kv_global;
    uint32_t key_local, key_global, sliding, experts, top_k, hexp, dense;
    float softcap, eps, rope_local, rope_global, rope_prop;
    bool k_eq_v;
} g4cfg;

static void load_config(const char *dir, g4cfg *c) {
    char path[512];
    snprintf(path, sizeof(path), "%s/config.json", dir);
    FILE *fp = fopen(path, "rb");
    if (!fp) die("cannot open %s", path);
    fseek(fp, 0, SEEK_END);
    long n = ftell(fp);
    fseek(fp, 0, SEEK_SET);
    char *buf = xmalloc((size_t)n + 1);
    if (fread(buf, 1, (size_t)n, fp) != (size_t)n) die("cannot read %s", path);
    buf[n] = 0;
    fclose(fp);

    /* Scope to the text_config object (vision_config repeats key names). */
    const char *tc = strstr(buf, "\"text_config\"");
    if (!tc) die("no text_config in %s", path);
    const char *end = strstr(tc, "\"vision_config\"");
    size_t len = end ? (size_t)(end - tc) : strlen(tc);

    double v;
#define REQ(key, field) \
    do { \
        if (!json_number(tc, len, key, &v)) die("missing %s in config", key); \
        c->field = (uint32_t)v; \
    } while (0)
#define REQF(key, field) \
    do { \
        if (!json_number(tc, len, key, &v)) die("missing %s in config", key); \
        c->field = (float)v; \
    } while (0)
    REQ("num_hidden_layers", layers);
    REQ("hidden_size", d_model);
    REQ("vocab_size", vocab);
    REQ("max_position_embeddings", ctx);
    REQ("num_attention_heads", heads);
    REQ("num_key_value_heads", kv_local);
    REQ("num_global_key_value_heads", kv_global);
    REQ("head_dim", key_local);
    REQ("global_head_dim", key_global);
    REQ("sliding_window", sliding);
    REQ("num_experts", experts);
    REQ("top_k_experts", top_k);
    REQ("moe_intermediate_size", hexp);
    REQ("intermediate_size", dense);
    REQF("final_logit_softcapping", softcap);
    REQF("rms_norm_eps", eps);
    if (!json_bool(tc, len, "attention_k_eq_v", &c->k_eq_v))
        die("missing attention_k_eq_v%s", "");
    const char *full = json_after_key(tc, len, "full_attention");
    const char *slide = json_after_key(tc, len, "sliding_attention");
    if (!full || !slide) die("missing rope_parameters%s", "");
    if (!json_number(full, len - (size_t)(full - tc), "rope_theta", &v))
        die("missing global rope_theta%s", "");
    c->rope_global = (float)v;
    if (!json_number(full, len - (size_t)(full - tc), "partial_rotary_factor", &v))
        die("missing partial_rotary_factor%s", "");
    c->rope_prop = (float)v;
    if (!json_number(slide, len - (size_t)(slide - tc), "rope_theta", &v))
        die("missing local rope_theta%s", "");
    c->rope_local = (float)v;
#undef REQ
#undef REQF
    free(buf);
}

/* ------------------------------------------------------------------ */
/* Conversion plan                                                     */
/* ------------------------------------------------------------------ */

typedef enum {
    TR_NONE = 0,       /* copy rows as-is */
    TR_GATE,           /* [E, D, 2H] -> gate half, rows [E, H, D] */
    TR_UP,             /* [E, D, 2H] -> up half */
    TR_DOWN,           /* [E, H, D] -> rows [E, D, Hpad] (+ zero pad) */
} transform;

typedef struct {
    char gname[96];    /* GGUF tensor name */
    char hname[192];   /* HF tensor name */
    g4q_type type;
    transform tr;
    uint32_t n_dims;
    uint64_t dims[4];  /* GGUF dims (dims[0] = contraction) */
} plan_item;

static plan_item *g_plan;
static int g_nplan, g_capplan;

static void plan(const char *gname, const char *hname, g4q_type type,
                 transform tr, uint32_t n_dims,
                 uint64_t d0, uint64_t d1, uint64_t d2) {
    if (g_nplan == g_capplan) {
        g_capplan = g_capplan ? g_capplan * 2 : 128;
        g_plan = realloc(g_plan, sizeof(plan_item) * (size_t)g_capplan);
    }
    plan_item *p = &g_plan[g_nplan++];
    memset(p, 0, sizeof(*p));
    snprintf(p->gname, sizeof(p->gname), "%s", gname);
    snprintf(p->hname, sizeof(p->hname), "%s", hname);
    p->type = type;
    p->tr = tr;
    p->n_dims = n_dims;
    p->dims[0] = d0;
    p->dims[1] = d1;
    p->dims[2] = d2;
}

static bool type_needs_pad(g4q_type t, uint64_t ncols) {
    int64_t bs = g4q_block_size(t);
    return bs > 1 && (int64_t)ncols % bs != 0;
}

static uint64_t pad_to_block(g4q_type t, uint64_t ncols) {
    int64_t bs = g4q_block_size(t);
    if (bs <= 1) return ncols;
    return (ncols + (uint64_t)bs - 1) / (uint64_t)bs * (uint64_t)bs;
}

/* Verifies the HF shape for a plan item and dies with a clear message on a
 * transposed/unexpected layout (contract: gemma4-port/02 §6). */
static void die_shape(const char *hname, const st_tensor *t, const char *want) {
    fprintf(stderr, "g4-quantize: unexpected layout for %s: got [", hname);
    for (uint32_t i = 0; i < t->n_dims; i++)
        fprintf(stderr, "%s%llu", i ? ", " : "",
                (unsigned long long)t->dims[i]);
    fprintf(stderr, "], want %s — see gemma4-port/02 §6\n", want);
    exit(1);
}

/* The two expert layouts both occur in the wild ([E, D, 2H] packed vs
 * [E, 2H, D] flattened-JAX); d_model differs from every other axis, so the
 * shape identifies the variant unambiguously — this is detection, not
 * guessing. */
static void check_shape(const g4cfg *c, const plan_item *p, const st_tensor *t) {
    const uint64_t D = c->d_model;
    const uint64_t H2x = 2ull * c->hexp;
    switch (p->tr) {
        case TR_GATE: case TR_UP:
            if (t->n_dims != 3 || t->dims[0] != c->experts ||
                !((t->dims[1] == D && t->dims[2] == H2x) ||
                  (t->dims[1] == H2x && t->dims[2] == D)))
                die_shape(p->hname, t, "[E, D, 2*Hexp] or [E, 2*Hexp, D]");
            return;
        case TR_DOWN:
            if (t->n_dims != 3 || t->dims[0] != c->experts ||
                !((t->dims[1] == c->hexp && t->dims[2] == D) ||
                  (t->dims[1] == D && t->dims[2] == c->hexp)))
                die_shape(p->hname, t, "[E, Hexp, D] or [E, D, Hexp]");
            return;
        case TR_NONE: {
            /* rows of dims[0], count = product of the rest */
            uint64_t rows = 1;
            for (uint32_t i = 1; i < p->n_dims; i++) rows *= p->dims[i];
            uint64_t want0 = p->n_dims == 1 ? p->dims[0] : rows;
            uint64_t want1 = p->dims[0];
            if (p->n_dims == 1) {
                if (t->n_dims != 1 || t->dims[0] != want0)
                    die("unexpected 1D shape for %s", p->hname);
            } else {
                if (t->n_dims != 2 || t->dims[0] != want0 || t->dims[1] != want1)
                    die("unexpected/transposed shape for %s; refusing to guess",
                        p->hname);
            }
            return;
        }
    }
}

int main(int argc, char **argv) {
    const char *hf_dir = NULL, *out_path = NULL, *cmp_name = NULL;
    const char *profile = "q8";
    bool dry_run = false;
    bool swap_gate_up = false;
    g4q_type ty_experts = G4Q_TYPE_COUNT, ty_down = G4Q_TYPE_COUNT;
    g4q_type ty_dense = G4Q_TYPE_COUNT, ty_attn = G4Q_TYPE_COUNT;
    g4q_type ty_embed = G4Q_TYPE_COUNT;

    for (int i = 1; i < argc; i++) {
#define VAL() (i + 1 < argc ? argv[++i] : (die("missing value for %s", argv[i]), (char *)0))
        if (!strcmp(argv[i], "--hf")) hf_dir = VAL();
        else if (!strcmp(argv[i], "--out")) out_path = VAL();
        else if (!strcmp(argv[i], "--profile")) profile = VAL();
        else if (!strcmp(argv[i], "--dry-run")) dry_run = true;
        else if (!strcmp(argv[i], "--swap-gate-up")) swap_gate_up = true;
        else if (!strcmp(argv[i], "--compare-tensor")) cmp_name = VAL();
        else if (!strcmp(argv[i], "--experts")) ty_experts = (g4q_type)atoi(VAL());
        else if (!strcmp(argv[i], "--routed-down")) ty_down = (g4q_type)atoi(VAL());
        else if (!strcmp(argv[i], "--dense")) ty_dense = (g4q_type)atoi(VAL());
        else if (!strcmp(argv[i], "--attn")) ty_attn = (g4q_type)atoi(VAL());
        else if (!strcmp(argv[i], "--embed")) ty_embed = (g4q_type)atoi(VAL());
        else die("unknown option %s", argv[i]);
#undef VAL
    }
    if (!hf_dir) die("--hf DIR is required%s", "");
    if (!out_path && !dry_run && !cmp_name) die("--out FILE is required%s", "");

    g4q_type def_e, def_w2, def_dense, def_attn, def_embed;
    if (!strcmp(profile, "f32")) {
        def_e = def_w2 = def_dense = def_attn = def_embed = G4Q_TYPE_F32;
    } else if (!strcmp(profile, "q8")) {
        def_e = def_w2 = def_dense = def_attn = def_embed = G4Q_TYPE_Q8_0;
    } else if (!strcmp(profile, "q4")) {
        def_e = def_w2 = G4Q_TYPE_Q4_K;
        def_dense = def_attn = def_embed = G4Q_TYPE_Q8_0;
    } else {
        die("unknown profile %s (q8|q4|f32; the q2 imatrix profile lands with M5)",
            profile);
    }
    if (ty_experts == G4Q_TYPE_COUNT) ty_experts = def_e;
    if (ty_down == G4Q_TYPE_COUNT) ty_down = def_w2;
    if (ty_dense == G4Q_TYPE_COUNT) ty_dense = def_dense;
    if (ty_attn == G4Q_TYPE_COUNT) ty_attn = def_attn;
    if (ty_embed == G4Q_TYPE_COUNT) ty_embed = def_embed;
    if (g4q_requires_imatrix(ty_experts) || g4q_requires_imatrix(ty_down))
        die("type requires an imatrix; not supported yet (M5)%s", "");

    g4cfg c;
    load_config(hf_dir, &c);
    fprintf(stderr,
            "config: %u layers, d_model %u, vocab %u, %u experts (top-%u, "
            "hexp %u), dense %u, ctx %u\n",
            c.layers, c.d_model, c.vocab, c.experts, c.top_k, c.hexp, c.dense,
            c.ctx);

    st_index ix;
    st_open_dir(&ix, hf_dir);
    fprintf(stderr, "safetensors: %d shard(s), %d tensors\n", ix.n_shards, ix.n);

    /* Q8_0 needs 32-divisible contractions; fall back to F32 per tensor
     * family when the toy dims are not divisible (real model always is). */
    const uint64_t D = c.d_model;
    const uint64_t HK_l = (uint64_t)c.heads * c.key_local;
    const uint64_t HK_g = (uint64_t)c.heads * c.key_global;
    const uint64_t down_pad = pad_to_block(ty_down, c.hexp);
#define FIT(ty, ncols) (type_needs_pad((ty), (ncols)) ? G4Q_TYPE_F32 : (ty))

    plan("token_embd.weight", "model.language_model.embed_tokens.weight",
         FIT(ty_embed, D), TR_NONE, 2, D, c.vocab, 0);
    plan("output_norm.weight", "model.language_model.norm.weight",
         G4Q_TYPE_F32, TR_NONE, 1, D, 0, 0);
    for (uint32_t i = 0; i < c.layers; i++) {
        bool glob = (i % 6) == 5;
        const uint64_t K = glob ? c.key_global : c.key_local;
        const uint64_t KV = glob ? c.kv_global : c.kv_local;
        const uint64_t HK = glob ? HK_g : HK_l;
        char g[96], h[192];
#define P(gsuf, hsuf, ty, tr, nd, d0, d1, d2) \
        do { \
            snprintf(g, sizeof(g), "blk.%u." gsuf, i); \
            snprintf(h, sizeof(h), "model.language_model.layers.%u." hsuf, i); \
            plan(g, h, (ty), (tr), (nd), (d0), (d1), (d2)); \
        } while (0)
        P("attn_norm.weight", "input_layernorm.weight", G4Q_TYPE_F32, TR_NONE, 1, D, 0, 0);
        P("post_attn_norm.weight", "post_attention_layernorm.weight", G4Q_TYPE_F32, TR_NONE, 1, D, 0, 0);
        P("attn_q.weight", "self_attn.q_proj.weight", FIT(ty_attn, D), TR_NONE, 2, D, HK, 0);
        P("attn_k.weight", "self_attn.k_proj.weight", FIT(ty_attn, D), TR_NONE, 2, D, KV * K, 0);
        if (!glob || !c.k_eq_v)
            P("attn_v.weight", "self_attn.v_proj.weight", FIT(ty_attn, D), TR_NONE, 2, D, KV * K, 0);
        P("attn_output.weight", "self_attn.o_proj.weight", FIT(ty_attn, HK), TR_NONE, 2, HK, D, 0);
        P("attn_q_norm.weight", "self_attn.q_norm.weight", G4Q_TYPE_F32, TR_NONE, 1, K, 0, 0);
        P("attn_k_norm.weight", "self_attn.k_norm.weight", G4Q_TYPE_F32, TR_NONE, 1, K, 0, 0);
        P("ffn_norm.weight", "pre_feedforward_layernorm.weight", G4Q_TYPE_F32, TR_NONE, 1, D, 0, 0);
        P("post_ffn_norm_moe.weight", "post_feedforward_layernorm_1.weight", G4Q_TYPE_F32, TR_NONE, 1, D, 0, 0);
        P("ffn_gate_inp.weight", "router.proj.weight", G4Q_TYPE_F32, TR_NONE, 2, D, c.experts, 0);
        P("router_scale", "router.scale", G4Q_TYPE_F32, TR_NONE, 1, D, 0, 0);
        P("per_expert_scale", "router.per_expert_scale", G4Q_TYPE_F32, TR_NONE, 1, c.experts, 0, 0);
        P("ffn_gate_exps.weight", "experts.gate_up_proj", FIT(ty_experts, D), TR_GATE, 3, D, c.hexp, c.experts);
        P("ffn_up_exps.weight", "experts.gate_up_proj", FIT(ty_experts, D), TR_UP, 3, D, c.hexp, c.experts);
        {
            g4q_type td = FIT(ty_down, down_pad) ;
            uint64_t d0 = td == G4Q_TYPE_F32 ? c.hexp : down_pad;
            P("ffn_down_exps.weight", "experts.down_proj", td, TR_DOWN, 3, d0, D, c.experts);
        }
        P("ffn_norm_shexp.weight", "pre_feedforward_layernorm_2.weight", G4Q_TYPE_F32, TR_NONE, 1, D, 0, 0);
        P("post_ffn_norm_shexp.weight", "post_feedforward_layernorm_2.weight", G4Q_TYPE_F32, TR_NONE, 1, D, 0, 0);
        P("ffn_gate_shexp.weight", "mlp.gate_proj.weight", FIT(ty_dense, D), TR_NONE, 2, D, c.dense, 0);
        P("ffn_up_shexp.weight", "mlp.up_proj.weight", FIT(ty_dense, D), TR_NONE, 2, D, c.dense, 0);
        P("ffn_down_shexp.weight", "mlp.down_proj.weight", FIT(ty_dense, c.dense), TR_NONE, 2, c.dense, D, 0);
        P("post_ffn_norm.weight", "post_feedforward_layernorm.weight", G4Q_TYPE_F32, TR_NONE, 1, D, 0, 0);
        P("skip_scale", "layer_scalar", G4Q_TYPE_F32, TR_NONE, 1, 1, 0, 0);
#undef P
    }
#undef FIT

    /* Validate sources and compute the total. */
    uint64_t total = 0;
    for (int i = 0; i < g_nplan; i++) {
        plan_item *p = &g_plan[i];
        const st_tensor *t = st_find(&ix, p->hname);
        if (!t) die("missing HF tensor %s", p->hname);
        check_shape(&c, p, t);
        uint64_t rows = 1;
        for (uint32_t d2 = 1; d2 < p->n_dims; d2++) rows *= p->dims[d2];
        total += g4q_row_size(p->type, (int64_t)p->dims[0]) * rows;
    }
    fprintf(stderr, "plan: %d tensors, %.2f GiB output\n", g_nplan,
            (double)total / (1024.0 * 1024.0 * 1024.0));
    if (dry_run) {
        for (int i = 0; i < g_nplan && i < 40; i++)
            fprintf(stderr, "  %-32s <- %-58s %s\n", g_plan[i].gname,
                    g_plan[i].hname, g4q_type_name(g_plan[i].type));
        if (g_nplan > 40) fprintf(stderr, "  ... (%d more)\n", g_nplan - 40);
        return 0;
    }

    g4_gguf_writer *w = NULL;
    char err[256];
    if (!cmp_name) {
        w = g4_gguf_writer_new();
        g4_gguf_writer_kv_str(w, "general.architecture", "gemma4");
        g4_gguf_writer_kv_u32(w, "gemma4.block_count", c.layers);
        g4_gguf_writer_kv_u32(w, "gemma4.embedding_length", c.d_model);
        g4_gguf_writer_kv_u32(w, "gemma4.vocab_size", c.vocab);
        g4_gguf_writer_kv_u32(w, "gemma4.context_length", c.ctx);
        g4_gguf_writer_kv_u32(w, "gemma4.attention.head_count", c.heads);
        g4_gguf_writer_kv_u32(w, "gemma4.attention.head_count_kv", c.kv_local);
        g4_gguf_writer_kv_u32(w, "gemma4.attention.global_head_count_kv", c.kv_global);
        g4_gguf_writer_kv_u32(w, "gemma4.attention.key_length", c.key_local);
        g4_gguf_writer_kv_u32(w, "gemma4.attention.global_key_length", c.key_global);
        g4_gguf_writer_kv_bool(w, "gemma4.attention.k_eq_v_global", c.k_eq_v);
        g4_gguf_writer_kv_u32(w, "gemma4.attention.sliding_window", c.sliding);
        g4_gguf_writer_kv_u32(w, "gemma4.attention.pattern_period", 6);
        g4_gguf_writer_kv_f32(w, "gemma4.attention.layer_norm_rms_epsilon", c.eps);
        g4_gguf_writer_kv_f32(w, "gemma4.rope.local.freq_base", c.rope_local);
        g4_gguf_writer_kv_f32(w, "gemma4.rope.global.freq_base", c.rope_global);
        g4_gguf_writer_kv_f32(w, "gemma4.rope.global.partial_factor", c.rope_prop);
        g4_gguf_writer_kv_u32(w, "gemma4.expert_count", c.experts);
        g4_gguf_writer_kv_u32(w, "gemma4.expert_used_count", c.top_k);
        g4_gguf_writer_kv_u32(w, "gemma4.expert_ffn_length", c.hexp);
        g4_gguf_writer_kv_u32(w, "gemma4.dense_ffn_length", c.dense);
        g4_gguf_writer_kv_f32(w, "gemma4.final_logit_softcap", c.softcap);
        for (int i = 0; i < g_nplan; i++) {
            plan_item *p = &g_plan[i];
            uint32_t nd = p->n_dims;
            while (nd > 1 && p->dims[nd - 1] == 0) nd--;
            if (g4_gguf_writer_tensor_info(w, p->gname, p->type, nd, p->dims))
                die("cannot declare tensor %s", p->gname);
        }
        if (g4_gguf_writer_begin(w, out_path, err, sizeof(err)))
            die("cannot start output: %s", err);
    }

    for (int i = 0; i < g_nplan; i++) {
        plan_item *p = &g_plan[i];
        if (cmp_name && strcmp(p->gname, cmp_name)) continue;
        const st_tensor *t = st_find(&ix, p->hname);
        uint64_t n_src;
        float *src = st_load_f32(&ix, t, &n_src);

        /* Materialize the GGUF-layout f32 rows. */
        uint64_t ncols = p->dims[0];
        uint64_t rows = 1;
        for (uint32_t d2 = 1; d2 < p->n_dims; d2++)
            if (p->dims[d2]) rows *= p->dims[d2];
        float *rowsbuf;
        if (p->tr == TR_NONE) {
            rowsbuf = src;
        } else {
            rowsbuf = xmalloc(sizeof(float) * (size_t)rows * ncols);
            const uint64_t E = c.experts, H = c.hexp;
            if (p->tr == TR_GATE || p->tr == TR_UP) {
                bool want_gate = (p->tr == TR_GATE) != swap_gate_up;
                uint64_t half = want_gate ? 0 : H;
                /* Prefer the real google layout [E, 2*H, D] when the shape is
                 * ambiguous (2*H == D): dims[1]==2*H matches it first. */
                if (t->dims[1] == 2 * H) {
                    /* src [E, 2H, D] -> rows [(e,h)][d] (contiguous copy) */
                    for (uint64_t e = 0; e < E; e++)
                        for (uint64_t hh = 0; hh < H; hh++)
                            memcpy(rowsbuf + (e * H + hh) * D,
                                   src + (e * 2 * H + half + hh) * D,
                                   sizeof(float) * (size_t)D);
                } else {
                    /* src [E, D, 2H] -> rows [(e,h)][d] */
                    for (uint64_t e = 0; e < E; e++)
                        for (uint64_t hh = 0; hh < H; hh++)
                            for (uint64_t d2 = 0; d2 < D; d2++)
                                rowsbuf[(e * H + hh) * D + d2] =
                                    src[(e * D + d2) * 2 * H + half + hh];
                }
            } else { /* TR_DOWN -> rows [(e,d)][h(+pad)] */
                memset(rowsbuf, 0, sizeof(float) * (size_t)rows * ncols);
                if (t->dims[1] == c.hexp) {
                    /* src [E, H, D] */
                    for (uint64_t e = 0; e < E; e++)
                        for (uint64_t d2 = 0; d2 < D; d2++)
                            for (uint64_t hh = 0; hh < H; hh++)
                                rowsbuf[(e * D + d2) * ncols + hh] =
                                    src[(e * H + hh) * D + d2];
                } else {
                    /* src [E, D, H] (contiguous copy into padded rows) */
                    for (uint64_t e = 0; e < E; e++)
                        for (uint64_t d2 = 0; d2 < D; d2++)
                            memcpy(rowsbuf + (e * D + d2) * ncols,
                                   src + (e * D + d2) * H,
                                   sizeof(float) * (size_t)H);
                }
            }
        }

        size_t row_bytes = g4q_row_size(p->type, (int64_t)ncols);
        uint8_t *enc = xmalloc(row_bytes * (size_t)rows);
        if (p->type == G4Q_TYPE_F32) {
            memcpy(enc, rowsbuf, sizeof(float) * (size_t)rows * ncols);
        } else {
            g4q_quantize_init(p->type);
            g4q_quantize_chunk(p->type, rowsbuf, enc, 0, (int64_t)rows,
                               (int64_t)ncols, NULL);
        }

        if (cmp_name) {
            float *back = xmalloc(sizeof(float) * (size_t)rows * ncols);
            for (uint64_t r = 0; r < rows; r++)
                g4q_dequant_row(p->type, enc + r * row_bytes,
                                back + r * ncols, (int64_t)ncols);
            double num = 0, den = 0;
            for (uint64_t k = 0; k < rows * ncols; k++) {
                double d2 = (double)back[k] - rowsbuf[k];
                num += d2 * d2;
                den += (double)rowsbuf[k] * rowsbuf[k];
            }
            printf("%s (%s, %llu x %llu): rel-rms-err %.5f\n", p->gname,
                   g4q_type_name(p->type), (unsigned long long)rows,
                   (unsigned long long)ncols,
                   den > 0 ? sqrt(num / den) : 0.0);
            free(back);
        } else {
            if (g4_gguf_writer_put(w, enc, row_bytes * rows))
                die("write failed for %s", p->gname);
            fprintf(stderr, "\r[%d/%d] %-40s", i + 1, g_nplan, p->gname);
        }

        free(enc);
        if (rowsbuf != src) free(rowsbuf);
        free(src);
    }

    if (!cmp_name) {
        if (g4_gguf_writer_end(w, err, sizeof(err)))
            die("cannot finish output: %s", err);
        fprintf(stderr, "\nwrote %s\n", out_path);
    }
    return 0;
}
