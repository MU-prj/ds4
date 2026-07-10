#ifndef G4_QUANTS_H
#define G4_QUANTS_H

/*
 * Narrow quantization API used by the DS4 GGUF writer.
 *
 * The enum values intentionally match GGUF/GGML type IDs so template metadata
 * can be copied without translation.  Only the formats used by the DS4 Flash
 * quantization recipes are implemented as output targets.
 */

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

#define G4Q_MAX_DIMS 4

typedef enum {
    G4Q_TYPE_F32     = 0,
    G4Q_TYPE_F16     = 1,
    G4Q_TYPE_Q4_0    = 2,
    G4Q_TYPE_Q4_1    = 3,
    G4Q_TYPE_Q5_0    = 6,
    G4Q_TYPE_Q5_1    = 7,
    G4Q_TYPE_Q8_0    = 8,
    G4Q_TYPE_Q8_1    = 9,
    G4Q_TYPE_Q2_K    = 10,
    G4Q_TYPE_Q3_K    = 11,
    G4Q_TYPE_Q4_K    = 12,
    G4Q_TYPE_Q5_K    = 13,
    G4Q_TYPE_Q6_K    = 14,
    G4Q_TYPE_Q8_K    = 15,
    G4Q_TYPE_IQ2_XXS = 16,
    G4Q_TYPE_IQ2_XS  = 17,
    G4Q_TYPE_IQ3_XXS = 18,
    G4Q_TYPE_IQ1_S   = 19,
    G4Q_TYPE_IQ4_NL  = 20,
    G4Q_TYPE_IQ3_S   = 21,
    G4Q_TYPE_IQ2_S   = 22,
    G4Q_TYPE_IQ4_XS  = 23,
    G4Q_TYPE_I8      = 24,
    G4Q_TYPE_I16     = 25,
    G4Q_TYPE_I32     = 26,
    G4Q_TYPE_I64     = 27,
    G4Q_TYPE_F64     = 28,
    G4Q_TYPE_IQ1_M   = 29,
    G4Q_TYPE_BF16    = 30,
    G4Q_TYPE_TQ1_0   = 34,
    G4Q_TYPE_TQ2_0   = 35,
    G4Q_TYPE_MXFP4   = 39,
    G4Q_TYPE_NVFP4   = 40,
    G4Q_TYPE_Q1_0    = 41,
    G4Q_TYPE_COUNT   = 42,
} g4q_type;

static inline size_t g4q_pad(size_t x, size_t n) {
    return ((x + n - 1) / n) * n;
}

const char *g4q_type_name(g4q_type type);
bool g4q_can_quantize(g4q_type type);
int64_t g4q_block_size(g4q_type type);
size_t g4q_row_size(g4q_type type, int64_t ne);
bool g4q_requires_imatrix(g4q_type type);
void g4q_quantize_init(g4q_type type);
size_t g4q_quantize_chunk(g4q_type type, const float *src, void *dst,
                           int64_t start, int64_t nrows, int64_t ncols,
                           const float *imatrix);

float g4q_f16_to_f32(uint16_t bits);
float g4q_bf16_to_f32(uint16_t bits);
void g4q_f32_to_f16_row(const float *src, uint16_t *dst, int64_t n);
void g4q_f32_to_bf16_row(const float *src, uint16_t *dst, int64_t n);

/* Reference dequantization (g4_dequant.c).  Decode conventions match the
 * ds4.c engine kernels; used by tests, the converter and the CPU path. */
bool g4q_can_dequantize(g4q_type type);
void g4q_dequant_row(g4q_type type, const void *src, float *dst, int64_t ncols);

#endif
