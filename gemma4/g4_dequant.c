/* Dequantization for the g4 engine test/reference path.
 *
 * Decode conventions match the ds4.c engine kernels exactly:
 * IQ2_XXS uses the GGML grid (byte values 8/25/43), the ksigns/kmask
 * sign tables and the 0.125 * d * (2*ls+1) scaling of the ds4.c scalar
 * path in ds4_vec_dot_iq2_xxs_q8_K.  Q2_K/Q4_K/Q8_0 match the writer
 * layout in g4_quants.c. */

#include "g4_quants.h"

#include <assert.h>
#include <string.h>

static const uint8_t kmask_iq2xs[8] = {
    1, 2, 4, 8, 16, 32, 64, 128
};

static const uint8_t ksigns_iq2xs[128] = {
      0, 129, 130,   3, 132,   5,   6, 135, 136,   9,  10, 139,  12, 141, 142,  15,
    144,  17,  18, 147,  20, 149, 150,  23,  24, 153, 154,  27, 156,  29,  30, 159,
    160,  33,  34, 163,  36, 165, 166,  39,  40, 169, 170,  43, 172,  45,  46, 175,
     48, 177, 178,  51, 180,  53,  54, 183, 184,  57,  58, 187,  60, 189, 190,  63,
    192,  65,  66, 195,  68, 197, 198,  71,  72, 201, 202,  75, 204,  77,  78, 207,
     80, 209, 210,  83, 212,  85,  86, 215, 216,  89,  90, 219,  92, 221, 222,  95,
     96, 225, 226,  99, 228, 101, 102, 231, 232, 105, 106, 235, 108, 237, 238, 111,
    240, 113, 114, 243, 116, 245, 246, 119, 120, 249, 250, 123, 252, 125, 126, 255,
};

static const uint64_t iq2xxs_grid[256] = {
    0x0808080808080808, 0x080808080808082b, 0x0808080808081919, 0x0808080808082b08,
    0x0808080808082b2b, 0x0808080808190819, 0x0808080808191908, 0x08080808082b0808,
    0x08080808082b082b, 0x08080808082b2b08, 0x08080808082b2b2b, 0x0808080819080819,
    0x0808080819081908, 0x0808080819190808, 0x0808080819192b08, 0x08080808192b0819,
    0x08080808192b1908, 0x080808082b080808, 0x080808082b08082b, 0x080808082b082b2b,
    0x080808082b2b082b, 0x0808081908080819, 0x0808081908081908, 0x0808081908190808,
    0x0808081908191919, 0x0808081919080808, 0x080808192b081908, 0x080808192b192b08,
    0x0808082b08080808, 0x0808082b0808082b, 0x0808082b082b082b, 0x0808082b2b08082b,
    0x0808190808080819, 0x0808190808081908, 0x0808190808190808, 0x08081908082b0819,
    0x08081908082b1908, 0x0808190819080808, 0x080819081908082b, 0x0808190819082b08,
    0x08081908192b0808, 0x080819082b080819, 0x080819082b081908, 0x080819082b190808,
    0x080819082b2b1908, 0x0808191908080808, 0x080819190808082b, 0x0808191908082b08,
    0x08081919082b0808, 0x080819191908192b, 0x08081919192b2b19, 0x080819192b080808,
    0x080819192b190819, 0x0808192b08082b19, 0x0808192b08190808, 0x0808192b19080808,
    0x0808192b2b081908, 0x0808192b2b2b1908, 0x08082b0808080808, 0x08082b0808081919,
    0x08082b0808082b08, 0x08082b0808191908, 0x08082b08082b2b08, 0x08082b0819080819,
    0x08082b0819081908, 0x08082b0819190808, 0x08082b081919082b, 0x08082b082b082b08,
    0x08082b1908081908, 0x08082b1919080808, 0x08082b2b0808082b, 0x08082b2b08191908,
    0x0819080808080819, 0x0819080808081908, 0x0819080808190808, 0x08190808082b0819,
    0x0819080819080808, 0x08190808192b0808, 0x081908082b081908, 0x081908082b190808,
    0x081908082b191919, 0x0819081908080808, 0x0819081908082b08, 0x08190819082b0808,
    0x0819081919190808, 0x0819081919192b2b, 0x081908192b080808, 0x0819082b082b1908,
    0x0819082b19081919, 0x0819190808080808, 0x0819190808082b08, 0x08191908082b0808,
    0x08191908082b1919, 0x0819190819082b19, 0x081919082b080808, 0x0819191908192b08,
    0x08191919192b082b, 0x0819192b08080808, 0x0819192b0819192b, 0x08192b0808080819,
    0x08192b0808081908, 0x08192b0808190808, 0x08192b0819080808, 0x08192b082b080819,
    0x08192b1908080808, 0x08192b1908081919, 0x08192b192b2b0808, 0x08192b2b19190819,
    0x082b080808080808, 0x082b08080808082b, 0x082b080808082b2b, 0x082b080819081908,
    0x082b0808192b0819, 0x082b08082b080808, 0x082b08082b08082b, 0x082b0819082b2b19,
    0x082b081919082b08, 0x082b082b08080808, 0x082b082b0808082b, 0x082b190808080819,
    0x082b190808081908, 0x082b190808190808, 0x082b190819080808, 0x082b19081919192b,
    0x082b191908080808, 0x082b191919080819, 0x082b1919192b1908, 0x082b192b2b190808,
    0x082b2b0808082b08, 0x082b2b08082b0808, 0x082b2b082b191908, 0x082b2b2b19081908,
    0x1908080808080819, 0x1908080808081908, 0x1908080808190808, 0x1908080808192b08,
    0x19080808082b0819, 0x19080808082b1908, 0x1908080819080808, 0x1908080819082b08,
    0x190808081919192b, 0x19080808192b0808, 0x190808082b080819, 0x190808082b081908,
    0x190808082b190808, 0x1908081908080808, 0x19080819082b0808, 0x19080819192b0819,
    0x190808192b080808, 0x190808192b081919, 0x1908082b08080819, 0x1908082b08190808,
    0x1908082b19082b08, 0x1908082b1919192b, 0x1908082b192b2b08, 0x1908190808080808,
    0x1908190808082b08, 0x19081908082b0808, 0x190819082b080808, 0x190819082b192b19,
    0x190819190819082b, 0x19081919082b1908, 0x1908192b08080808, 0x19082b0808080819,
    0x19082b0808081908, 0x19082b0808190808, 0x19082b0819080808, 0x19082b0819081919,
    0x19082b1908080808, 0x19082b1919192b08, 0x19082b19192b0819, 0x19082b192b08082b,
    0x19082b2b19081919, 0x19082b2b2b190808, 0x1919080808080808, 0x1919080808082b08,
    0x1919080808190819, 0x1919080808192b19, 0x19190808082b0808, 0x191908082b080808,
    0x191908082b082b08, 0x1919081908081908, 0x191908191908082b, 0x191908192b2b1908,
    0x1919082b2b190819, 0x191919082b190808, 0x191919082b19082b, 0x1919191908082b2b,
    0x1919192b08080819, 0x1919192b19191908, 0x19192b0808080808, 0x19192b0808190819,
    0x19192b0808192b19, 0x19192b08192b1908, 0x19192b1919080808, 0x19192b2b08082b08,
    0x192b080808081908, 0x192b080808190808, 0x192b080819080808, 0x192b0808192b2b08,
    0x192b081908080808, 0x192b081919191919, 0x192b082b08192b08, 0x192b082b192b0808,
    0x192b190808080808, 0x192b190808081919, 0x192b191908190808, 0x192b19190819082b,
    0x192b19192b081908, 0x192b2b081908082b, 0x2b08080808080808, 0x2b0808080808082b,
    0x2b08080808082b2b, 0x2b08080819080819, 0x2b0808082b08082b, 0x2b08081908081908,
    0x2b08081908192b08, 0x2b08081919080808, 0x2b08082b08190819, 0x2b08190808080819,
    0x2b08190808081908, 0x2b08190808190808, 0x2b08190808191919, 0x2b08190819080808,
    0x2b081908192b0808, 0x2b08191908080808, 0x2b0819191908192b, 0x2b0819192b191908,
    0x2b08192b08082b19, 0x2b08192b19080808, 0x2b08192b192b0808, 0x2b082b080808082b,
    0x2b082b1908081908, 0x2b082b2b08190819, 0x2b19080808081908, 0x2b19080808190808,
    0x2b190808082b1908, 0x2b19080819080808, 0x2b1908082b2b0819, 0x2b1908190819192b,
    0x2b1908192b080808, 0x2b19082b19081919, 0x2b19190808080808, 0x2b191908082b082b,
    0x2b19190819081908, 0x2b19191919190819, 0x2b192b082b080819, 0x2b192b19082b0808,
    0x2b2b08080808082b, 0x2b2b080819190808, 0x2b2b08082b081919, 0x2b2b081908082b19,
    0x2b2b082b08080808, 0x2b2b190808192b08, 0x2b2b2b0819190808, 0x2b2b2b1908081908,
};

/* Q8_0: 32 weights per 34-byte block, y = d * q. */
static void g4q_dequant_row_q8_0(const uint8_t *src, float *dst, int64_t ncols) {
    for (int64_t b = 0; b < ncols / 32; b++) {
        const uint8_t *blk = src + (size_t)b * 34;
        uint16_t hd;
        memcpy(&hd, blk, sizeof(hd));
        const float d = g4q_f16_to_f32(hd);
        const int8_t *qs = (const int8_t *)(blk + 2);
        for (int j = 0; j < 32; j++) dst[32 * b + j] = d * qs[j];
    }
}

/* Q2_K: 256 weights per 84-byte block: scales[16] | qs[64] | d f16 | dmin f16.
 * Sub-block j of 16 weights: y = d*(sc&0xF)*q - dmin*(sc>>4), q in 0..3. */
static void g4q_dequant_row_q2_k(const uint8_t *src, float *dst, int64_t ncols) {
    for (int64_t b = 0; b < ncols / 256; b++) {
        const uint8_t *blk = src + (size_t)b * 84;
        const uint8_t *scales = blk;
        const uint8_t *qs = blk + 16;
        uint16_t hd, hmin;
        memcpy(&hd, blk + 80, sizeof(hd));
        memcpy(&hmin, blk + 82, sizeof(hmin));
        const float d = g4q_f16_to_f32(hd);
        const float dmin = g4q_f16_to_f32(hmin);
        float *y = dst + 256 * b;
        for (int j = 0; j < 256; j += 128) {
            for (int l = 0; l < 32; l++) {
                const uint8_t q = qs[j / 4 + l];
                const int idx[4] = { j + l, j + l + 32, j + l + 64, j + l + 96 };
                for (int k = 0; k < 4; k++) {
                    const uint8_t sc = scales[idx[k] / 16];
                    const int qv = (q >> (2 * k)) & 3;
                    y[idx[k]] = d * (sc & 0xF) * qv - dmin * (sc >> 4);
                }
            }
        }
    }
}

static void g4q_get_scale_min_k4_pub(int j, const uint8_t *q, uint8_t *d, uint8_t *m) {
    if (j < 4) {
        *d = q[j] & 63;
        *m = q[j + 4] & 63;
    } else {
        *d = (q[j + 4] & 0xF) | ((q[j - 4] >> 6) << 4);
        *m = (q[j + 4] >> 4) | ((q[j - 0] >> 6) << 4);
    }
}

/* Q4_K: 256 weights per 144-byte block: d f16 | dmin f16 | scales[12] | qs[128].
 * Sub-block j of 32 weights: y = d*sc6*q - dmin*m6, q in 0..15. */
static void g4q_dequant_row_q4_k(const uint8_t *src, float *dst, int64_t ncols) {
    for (int64_t b = 0; b < ncols / 256; b++) {
        const uint8_t *blk = src + (size_t)b * 144;
        uint16_t hd, hmin;
        memcpy(&hd, blk + 0, sizeof(hd));
        memcpy(&hmin, blk + 2, sizeof(hmin));
        const float d = g4q_f16_to_f32(hd);
        const float dmin = g4q_f16_to_f32(hmin);
        const uint8_t *scales = blk + 4;
        const uint8_t *qs = blk + 16;
        float *y = dst + 256 * b;
        for (int j = 0; j < 256; j += 64) {
            const int sb = j / 32;
            uint8_t sc0, m0, sc1, m1;
            g4q_get_scale_min_k4_pub(sb, scales, &sc0, &m0);
            g4q_get_scale_min_k4_pub(sb + 1, scales, &sc1, &m1);
            const uint8_t *q = qs + (j / 64) * 32;
            for (int l = 0; l < 32; l++) {
                y[j + l]      = d * sc0 * (q[l] & 0xF) - dmin * m0;
                y[j + l + 32] = d * sc1 * (q[l] >> 4)  - dmin * m1;
            }
        }
    }
}

/* IQ2_XXS: 256 weights per 66-byte block: d f16 | 8 x { u32 grid indices,
 * u32 sign indices + 4-bit scale }.  Same decode as the ds4.c engine:
 * y = 0.125 * d * (2*ls+1) * grid_byte * sign. */
static void g4q_dequant_row_iq2_xxs(const uint8_t *src, float *dst, int64_t ncols) {
    for (int64_t b = 0; b < ncols / 256; b++) {
        const uint8_t *blk = src + (size_t)b * 66;
        uint16_t hd;
        memcpy(&hd, blk, sizeof(hd));
        const float d = g4q_f16_to_f32(hd);
        float *y = dst + 256 * b;
        for (int ib = 0; ib < 8; ib++) {
            uint32_t aux[2];
            memcpy(aux, blk + 2 + 8 * ib, sizeof(aux));
            const uint8_t *aux8 = (const uint8_t *)&aux[0];
            const float db = 0.125f * d * (2 * (aux[1] >> 28) + 1);
            for (int k = 0; k < 4; k++) {
                const uint8_t *grid = (const uint8_t *)(iq2xxs_grid + aux8[k]);
                const uint8_t signs = ksigns_iq2xs[(aux[1] >> (7 * k)) & 127];
                for (int j = 0; j < 8; j++) {
                    const float v = db * grid[j];
                    y[32 * ib + 8 * k + j] = (signs & kmask_iq2xs[j]) ? -v : v;
                }
            }
        }
    }
}

bool g4q_can_dequantize(g4q_type type) {
    switch (type) {
        case G4Q_TYPE_F32:
        case G4Q_TYPE_F16:
        case G4Q_TYPE_BF16:
        case G4Q_TYPE_Q8_0:
        case G4Q_TYPE_Q2_K:
        case G4Q_TYPE_Q4_K:
        case G4Q_TYPE_IQ2_XXS:
            return true;
        default:
            return false;
    }
}

void g4q_dequant_row(g4q_type type, const void *src, float *dst, int64_t ncols) {
    const uint8_t *s = (const uint8_t *)src;
    switch (type) {
        case G4Q_TYPE_F32:
            memcpy(dst, src, (size_t)ncols * sizeof(float));
            break;
        case G4Q_TYPE_F16:
            for (int64_t i = 0; i < ncols; i++)
                dst[i] = g4q_f16_to_f32(((const uint16_t *)src)[i]);
            break;
        case G4Q_TYPE_BF16:
            for (int64_t i = 0; i < ncols; i++)
                dst[i] = g4q_bf16_to_f32(((const uint16_t *)src)[i]);
            break;
        case G4Q_TYPE_Q8_0:
            g4q_dequant_row_q8_0(s, dst, ncols);
            break;
        case G4Q_TYPE_Q2_K:
            g4q_dequant_row_q2_k(s, dst, ncols);
            break;
        case G4Q_TYPE_Q4_K:
            g4q_dequant_row_q4_k(s, dst, ncols);
            break;
        case G4Q_TYPE_IQ2_XXS:
            g4q_dequant_row_iq2_xxs(s, dst, ncols);
            break;
        default:
            assert(!"g4q_dequant_row: unsupported type");
    }
}
