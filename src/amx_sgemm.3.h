#pragma once

#include "../dougallj/amx.h"

#include <stdlib.h>

/*
 *  amx registers
 *  row: a row contains 16 floats or 32 uint16_t or 64 uint8_t or others
 *  register (groups) x, y and z
 *  x and y contains 8 rows
 *  z contains 64 rows
 */

/*
 *  Load data to a register row
 *
 *  addr: base addr, 0x40 or 0x80 bytes aligin
 *  offset: in register z, 0x40 bytes per step
 *  mode: decide load 0x40 bytes with 0 or 0x80 with 1
 */
void amx_ldz(const uint8_t *addr, uint64_t offset, uint64_t mode)
{
    AMX_LDZ(((mode & 1ull) << 62) |
            ((offset & ((1ull << 6) - 1)) << 56) |
            (((uint64_t)addr & ((1ull << 56) - 1))));
}

/* same as amx_ldz, but store */
void amx_stz(uint8_t *addr, uint64_t offset, uint64_t mode)
{
    AMX_STZ(((mode & 1ull) << 62) |
            ((offset & ((1ull << 6) - 1)) << 56) |
            (((uint64_t)addr & ((1ull << 56) - 1))));
}

/* same as amx_ldz, but offset is limit from 0 to 7 */
void amx_ldx(uint8_t *addr, uint64_t offset, uint64_t mode)
{
    AMX_LDX(((mode & 1ull) << 62) |
            ((offset & ((1ull << 3) - 1)) << 56) |
            (((uint64_t)addr & ((1ull << 56) - 1))));
}

/* same as amx_ldx */
void amx_ldy(uint8_t *addr, uint64_t offset, uint64_t mode)
{
    AMX_LDY(((mode & 1ull) << 62) |
            ((offset & ((1ull << 3) - 1)) << 56) |
            (((uint64_t)addr & ((1ull << 56) - 1))));
}

/* same as amx_ldx, but store */
void amx_stx(uint8_t *addr, uint64_t offset, uint64_t mode)
{
    AMX_STX(((mode & 1ull) << 62) |
            ((offset & ((1ull << 3) - 1)) << 56) |
            (((uint64_t)addr & ((1ull << 56) - 1))));
}

/* same as amx_ldy */
void amx_sty(uint8_t *addr, uint64_t offset, uint64_t mode)
{
    AMX_STY(((mode & 1ull) << 62) |
            ((offset & ((1ull << 3) - 1)) << 56) |
            (((uint64_t)addr & ((1ull << 56) - 1))));
}

/*
 *  amx float multiply add
 *  16 floats in a register x row as a vertical vecoter, 
 *  and 16 floats a in register y row as a horizontal vecoter.
 *  matrix multiply then and get a 16*16 matirx stored in regiter z
 *  
 *  xoffset, yoffset with 0x40 bytes step
 *  zoffset, 0~63, the value used there is i = zoffset & 0x3
 *      for float operations i select the rows in z [i, 4 + i, ..., 60 + i]
 *  zignore, if setted, then the old value in z is not added
 */
void amx_fma32(uint64_t xoffset, uint64_t yoffset, uint64_t zoffset, uint64_t zignore)
{
    AMX_FMA32((yoffset << 6) |
              (xoffset << 6 << 10) |
              (zoffset << 20) |
              (zignore << 27));
}

void transformA(const float *A, float *A0, uint64_t sizei, uint64_t sizek)
{
    for (uint64_t i = 0ull; i < sizei; i += 32ull)
    {
        for (uint64_t k = 0ull; k < sizek; k += 32ull)
        {
            // load A[i:i+32][k:k+32]
            for (uint64_t ii = 0ull; ii < 16ull; ii++)
            {
                amx_ldz((uint8_t *)(&A[(i + 00ull + ii) * sizek + k]), ii << 2, 1ull);
                amx_ldz((uint8_t *)(&A[(i + 16ull + ii) * sizek + k]), (ii << 2) + 2ull, 1ull);
            }
            // transpose z to x, y
            uint64_t oprand_to_x = 0x8000000004004000;
            uint64_t oprand_to_y = 0x8000000010004000;
            uint64_t zoffsets[4] = {0ull, 32ull, 1ull, 33ull};
            for (int z = 0; z < 4; z++)
            {
                uint64_t zoffset = zoffsets[z];
                for (uint64_t offset = 0ull; offset < 8ull; offset += 2ull)
                {
                    AMX_EXTRY(oprand_to_x | (((offset >> 1 << 2) + zoffset + 0ull) << 20) | (offset << 6));
                    AMX_EXTRY(oprand_to_x | (((offset >> 1 << 2) + zoffset + 2ull) << 20) | ((offset + 1ull) << 6));
                }
                for (uint64_t offset = 0ull; offset < 8ull; offset += 2ull)
                {
                    amx_stx((uint8_t *)A0, offset, 1ull);
                    A0 += 32;
                }
                zoffset += (4ull << 2);
                for (uint64_t offset = 0ull; offset < 8ull; offset += 2ull)
                {
                    AMX_EXTRY(oprand_to_y | (((offset >> 1 << 2) + zoffset + 0ull) << 20) | (offset << 6));
                    AMX_EXTRY(oprand_to_y | (((offset >> 1 << 2) + zoffset + 2ull) << 20) | ((offset + 1ull) << 6));
                }
                for (uint64_t offset = 0ull; offset < 8ull; offset += 2ull)
                {
                    amx_sty((uint8_t *)A0, offset, 1ull);
                    A0 += 32;
                }
            }
        }
    }
}

void transformB(const float *B, float *B0, uint64_t sizek, uint64_t sizej)
{
    for (uint64_t j = 0; j < sizej; j += 32ull)
    {
        for (uint64_t k = 0; k < sizek; k++)
        {
            memcpy(B0, &B[sizej * k + j], sizeof(float) * 32);
            B0 += 32;
        }
    }
}

void _amx_sgemm(float *A, float *B, float *C, const uint64_t sizei, const uint64_t sizej, const uint64_t sizek)
{
    // limitation
    if (sizei == 0ull || sizek == 0ull || sizej == 0ull)
        return;
    if (sizei % 32 != 0 || sizek % 32 != 0 || sizej % 32 != 0)
        return;
    // A0[sizei / 32][sizek][2][16]
    float *A0 = (float *)aligned_alloc(128, sizei * sizek * sizeof(float));
    // B0[sizej / 32][sizek][2][16]
    float *B0 = (float *)aligned_alloc(128, sizek * sizej * sizeof(float));

    transformB(B, B0, sizek, sizej);
    AMX_START();
    transformA(A, A0, sizei, sizek);
    // 2 * 16 rows of A as a row tile
    for (uint64_t i = 0ull; i < sizei; i += 32ull)
    {
        float *A0i = A0 + i * sizek;
        // 2 * 16 columns of B as a column tile
        for (uint64_t j = 0ull; j < sizej; j += 32ull)
        {
            float *B0j = B0 + j * sizek;
            for (uint64_t k = 0ull; k < sizek; k += 1ull)
            {
                // load A[i:i+32][k]
                amx_ldy((uint8_t *)(A0i + 32ull * k), 0ull, 1ull);
                // load B[k][j:j+32]
                amx_ldx((uint8_t *)(B0j + 32ull * k), 0ull, 1ull);

                uint64_t zignore = k == 0ull ? 1ull : 0ull;
                // fma32 A[i:i+16][k] B[k][j:j+16]
                amx_fma32(0ull, 0ull, 0ull, zignore);
                // fma32 A[i:i+16][k] B[k][j+16:j+32]
                amx_fma32(1ull, 0ull, 1ull, zignore);
                // fma32 A[i+16:i+32][k] B[k][j:j+16]
                amx_fma32(0ull, 1ull, 2ull, zignore);
                // fma32 A[i+16:i+32][k] B[k][j+16:j+32]
                amx_fma32(1ull, 1ull, 3ull, zignore);
            }
            float *Cij = C + i * sizej + j; // C[i][j]
            for (uint64_t offset = 0; offset < 16ull; offset++)
            {
                amx_stz((uint8_t *)(Cij + sizej * offset), (offset << 2), 1ull);
                amx_stz((uint8_t *)(Cij + sizej * (offset + 16ull)), (offset << 2) + 2ull, 1ull);
            }
        }
    }
    AMX_STOP();
    free(A0);
    free(B0);
}

void amx_sgemm(float *A, float *B, float *C, const uint64_t size)
{
    _amx_sgemm(A, B, C, size, size, size);
    return;
}
