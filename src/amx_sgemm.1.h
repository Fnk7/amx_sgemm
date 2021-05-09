#pragma once

#include "../dougallj/amx.h"

/*
 *  amx registers
 *  row: a row contains 16 floats or 32 uint16_t or 64 uint8_t or others
 *  register (groups) x, y and z
 *  x and y contains 8 rows
 *  z contains 64 rows
 */

/*
 *  Load data to a register row
 *  addr base addr, 0x40 or 0x80 bytes aligin
 *  offset in register z, 0x40 bytes per step
 *  mode decide load 0x40 bytes with 0 or 0x80 with 1
 */
void amx_ldz(uint8_t *addr, uint64_t offset, uint64_t mode)
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

/*
 *  store a tile data from register z to C
 *  
 *  C is the address of C_[i][j], where C_ is parameter of sgemm
 *  C_[i:i+16][j:j+32] is stored
 */
void store_Z_to_C(float *C, uint64_t size)
{
    for (uint64_t offset = 0; offset < 16ull; offset++)
    {
        // store C_[i+offset][j:j+32]
        amx_stz((uint8_t *)(C + size * offset), (offset << 2), 1ull);
    }
}

/*
 *  load a tile data from A to register z
 *  and wait for transpose
 *  
 *  A is the address of A_[i][k], where A_ is parameter of sgemm
 *  A_[i:i+16][k:k+16] is loaded
 */
void load_A_to_Z(float *A, uint64_t size)
{
    for (uint64_t offest = 0; offest < 16ull; offest++)
    {
        // load A_[i+offset][k:k+16]
        amx_ldz((uint8_t *)(A + size * offest), (offest << 2) + 2ull, 0ull);
    }
}

/*
 *  transpose columns in register z, and store into register y
 *  it prepares the 8 columns (16 floats per column) data of A used by fma32 
 * 
 *  extry.c have more infomation
 */
void transpose_Z_to_Y(uint64_t start)
{
    // move column in z to y
    const uint64_t const_oprand = 0x8000000010004000;
    uint64_t oprand = const_oprand | ((2ull + start) << 20);
    for (uint64_t i = 0; i < 8ull; i++)
    {
        AMX_EXTRY(oprand);
        oprand += ((1ull << 2) << 20); // zoffset
        oprand += (1ull << 6);         // yoffset plus 16 float
    }
}

/*
 *  load a tile data from B to register x
 *  it prepares the 2 tile of 4 rows (16 floats per row) data of B used by fma32 
 * 
 *  extry.c have more infomation
 */
void load_B_to_X(float *B, uint64_t size)
{
    for (uint64_t i = 0; i < 4; i++)
    {
        amx_ldx((uint8_t *)(B + size * i), i << 1, 1ull);
    }
}

void _amx_sgemm(float *A, float *B, float *C, const uint64_t sizei, const uint64_t sizej, const uint64_t sizek);

void amx_sgemm(float *A, float *B, float *C, const uint64_t size)
{
    _amx_sgemm(A, B, C, size, size, size);
    return;
    // limitation
    if (size == 0ull)
        return;
    if (size % 32 != 0)
        return;
    AMX_START();
    // 16 rows of A as a row tile
    for (uint64_t i = 0ull; i < size; i += 16ull)
    {
        float *Ai = A + i * size;
        // 2 * 16 columns of B as a column tile
        for (uint64_t j = 0ull; j < size; j += 32ull)
        {
            float *Bj = B + j;
            /*  for all 16 vertical elements of the row tile of A (16 rows), 
             *  and for all horizontal elements of the column tile of B (2 * 16 columns)
             *  
             *  like for all elements of a row of A,
             *  for all elements of a column of B,
             *  plus them one by one and then sum,
             *  but there are 16 columns/rows
             */
            for (uint64_t k = 0ull; k < size; k += 16ull)
            {
                float *Aik = Ai + k;        // A[i][k]
                float *Bkj = Bj + k * size; // B[k][j]
                load_A_to_Z(Aik, size);     // load A[i:i+16][k:k+16]
                for (uint64_t l = 0ull; l < 4; l++)
                {
                    // load B[k+l*4:k+l*4+4][j:j+2*16]
                    load_B_to_X(Bkj + (l << 2) * size, size);
                    // if l == 0
                    // transpose columns 0~8, A[i:i+16][k:k+8]
                    // else
                    // transpose columns 8~16, A[i:i+16][k+8:k+16]
                    if (l % 2 == 0)
                        // >> 1, 2nd bit of l. << 3 column offset. << 2 float size.
                        transpose_Z_to_Y(l >> 1 << 3 << 2);
                    for (uint64_t m = 0; m < 4; m++)
                    {
                        uint64_t xoffset = m << 1;
                        // ((l & 1ull) << 2) select 0~3 or 4~7 rows of register y
                        uint64_t yoffset = m + ((l & 1ull) << 2);
                        uint64_t zignore = k == 0ull && l == 0ull && m == 0ull ? 1ull : 0ull;
                        // fma32 A[i:i+16][k+op1(l,m)] and B[k+op2(l,m)][j:j+16]
                        amx_fma32(xoffset, yoffset, 0ull, zignore);
                        xoffset += 1ull;
                        // fma32 A[i:i+16][k+op1(l,m)] and B[k+op2(l,m)][j+16:j+32]
                        amx_fma32(xoffset, yoffset, 1ull, zignore);
                    }
                }
            }
            float *Cij = C + i * size + j; // C[i][j]
            store_Z_to_C(Cij, size);
        }
    }
    AMX_STOP();
}

void _amx_sgemm(float *A, float *B, float *C, const uint64_t sizei, const uint64_t sizej, const uint64_t sizek)
{
    // limitation
    if (sizei == 0ull || sizek == 0ull || sizej == 0ull)
        return;
    if (sizei % 32 != 0 || sizek % 32 != 0 || sizej % 32 != 0)
        return;
    AMX_START();
    // 16 rows of A as a row tile
    for (uint64_t i = 0ull; i < sizei; i += 16ull)
    {
        float *Ai = A + i * sizek;
        // 2 * 16 columns of B as a column tile
        for (uint64_t j = 0ull; j < sizej; j += 32ull)
        {
            float *Bj = B + j;
            /*  for all 16 vertical elements of the row tile of A (16 rows), 
             *  and for all horizontal elements of the column tile of B (2 * 16 columns)
             *  
             *  like for all elements of a row of A,
             *  for all elements of a column of B,
             *  plus them one by one and then sum,
             *  but there are 16 columns/rows
             */
            for (uint64_t k = 0ull; k < sizek; k += 16ull)
            {
                float *Aik = Ai + k;        // A[i][k]
                float *Bkj = Bj + k * sizej; // B[k][j]
                load_A_to_Z(Aik, sizek);     // load A[i:i+16][k:k+16]
                for (uint64_t l = 0ull; l < 4; l++)
                {
                    // load B[k+l*4:k+l*4+4][j:j+2*16]
                    load_B_to_X(Bkj + (l << 2) * sizej, sizej);
                    // if l == 0
                    // transpose columns 0~8, A[i:i+16][k:k+8]
                    // else
                    // transpose columns 8~16, A[i:i+16][k+8:k+16]
                    if (l % 2 == 0)
                        // >> 1, 2nd bit of l. << 3 column offset. << 2 float size.
                        transpose_Z_to_Y(l >> 1 << 3 << 2);
                    for (uint64_t m = 0; m < 4; m++)
                    {
                        uint64_t xoffset = m << 1;
                        // ((l & 1ull) << 2) select 0~3 or 4~7 rows of register y
                        uint64_t yoffset = m + ((l & 1ull) << 2);
                        uint64_t zignore = k == 0ull && l == 0ull && m == 0ull ? 1ull : 0ull;
                        // fma32 A[i:i+16][k+op1(l,m)] and B[k+op2(l,m)][j:j+16]
                        amx_fma32(xoffset, yoffset, 0ull, zignore);
                        xoffset += 1ull;
                        // fma32 A[i:i+16][k+op1(l,m)] and B[k+op2(l,m)][j+16:j+32]
                        amx_fma32(xoffset, yoffset, 1ull, zignore);
                    }
                }
            }
            float *Cij = C + i * sizej + j; // C[i][j]
            store_Z_to_C(Cij, sizej);
        }
    }
    AMX_STOP();
}
