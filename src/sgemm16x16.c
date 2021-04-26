/*
 *  shows how to use fma32 for 16*16 matmul
 */

#include <Accelerate/Accelerate.h>

#include <stdio.h>
#include <string.h>
#include <stdlib.h>

#include "amx.h"

__attribute__((aligned(0x80))) float MatrixA[16][16];
__attribute__((aligned(0x80))) float MatrixB[16][16];
__attribute__((aligned(0x80))) float MatrixC[16][16];

/*
    get the result of matmul(A, B)
    A is transposed
*/

void acc_gemm()
{
    float *A = &MatrixA[0][0];
    float *B = &MatrixB[0][0];
    float *C = &MatrixC[0][0];
    cblas_sgemm(CblasRowMajor, CblasTrans, CblasNoTrans,
                16, 16, 16, 1.0, A, 16, B, 16, 0.0, C, 16);
}

void amx_gemm()
{
    AMX_START();
    for (uint64_t i = 0; i < 8; i++)
    {
        AMX_LDX((i << 56) | (uint64_t)&MatrixB[i][0]);
        AMX_LDY((i << 56) | (uint64_t)&MatrixA[i][0]);
    }
    AMX_FMA32(1l << 27);
    for (uint64_t i = 1; i < 8; i++)
    {
        AMX_FMA32((i << 6 << 10) | (i << 6));
    }
    for (uint64_t i = 0; i < 8; i++)
    {
        AMX_LDX((i << 56) | (uint64_t)&MatrixB[i + 8][0]);
        AMX_LDY((i << 56) | (uint64_t)&MatrixA[i + 8][0]);
    }
    for (uint64_t i = 0; i < 8; i++)
    {
        AMX_FMA32((i << 6 << 10) | (i << 6));
    }
    for (uint64_t i = 0; i < 64; i += 4)
    {
        AMX_STZ((i << 56) | (uint64_t)&MatrixC[i >> 2][0]);
    }
    AMX_STOP();
}

void for_gemm()
{
    for (int i = 0; i < 16; i++)
    {
        for (int j = 0; j < 16; j++)
        {
            MatrixC[i][j] = 0;
            /*
                A is transposed
                MatrixA[:][i] is row i of A
            */
            for (int k = 0; k < 16; k++)
            {
                MatrixC[i][j] += MatrixA[k][i] * MatrixB[k][j];
            }
        }
    }
}

void initXY()
{
    srand(7);
    for (int i = 0; i < 16; i++)
    {
        for (int j = 0; j < 16; j++)
        {
            MatrixA[i][j] = (rand() % 20 + 1) / 100.0;
            MatrixB[i][j] = (rand() % 20 + 1) / 100.0;
        }
    }
}

void initZ()
{
    for (int i = 0; i < 16; i++)
    {
        for (int j = 0; j < 16; j++)
        {
            MatrixC[i][j] = 0.f;
        }
    }
}

void printZ()
{
    for (int i = 0; i < 16; i++)
    {
        for (int j = 0; j < 16; j++)
        {
            printf("%.3f ", MatrixC[i][j]);
        }
        printf("\n");
    }
}

int main()
{
    initXY();
    printf("acc_gemm\n");
    initZ();
    acc_gemm();
    printZ();
    printf("amx_gemm\n");
    initZ();
    amx_gemm();
    printZ();
    printf("for_gemm\n");
    initZ();
    for_gemm();
    printZ();
}
