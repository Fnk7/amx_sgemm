// compile options: -O3 -framework Accelerate

#include <sys/time.h>
#include <Accelerate/Accelerate.h>

#include "amx_sgemm.h"

#define MATRIX_M 1024ull
#define MATRIX_N 2048ull
#define MATRIX_K 1024ull

#define MAX_FLOAT_DIFF 0.00000f

__attribute__((aligned(0x80))) float MatrixA[MATRIX_M][MATRIX_K];
__attribute__((aligned(0x80))) float MatrixB[MATRIX_K][MATRIX_N];
__attribute__((aligned(0x80))) float MatrixC[MATRIX_M][MATRIX_N];

__attribute__((aligned(0x80))) float McblasC[MATRIX_M][MATRIX_N];

void initMatrixAB()
{
    srand(7);
    for (int i = 0; i < MATRIX_M; i++)
    {
        for (int j = 0; j < MATRIX_K; j++)
        {
            MatrixA[i][j] = (rand() % 20 + 1) / 100.0;
        }
    }
    for (int i = 0; i < MATRIX_K; i++)
    {
        for (int j = 0; j < MATRIX_N; j++)
        {
            MatrixB[i][j] = (rand() % 20 + 1) / 100.0;
        }
    }
}

int main()
{
    struct timeval start, end;
    uint64_t diff;
    initMatrixAB();
    // accelerate blas using amx
    float *A = &MatrixA[0][0];
    float *B = &MatrixB[0][0];
    float *C = &McblasC[0][0];
    for (int i = 0; i < 4; i++)
        cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
                    MATRIX_M, MATRIX_N, MATRIX_K, 1.0, A, MATRIX_K, B, MATRIX_N, 0.0, C, MATRIX_N);
    gettimeofday(&start, NULL);
    for (int i = 0; i < 8; i++)
        cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
                    MATRIX_M, MATRIX_N, MATRIX_K, 1.0, A, MATRIX_K, B, MATRIX_N, 0.0, C, MATRIX_N);
    gettimeofday(&end, NULL);
    diff = 1000000ull * (end.tv_sec - start.tv_sec) + end.tv_usec - start.tv_usec;
    printf("BLAS time: %llu\n", diff);
    // my code using amx
    for (int i = 0; i < 4; i++)
        _amx_sgemm((float *)MatrixA, (float *)MatrixB, (float *)MatrixC, MATRIX_M, MATRIX_N, MATRIX_K);
    gettimeofday(&start, NULL);
    for (int i = 0; i < 8; i++)
        _amx_sgemm((float *)MatrixA, (float *)MatrixB, (float *)MatrixC, MATRIX_M, MATRIX_N, MATRIX_K);
    gettimeofday(&end, NULL);
    diff = 1000000ull * (end.tv_sec - start.tv_sec) + end.tv_usec - start.tv_usec;
    printf("AMX  time: %llu\n", diff);
    // check
    uint64_t count = 0;
    for (uint64_t i = 0; i < MATRIX_M; i++)
        for (uint64_t j = 0; j < MATRIX_N; j++)
            count += fabsf(MatrixC[i][j] - McblasC[i][j]) <= MAX_FLOAT_DIFF ? 0 : 1;
    if (count)
        printf("Error count: %llu\n", count);
    else
        printf("Success!\n");
    return !(count == 0);
}
