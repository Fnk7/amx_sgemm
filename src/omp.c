// compile options: -O3 -framework Accelerate
#include <omp.h>
#include <sys/time.h>
#include <Accelerate/Accelerate.h>

#include "amx_sgemm.h"

#define MATRIX_M 1024ull
#define MATRIX_N 1024ull
#define MATRIX_K 1024ull

#define MAX_FLOAT_DIFF 0.00000f

__attribute__((aligned(0x80))) float MatrixA[8][MATRIX_M][MATRIX_K];
__attribute__((aligned(0x80))) float MatrixB[8][MATRIX_K][MATRIX_N];
__attribute__((aligned(0x80))) float MatrixC[8][MATRIX_M][MATRIX_N];

__attribute__((aligned(0x80))) float McblasC[8][MATRIX_M][MATRIX_N];

void initMatrixAB()
{
    srand(7);
    for (int num = 0; num < 8; num++)
    {
        for (int i = 0; i < MATRIX_M; i++)
        {
            for (int j = 0; j < MATRIX_K; j++)
            {
                MatrixA[num][i][j] = (rand() % 20 + 1) / 100.0;
            }
        }
    }
    for (int num = 0; num < 8; num++)
    {
        for (int i = 0; i < MATRIX_K; i++)
        {
            for (int j = 0; j < MATRIX_N; j++)
            {
                MatrixB[num][i][j] = (rand() % 20 + 1) / 100.0;
            }
        }
    }
}

int main()
{
    omp_set_num_threads(8);
    struct timeval start, end;
    uint64_t diff;
    initMatrixAB();
    // accelerate blas using amx
    for (int i = 0; i < 8; i++)
        cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
                    MATRIX_M, MATRIX_N, MATRIX_K, 1.0,
                    &MatrixA[i][0][0], MATRIX_K,
                    &MatrixB[i][0][0], MATRIX_N,
                    0.0,
                    &McblasC[i][0][0], MATRIX_N);
    gettimeofday(&start, NULL);
    #pragma omp parallel for
    for (int i = 0; i < 8; i++)
    {
        struct timeval omp_time;
        gettimeofday(&omp_time, NULL);
        printf(">>> start: %d %d %ld %d\n", omp_get_thread_num(), i, omp_time.tv_sec, omp_time.tv_usec);
        cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
                    MATRIX_M, MATRIX_N, MATRIX_K, 1.0,
                    &MatrixA[i][0][0], MATRIX_K,
                    &MatrixB[i][0][0], MATRIX_N,
                    0.0,
                    &McblasC[i][0][0], MATRIX_N);
        gettimeofday(&omp_time, NULL);
        printf(">>>   end: %d %d %ld %d\n", omp_get_thread_num(), i, omp_time.tv_sec, omp_time.tv_usec);
    }
    gettimeofday(&end, NULL);
    diff = 1000000ull * (end.tv_sec - start.tv_sec) + end.tv_usec - start.tv_usec;
    printf("BLAS time: %llu\n", diff);
    // my code using amx
    for (int i = 0; i < 8; i++)
        _amx_sgemm(&MatrixA[i][0][0],
                   &MatrixB[i][0][0],
                   &MatrixC[i][0][0],
                   MATRIX_M, MATRIX_N, MATRIX_K);
    gettimeofday(&start, NULL);
    #pragma omp parallel for
    for (int i = 0; i < 8; i++)
    {
        struct timeval omp_time;
        gettimeofday(&omp_time, NULL);
        printf(">>> start: %d %d %ld %d\n", omp_get_thread_num(), i, omp_time.tv_sec, omp_time.tv_usec);
        _amx_sgemm(&MatrixA[i][0][0],
                   &MatrixB[i][0][0],
                   &MatrixC[i][0][0],
                   MATRIX_M, MATRIX_N, MATRIX_K);
        gettimeofday(&omp_time, NULL);
        printf(">>>   end: %d %d %ld %d\n", omp_get_thread_num(), i, omp_time.tv_sec, omp_time.tv_usec);
    }
    gettimeofday(&end, NULL);
    diff = 1000000ull * (end.tv_sec - start.tv_sec) + end.tv_usec - start.tv_usec;
    printf("AMX  time: %llu\n", diff);
    // check
    uint64_t count = 0;
    for (int num = 0; num < 8; num++)
        for (uint64_t i = 0; i < MATRIX_M; i++)
            for (uint64_t j = 0; j < MATRIX_N; j++)
                count += fabsf(MatrixC[num][i][j] - McblasC[num][i][j]) <= MAX_FLOAT_DIFF ? 0 : 1;
    if (count)
        printf("Error count: %llu\n", count);
    else
        printf("Success!\n");
    return !(count == 0);
}
