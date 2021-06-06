// compile options: -O3 -framework Accelerate
#include <omp.h>
#include <sys/time.h>

// #define OMP_ACCELERATE

#ifdef OMP_ACCELERATE
#include <Accelerate/Accelerate.h>
#elif defined(OMP_OPENBLAS)
#include <string.h>
#include <math.h>
#include <cblas.h>
#else
#error "OMP_ACCELERATE or OMP_OPENBLAS should be defined!"
#endif

#include "amx_sgemm.3.h"

#ifdef OMP_MATRIX_SIZE
#define MATRIX_M OMP_MATRIX_SIZE
#define MATRIX_N OMP_MATRIX_SIZE
#define MATRIX_K OMP_MATRIX_SIZE
#else
#define MATRIX_M 1024ull
#define MATRIX_N 1024ull
#define MATRIX_K 1024ull
#endif

#ifdef OMP_OPENBLAS
#define MAX_FLOAT_DIFF 0.00050f
#else
#define MAX_FLOAT_DIFF 0.00000f
#endif

#ifndef OMP_THREAD_NUM
#define OMP_THREAD_NUM 8
#elif OMP_THREAD_NUM > 8
#error "OMP_THREAD_NUM must lower or equal than 8!"
#endif

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
                MatrixA[num][i][j] = (rand() % 20 + 1 + num) / 100.0;
            }
        }
    }
    for (int num = 0; num < 8; num++)
    {
        for (int i = 0; i < MATRIX_K; i++)
        {
            for (int j = 0; j < MATRIX_N; j++)
            {
                MatrixB[num][i][j] = (rand() % 20 + 1 + num) / 100.0;
            }
        }
    }
}

int main()
{
    omp_set_num_threads(OMP_THREAD_NUM);
    struct timeval start, end;
    gettimeofday(&start, NULL);
    long base_time = start.tv_sec;

    uint64_t diff;
    initMatrixAB();

#pragma omp parallel
    {
        // library blas 
        // accelerate using amx (or amx and neon?)
        // openblas using neon
#pragma omp for
        for (int i = 0; i < 8; i++)
            cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
                        MATRIX_M, MATRIX_N, MATRIX_K, 1.0,
                        &MatrixA[i][0][0], MATRIX_K,
                        &MatrixB[i][0][0], MATRIX_N,
                        0.0,
                        &McblasC[i][0][0], MATRIX_N);
#pragma omp single
        {
#ifdef OMP_ACCELERATE
            printf("--- Accelerate ---\n");
#elif defined(OMP_OPENBLAS)
            printf("--- OpenBLAS ---\n");
#endif
            gettimeofday(&start, NULL);
        }
#pragma omp for
        for (int i = 0; i < 8; i++)
        {
            struct timeval omp_time;
            gettimeofday(&omp_time, NULL);
            printf(">>> start: %d %d %ld\n", omp_get_thread_num(), i, (omp_time.tv_sec - base_time) * 1000000 + omp_time.tv_usec);
            cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
                        MATRIX_M, MATRIX_N, MATRIX_K, 1.0,
                        &MatrixA[i][0][0], MATRIX_K,
                        &MatrixB[i][0][0], MATRIX_N,
                        0.0,
                        &McblasC[i][0][0], MATRIX_N);
            gettimeofday(&omp_time, NULL);
            printf(">>>   end: %d %d %ld\n", omp_get_thread_num(), i, (omp_time.tv_sec - base_time) * 1000000 + omp_time.tv_usec);
        }
#pragma omp single
        {
            gettimeofday(&end, NULL);
            diff = 1000000ull * (end.tv_sec - start.tv_sec) + end.tv_usec - start.tv_usec;
            printf("Time: %llu\n\n", diff);
        }
    }

#pragma omp parallel
    {
        // my code using amx
#pragma omp for
        for (int i = 0; i < 8; i++)
            _amx_sgemm(&MatrixA[i][0][0],
                       &MatrixB[i][0][0],
                       &MatrixC[i][0][0],
                       MATRIX_M, MATRIX_N, MATRIX_K);
#pragma omp single
        {
            printf("--- My_AMX_SGEMM ---\n");
            gettimeofday(&start, NULL);
        }
#pragma omp for
        for (int i = 0; i < 8; i++)
        {
            struct timeval omp_time;
            gettimeofday(&omp_time, NULL);
            printf(">>> start: %d %d %ld\n", omp_get_thread_num(), i, (omp_time.tv_sec - base_time) * 1000000 + omp_time.tv_usec);
            _amx_sgemm(&MatrixA[i][0][0],
                       &MatrixB[i][0][0],
                       &MatrixC[i][0][0],
                       MATRIX_M, MATRIX_N, MATRIX_K);
            gettimeofday(&omp_time, NULL);
            printf(">>>   end: %d %d %ld\n", omp_get_thread_num(), i, (omp_time.tv_sec - base_time) * 1000000 + omp_time.tv_usec);
        }
#pragma omp single
        {
            gettimeofday(&end, NULL);
            diff = 1000000ull * (end.tv_sec - start.tv_sec) + end.tv_usec - start.tv_usec;
            printf("Time: %llu\n\n", diff);
        }
    }

    // check
    int count = 0;
    for (int num = 0; num < 8; num++)
        for (int i = 0; i < MATRIX_M; i++)
            for (int j = 0; j < MATRIX_N; j++)
            {
                count += fabsf(MatrixC[num][i][j] - McblasC[num][i][j]) <= MAX_FLOAT_DIFF ? 0 : 1;
                if ((count <= 9) && (fabsf(MatrixC[num][i][j] - McblasC[num][i][j]) <= MAX_FLOAT_DIFF ? 0 : 1))
                {
                    printf("Error %d:  MAX_FLOAT_DIFF %f, MatrixC[%d][%d][%d] %f, McblasC[%d][%d][%d] %f.\n",
                           count, MAX_FLOAT_DIFF,
                           num, i, j,
                           MatrixC[num][i][j],
                           num, i, j,
                           McblasC[num][i][j]);
                }
            }

    if (count)
        printf("Error count: %d\n", count);
    else
        printf("Success!\n");
    return !(count == 0);
}
