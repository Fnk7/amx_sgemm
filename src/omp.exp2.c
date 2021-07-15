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
#ifdef OMP_MATRIX_SIZE_M
#define MATRIX_M OMP_MATRIX_SIZE_M
#else
#define MATRIX_M 512
#endif
#ifdef OMP_MATRIX_SIZE_N
#define MATRIX_N OMP_MATRIX_SIZE_N
#else
#define MATRIX_N 32
#endif
#ifdef OMP_MATRIX_SIZE_K
#define MATRIX_K OMP_MATRIX_SIZE_K
#else
#define MATRIX_K 512
#endif
#endif

#ifdef OMP_OPENBLAS
#define MAX_FLOAT_DIFF 0.00500f
#else
#define MAX_FLOAT_DIFF 0.00000f
#endif

#ifndef OMP_THREAD_NUM
#define OMP_THREAD_NUM 8
#elif OMP_THREAD_NUM > 8
#error "OMP_THREAD_NUM must lower or equal than 8!"
#endif

#ifndef OMP_SGEMM_REPETITION
#define OMP_SGEMM_REPETITION 1
#endif

#ifndef OMP_NUM_OF_A
#define OMP_NUM_OF_A 8
#endif

__attribute__((aligned(0x80))) float MatrixA[OMP_NUM_OF_A][MATRIX_M][MATRIX_K];
__attribute__((aligned(0x80))) float MatrixB[MATRIX_K][MATRIX_N];
__attribute__((aligned(0x80))) float MatrixC[OMP_NUM_OF_A][MATRIX_M][MATRIX_N];

__attribute__((aligned(0x80))) float McblasC[OMP_NUM_OF_A][MATRIX_M][MATRIX_N];

#ifdef OMP_NO_PREHOT
__attribute__((aligned(0x80))) char nohot1[32 * 1024 * 1024]; // 32 MB
__attribute__((aligned(0x80))) char nohot2[32 * 1024 * 1024]; // 32 MB
#endif

void initMatrixAB()
{
    srand(7);
    for (int num = 0; num < OMP_NUM_OF_A; num++)
    {
        for (int i = 0; i < MATRIX_M; i++)
        {
            for (int j = 0; j < MATRIX_K; j++)
            {
                MatrixA[num][i][j] = (rand() % 20 + 1 + num) / 100.0;
            }
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
    printf("Running with settings: Thread num %d, Num of Matrix A %d, M %d, K %d, N %d\n",
           OMP_THREAD_NUM, OMP_NUM_OF_A,
           MATRIX_M, MATRIX_K, MATRIX_N);

    int a_per_thread = OMP_NUM_OF_A / OMP_THREAD_NUM;
    if (a_per_thread == 0)
        return -1;

    initMatrixAB();

    omp_set_num_threads(OMP_THREAD_NUM);

    struct timeval start, end;
    uint64_t diff;
    for (int test_time = 0; test_time < 32; test_time++)
    {
#ifdef OMP_NO_PREHOT
        memcpy(nohot1, nohot2, sizeof(nohot1));
#endif
        // library blas
        // accelerate using amx (or amx and neon?)
        // openblas using neon
#pragma omp parallel
        {
            int tid = omp_get_thread_num();
#ifndef OMP_NO_PREHOT
            cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
                        MATRIX_M * a_per_thread, MATRIX_N, MATRIX_K, 1.0,
                        &MatrixA[tid * a_per_thread][0][0], MATRIX_K,
                        &MatrixB[0][0], MATRIX_N,
                        0.0,
                        &McblasC[tid * a_per_thread][0][0], MATRIX_N);
#endif
#pragma omp single
            {
#ifdef OMP_ACCELERATE
                printf("Accelerate-----Time: ");
#elif defined(OMP_OPENBLAS)
                printf("OpenBLAS-------Time: ");
#endif
                gettimeofday(&start, NULL);
            }
            cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
                        MATRIX_M * a_per_thread, MATRIX_N, MATRIX_K, 1.0,
                        &MatrixA[tid * a_per_thread][0][0], MATRIX_K,
                        &MatrixB[0][0], MATRIX_N,
                        0.0,
                        &McblasC[tid * a_per_thread][0][0], MATRIX_N);
#pragma omp single
            {
                gettimeofday(&end, NULL);
                diff = 1000000ull * (end.tv_sec - start.tv_sec) + end.tv_usec - start.tv_usec;
                printf(" %llu\n", diff);
            }
        }

        // my code using amx
#pragma omp parallel
        {
            int tid = omp_get_thread_num();
#ifndef OMP_NO_PREHOT
            _amx_sgemm(&MatrixA[tid * a_per_thread][0][0],
                       &MatrixB[0][0],
                       &MatrixC[tid * a_per_thread][0][0],
                       MATRIX_M * a_per_thread, MATRIX_N, MATRIX_K);
#endif
#pragma omp single
            {
                printf("My_AMX_SGEMM---Time: ");
                gettimeofday(&start, NULL);
            }
            _amx_sgemm(&MatrixA[tid * a_per_thread][0][0],
                       &MatrixB[0][0],
                       &MatrixC[tid * a_per_thread][0][0],
                       MATRIX_M * a_per_thread, MATRIX_N, MATRIX_K);
#pragma omp single
            {
                gettimeofday(&end, NULL);
                diff = 1000000ull * (end.tv_sec - start.tv_sec) + end.tv_usec - start.tv_usec;
                printf(" %llu\n", diff);
            }
        }

        // check
        int count = 0;
        for (int num = 0; num < OMP_NUM_OF_A; num++)
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
        {
            printf("Error count: %d\n", count);
            return count == 0;
        }
    }
    return 0;
}
