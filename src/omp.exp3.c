// compile options: -O3 -framework Accelerate
#include <omp.h>
#include <sys/time.h>

#include <Accelerate/Accelerate.h>

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
#define MATRIX_N 512
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

    int omp_thread_num[8];
    long start_time[8];
    long end_time[8];

    uint64_t diff;
    initMatrixAB();

#pragma omp parallel
    {
        // library blas
        // accelerate using amx (or amx and neon?)
        // openblas using neon
// #pragma omp for
//         for (int i = 0; i < 8; i++)
//             cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
//                         MATRIX_M, MATRIX_N, MATRIX_K, 1.0,
//                         &MatrixA[i][0][0], MATRIX_K,
//                         &MatrixB[i][0][0], MATRIX_N,
//                         0.0,
//                         &McblasC[i][0][0], MATRIX_N);
#pragma omp single
        {
            printf("Accelerate ");
            gettimeofday(&start, NULL);
        }
#pragma omp for
        for (int i = 0; i < 8; i++)
        {
            omp_thread_num[i] = omp_get_thread_num();
            struct timeval omp_time;
            gettimeofday(&omp_time, NULL);
            start_time[i] = (omp_time.tv_sec - base_time) * 1000000 + omp_time.tv_usec;
            cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
                        MATRIX_M, MATRIX_N, MATRIX_K, 1.0,
                        &MatrixA[i][0][0], MATRIX_K,
                        &MatrixB[i][0][0], MATRIX_N,
                        0.0,
                        &McblasC[i][0][0], MATRIX_N);
            gettimeofday(&omp_time, NULL);
            end_time[i] = (omp_time.tv_sec - base_time) * 1000000 + omp_time.tv_usec;
        }
#pragma omp single
        {
            gettimeofday(&end, NULL);
            diff = 1000000ull * (end.tv_sec - start.tv_sec) + end.tv_usec - start.tv_usec;
            printf(" time: %llu\n", diff);
            long min_start = start_time[0];
            for (int i = 1; i < 8; i++)
                min_start = min_start <= start_time[i] ? min_start : start_time[i];
            for (int i = 0; i < 8; i++)
                printf("%d\t%d\t%5ld\t%5ld\n", i, omp_thread_num[i], start_time[i] - min_start, end_time[i] - min_start);
            printf("\n");
        }
    }

#pragma omp parallel
    {
        // my code using amx
// #pragma omp for
//         for (int i = 0; i < 8; i++)
//             _amx_sgemm(&MatrixA[i][0][0],
//                        &MatrixB[i][0][0],
//                        &MatrixC[i][0][0],
//                        MATRIX_M, MATRIX_N, MATRIX_K);
#pragma omp single
        {
            printf("AMX_SGEMM ");
            gettimeofday(&start, NULL);
        }
#pragma omp for
        for (int i = 0; i < 8; i++)
        {
            omp_thread_num[i] = omp_get_thread_num();
            struct timeval omp_time;
            gettimeofday(&omp_time, NULL);
            start_time[i] = (omp_time.tv_sec - base_time) * 1000000 + omp_time.tv_usec;
            for (int j = 0; j < OMP_SGEMM_REPETITION; j++)
                _amx_sgemm(&MatrixA[i][0][0],
                           &MatrixB[i][0][0],
                           &MatrixC[i][0][0],
                           MATRIX_M, MATRIX_N, MATRIX_K);
            gettimeofday(&omp_time, NULL);
            end_time[i] = (omp_time.tv_sec - base_time) * 1000000 + omp_time.tv_usec;
        }
#pragma omp single
        {
            gettimeofday(&end, NULL);
            diff = 1000000ull * (end.tv_sec - start.tv_sec) + end.tv_usec - start.tv_usec;
            printf(" time: %llu\n", diff);
            long min_start = start_time[0];
            for (int i = 1; i < 8; i++)
                min_start = min_start <= start_time[i] ? min_start : start_time[i];
            for (int i = 0; i < 8; i++)
                printf("%d\t%d\t%5ld\t%5ld\n", i, omp_thread_num[i], start_time[i] - min_start, end_time[i] - min_start);
            printf("\n");
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
    return !(count == 0);
}
