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
    long times[8][OMP_SGEMM_REPETITION][3];

    const uint64_t sizei = MATRIX_M;
    const uint64_t sizej = MATRIX_N;
    const uint64_t sizek = MATRIX_K;
    // limitation
    if (sizei == 0ull || sizek == 0ull || sizej == 0ull)
        return -1;
    if (sizei % 32 != 0 || sizek % 32 != 0 || sizej % 32 != 0)
        return -1;

    uint64_t diff;
    initMatrixAB();   
#pragma omp parallel
    {
#pragma omp single
        {
            gettimeofday(&start, NULL);
        }
#pragma omp for
        for (int _i = 0; _i < 8; _i++)
        {
            omp_thread_num[_i] = omp_get_thread_num();
            struct timeval omp_time;
            float *A = &MatrixA[_i][0][0];
            float *B = &MatrixB[_i][0][0];
            float *C = &MatrixC[_i][0][0];
            // A0[sizei / 32][sizek][2][16]
            float *A0 = (float *)aligned_alloc(128, sizei * sizek * sizeof(float));
            // B0[sizej / 32][sizek][2][16]
            float *B0 = (float *)aligned_alloc(128, sizek * sizej * sizeof(float));
            transformB(B, B0, sizek, sizej);
            gettimeofday(&omp_time, NULL);
            start_time[_i] = (omp_time.tv_sec - base_time) * 1000000 + omp_time.tv_usec;
            AMX_START();
            for (int _j = 0; _j < OMP_SGEMM_REPETITION; _j++)
            {
                gettimeofday(&omp_time, NULL);
                times[_i][_j][0] = (omp_time.tv_sec - base_time) * 1000000 + omp_time.tv_usec;
                transformA(A, A0, sizei, sizek);
                gettimeofday(&omp_time, NULL);
                times[_i][_j][1] = (omp_time.tv_sec - base_time) * 1000000 + omp_time.tv_usec;
                for (uint64_t i = 0ull; i < sizei; i += 32ull)
                {
                    float *A0i = A0 + i * sizek;
                    for (uint64_t j = 0ull; j < sizej; j += 32ull)
                    {
                        float *B0j = B0 + j * sizek;
                        for (uint64_t k = 0ull; k < sizek; k += 1ull)
                        {
                            amx_ldy((uint8_t *)(A0i + 32ull * k), 0ull, 1ull);
                            amx_ldx((uint8_t *)(B0j + 32ull * k), 0ull, 1ull);
                            uint64_t zignore = k == 0ull ? 1ull : 0ull;
                            amx_fma32(0ull, 0ull, 0ull, zignore);
                            amx_fma32(1ull, 0ull, 1ull, zignore);
                            amx_fma32(0ull, 1ull, 2ull, zignore);
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
                gettimeofday(&omp_time, NULL);
                times[_i][_j][2] = (omp_time.tv_sec - base_time) * 1000000 + omp_time.tv_usec;
            }
            AMX_STOP();
            gettimeofday(&omp_time, NULL);
            end_time[_i] = (omp_time.tv_sec - base_time) * 1000000 + omp_time.tv_usec;
            free(A0);
            free(B0);
        }
#pragma omp single
        {
            gettimeofday(&end, NULL);
            diff = 1000000ull * (end.tv_sec - start.tv_sec) + end.tv_usec - start.tv_usec;
            printf("Time: %llu\n", diff);
            long min_start = (start.tv_sec - base_time) * 1000000 + start.tv_usec;
            for (int i = 0; i < 8; i++)
            {
                printf("%d\t%d\t%5ld", i, omp_thread_num[i], start_time[i] - min_start);
                for (int j = 0; j < OMP_SGEMM_REPETITION; j++)
                    for (int k = 0; k < 3; k++)
                        if (j == 0 && k == 0)
                            printf("\t%5ld", times[i][j][k] - start_time[i]);
                        else if (k == 0)
                            printf("\t%5ld", times[i][j][k] - times[i][j - 1][2]);
                        else
                            printf("\t%5ld", times[i][j][k] - times[i][j][k - 1]);
                printf("%5ld\n", end_time[i] - times[i][OMP_SGEMM_REPETITION - 1][2]);
            }
            printf("\n");
        }
    }
    return 0;
}
