// compile options: -O3 -framework Accelerate

#include <sys/time.h>
#include <Accelerate/Accelerate.h>

#include "amx_sgemm.h"

#define MATRIX_SIZE 1024ull

#define MAX_FLOAT_DIFF 0.00000f

__attribute__((aligned(0x80))) float MatrixA[MATRIX_SIZE][MATRIX_SIZE];
__attribute__((aligned(0x80))) float MatrixB[MATRIX_SIZE][MATRIX_SIZE];
__attribute__((aligned(0x80))) float MatrixC[MATRIX_SIZE][MATRIX_SIZE];

__attribute__((aligned(0x80))) float McblasC[MATRIX_SIZE][MATRIX_SIZE];

void initMatrixAB()
{
    srand(7);
    for (int i = 0; i < MATRIX_SIZE; i++)
    {
        for (int j = 0; j < MATRIX_SIZE; j++)
        {
            MatrixA[i][j] = (rand() % 20 + 1) / 100.0;
            MatrixB[i][j] = (rand() % 20 + 1) / 100.0;
        }
    }
}

int main()
{
    struct timeval start, end;
    uint64_t diff;
    initMatrixAB();
    // my code using amx
    gettimeofday(&start, NULL);
    amx_sgemm((float *)MatrixA, (float *)MatrixB, (float *)MatrixC, MATRIX_SIZE);
    gettimeofday(&end, NULL);
    diff = 1000000ull * (end.tv_sec - start.tv_sec) + end.tv_usec - start.tv_usec;
    printf("AMX  time: %llu\n", diff);
    // accelerate blas using amx
    float *A = &MatrixA[0][0];
    float *B = &MatrixB[0][0];
    float *C = &McblasC[0][0];
    gettimeofday(&start, NULL);
    cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
                MATRIX_SIZE, MATRIX_SIZE, MATRIX_SIZE, 1.0, A, MATRIX_SIZE, B, MATRIX_SIZE, 0.0, C, MATRIX_SIZE);
    gettimeofday(&end, NULL);
    diff = 1000000ull * (end.tv_sec - start.tv_sec) + end.tv_usec - start.tv_usec;
    printf("BLAS time: %llu\n", diff);
    // check
    uint64_t count = 0;
    for (uint64_t i = 0; i < MATRIX_SIZE; i++)
        for (uint64_t j = 0; j < MATRIX_SIZE; j++)
            count += (MatrixC[i][j] - McblasC[i][j]) <= MAX_FLOAT_DIFF  ? 0 : 1;
    if (count)
        printf("Error count: %llu\n", count);
    else
        printf("Success!\n");
    return 0;
}
