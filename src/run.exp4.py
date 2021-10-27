#!/usr/bin/python
import os
import time

def getCompileCmd(accelerate=True, repetition=1, thread_num=8, matrix_size_m=512, matrix_size_k=512, matrix_size_n=512):
    cmd = 'clang'
    cmd += ' -O3'
    cmd += ' -Xpreprocessor -fopenmp'
    if accelerate:
        cmd += ' -framework Accelerate'
    cmd += ' -I"$(brew --prefix libomp)/include" -L"$(brew --prefix libomp)/lib"'
    cmd += ' -lomp'
    if not accelerate:
        cmd += ' -I"$(brew --prefix openblas)/include" -L"$(brew --prefix openblas)/lib"'
        cmd += ' -lopenblas'
    if accelerate:
        cmd += ' -DOMP_ACCELERATE'
    else:
        cmd += ' -DOMP_OPENBLAS'
    cmd += ' -DOMP_SGEMM_REPETITION={}'.format(repetition)
    cmd += ' -DOMP_THREAD_NUM={}'.format(thread_num)
    cmd += ' -DOMP_MATRIX_SIZE_M={}'.format(matrix_size_m)
    cmd += ' -DOMP_MATRIX_SIZE_K={}'.format(matrix_size_k)
    cmd += ' -DOMP_MATRIX_SIZE_N={}'.format(matrix_size_n)
    cmd += ' omp.exp4.c'
    return cmd


def run(cmd, output):
    err = 0
    err += abs(os.system(cmd))
    time_start = time.time()
    err += abs(os.system("./a.out > omp_result/exp4/{}".format(output)))
    time_end = time.time()
    min_time = time_end - time_start
    for _ in range(31):
        time_start = time.time()
        err += abs(os.system("./a.out > omp_result/tmp.txt"))
        time_end = time.time()
        if min_time > time_end - time_start:
            min_time = time_end - time_start
            err += abs(os.system("cp omp_result/tmp.txt omp_result/exp4/{}".format(output)))
    if err != 0:
        print("--- Failed! ---")
        exit(-1)


if __name__ == '__main__':
    os.system("echo OMP_NUM_THREADS=$OMP_NUM_THREADS > omp_result/env_omp_threads.txt")
    with open("omp_result/env_omp_threads.txt") as env_omp_threads:
        line = env_omp_threads.readlines()[0]
        if line != "OMP_NUM_THREADS=1\n":
            print("--- Failed! ---")
            print("$OMP_NUM_THREADS should be 1")
            print("export OMP_NUM_THREADS=1")
            exit(-1)
    
    for thread_num in [1, 2, 4, 8]:
        time_start = time.time()
        repetition = 3
        matrix_size_m = 512
        matrix_size_n = 1024
        for matrix_size_k in [128, 256, 512, 1024, 2048]:
            cmd = getCompileCmd(True, repetition, thread_num, matrix_size_m, matrix_size_k, matrix_size_n)
            output = "{2:0>4d}_{3:0>4d}_{4:0>4d}-{1}.txt".format(repetition, thread_num, matrix_size_m, matrix_size_k, matrix_size_n)
            run(cmd, output)
        time_end = time.time()
        print("Finished omp_thread_num {}, time cost {}".format(thread_num, time_end - time_start))
    pass
