#!/usr/bin/python
import os
import time

result_dir="./omp_result2"

def getCompileCmd(accelerate=True, num_of_a=2, thread_num=2, matrix_size_mk=1024, matrix_size_n=32):
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
    cmd += ' -DOMP_NUM_OF_A={}'.format(num_of_a)
    cmd += ' -DOMP_THREAD_NUM={}'.format(thread_num)
    cmd += ' -DOMP_MATRIX_SIZE_M={}'.format(matrix_size_mk)
    cmd += ' -DOMP_MATRIX_SIZE_K={}'.format(matrix_size_mk)
    cmd += ' -DOMP_MATRIX_SIZE_N={}'.format(matrix_size_n)
    cmd += ' omp.exp2.c'
    return cmd


def run(cmd, output):
    err = 0
    err += abs(os.system(cmd))
    err += abs(os.system("./a.out > {}/{}".format(result_dir, output)))
    if err != 0:
        print("--- Failed! ---")
        exit(-1)


if __name__ == '__main__':
    os.system("echo OMP_NUM_THREADS=$OMP_NUM_THREADS > {}/env_omp_threads.txt".format(result_dir))
    with open("{}/env_omp_threads.txt".format(result_dir)) as env_omp_threads:
        line = env_omp_threads.readlines()[0]
        if line != "OMP_NUM_THREADS=1\n":
            print("--- Failed! ---")
            print("$OMP_NUM_THREADS should be 1")
            print("export OMP_NUM_THREADS=1")
            exit(-1)
    
    for accelerate in [True, False]:
        for num_of_a in [2, 4, 8]:
            time_start = time.time()
            for thread_num in [num for num in [1, 2, 4, 8] if num <= num_of_a]:
                for matrix_size_mk in [256, 512, 1024]:
                    for matrix_size_n in [32, 64]:
                        cmd = getCompileCmd(accelerate, num_of_a, thread_num, matrix_size_mk, matrix_size_n)
                        output = "num_of_a_{0}-thread_num_{1}-matrix_size_{2:0>4d}_{3:0>4d}_{4:0>4d}-{5}.txt".format(num_of_a, thread_num, matrix_size_mk, matrix_size_mk, matrix_size_n, "Accelerate" if accelerate else "OpenBLAS")
                        run(cmd, output)
            time_end = time.time()
            print("Finished num_of_a {}, time cost {}".format(num_of_a, time_end - time_start))
    pass

