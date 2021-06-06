#!/usr/bin/python
import os

def getCompileCmd(accelerate=True, thread_num=8, matrix_size=1024):
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
    cmd += ' -DOMP_THREAD_NUM={}'.format(thread_num)
    cmd += ' -DOMP_MATRIX_SIZE={}'.format(matrix_size)
    cmd += ' omp.c'
    return cmd

if __name__ == '__main__':
    err = 0
    for thread_num in (1, 2, 4, 8):
        print(getCompileCmd(accelerate=False, thread_num=thread_num))
        err += abs(os.system(getCompileCmd(accelerate=False, thread_num=thread_num)))
        err += abs(os.system("./a.out"))
    if err != 0:
        exit(-1)

