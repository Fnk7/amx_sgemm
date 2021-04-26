#include <algorithm>
#include <chrono>
#include <iomanip>
#include <iostream>
#include <numeric>
#include <vector>

#if defined(USE_OPENBLAS)
#include "openblas_gemm.h"
#elif defined(USE_ACCELERATE)
#include "accelerate_gemm.h"
#elif defined(USE_EIGEN)
#include "eigen_gemm.h"
#elif defined(USE_METAL)
#include "metal_gemm.h"
#endif

#if defined(DTYPE_FLOAT)
using DTYPE = float;
#elif defined(DTYPE_DOUBLE)
using DTYPE = double;
#endif

template <class T> void benchmark(GEMM<T> *gemm, size_t n, int n_trials)
{
    gemm->init_matrices();

    std::vector<double> timings;
    for (int i = 0; i < n_trials; i++) {

        const std::chrono::steady_clock::time_point start =
            std::chrono::steady_clock::now();

        gemm->run();

        const std::chrono::steady_clock::time_point end =
            std::chrono::steady_clock::now();

        if (i < 1) continue;

        double runtime =
            std::chrono::duration_cast<std::chrono::duration<double>>(end -
                                                                      start)
                .count();
        timings.push_back(runtime);
    }

    double min_rt = *std::min_element(timings.begin(), timings.end());
    double max_rt = *std::max_element(timings.begin(), timings.end());
    double mean_rt =
        std::accumulate(timings.begin(), timings.end(), 0.0) / timings.size();
    double gflops = (2 * n * n * n / min_rt / 1e9);

    std::cout << std::fixed << std::left << std::setw(8) << n << std::left
              << std::setw(12) << std::setprecision(3) << gflops << std::left
              << std::setw(12) << std::setprecision(5) << mean_rt << std::left
              << std::setw(12) << std::setprecision(5) << min_rt << std::left
              << std::setw(12) << std::setprecision(5) << max_rt << std::endl;
}

int main(int argc, char *argv[])
{
    std::cout << std::left << std::setw(8) << "N" << std::left << std::setw(12)
              << "GFLOP/s" << std::left << std::setw(12) << "mean(rt)"
              << std::left << std::setw(12) << "min(rt)" << std::left
              << std::setw(12) << "max(rt)" << std::endl;

    for (size_t i = 6; i < 14; i++) {
        size_t n = 1 << i;

#if defined(USE_OPENBLAS)
        GEMM<DTYPE> *gemm = new OpenBLASGEMM<DTYPE>(n);
#elif defined(USE_ACCELERATE)
        GEMM<DTYPE> *gemm = new AccelerateGEMM<DTYPE>(n);
#elif defined(USE_EIGEN)
        GEMM<DTYPE> *gemm = new EigenGEMM<DTYPE>(n);
#elif defined(USE_METAL)
        GEMM<DTYPE> *gemm = new MetalGEMM<DTYPE>(n);
#endif

        benchmark(gemm, n, 20);

        delete gemm;
    }
}
