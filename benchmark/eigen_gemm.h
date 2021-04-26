#pragma once

#include <Eigen/Dense>
#include <type_traits>

#include "gemm.h"

template <class T> class EigenGEMM : public GEMM<T>
{
protected:
    using Mat = Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic>;
    size_t n;
    Mat a;
    Mat b;
    Mat c;

public:
    EigenGEMM(size_t n) : n(n), a(n, n), b(n, n), c(n, n) {}

    ~EigenGEMM() {}

    virtual void run() { c = a * b; }

    virtual void init_matrices()
    {
        a = Mat::Constant(n, n, static_cast<T>(1.0));
        b = Mat::Constant(n, n, static_cast<T>(1.0));
        c = Mat::Constant(n, n, static_cast<T>(0.0));
    }
};
