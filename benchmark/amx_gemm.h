#pragma once

#include <type_traits>

#include "gemm.h"

#if USE_AMX == 1
#include "../src/amx_sgemm.1.h"
#elif USE_AMX == 2
#include "../src/amx_sgemm.2.h"
#elif USE_AMX == 3
#include "../src/amx_sgemm.3.h"
#else
#include "../src/amx_sgemm.h"
#endif

template <class T> class AMXGEMM : public GEMM<T>
{
protected:
    size_t n;
    T *a;
    T *b;
    T *c;

public:
    AMXGEMM(size_t n)
        : n(n), a(new (std::align_val_t{128}) T[n * n]),
          b(new (std::align_val_t{128}) T[n * n]),
          c(new (std::align_val_t{128}) T[n * n])
    // : n(n), a(new T[n * n]), b(new T[n * n]), c(new T[n * n])
    {
    }

    ~AMXGEMM()
    {
        delete[] a;
        delete[] b;
        delete[] c;
    }

    virtual void run()
    {
        if constexpr (std::is_same<T, float>::value) {
            amx_sgemm(a, b, c, n);
        }
    }

    virtual void init_matrices()
    {
        for (int i = 0; i < n; i++) {
            for (int j = 0; j < n; j++) {
                a[i * n + j] = static_cast<T>(1.0);
                b[i * n + j] = static_cast<T>(1.0);
                c[i * n + j] = static_cast<T>(0.0);
            }
        }
    }
};
