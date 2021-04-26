#pragma once

template <class T> class GEMM
{
  public:

    virtual ~GEMM(){}

    virtual void run() = 0;
    virtual void init_matrices() = 0;
};
