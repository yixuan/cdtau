#ifndef CDTAU_BLAS_H
#define CDTAU_BLAS_H

#ifdef EIGEN_USE_BLAS
#include <RcppEigen.h>
#include <stdexcept>

// ============================= ?copy =============================
// y = x
template <typename Scalar>
inline void blas_copy(int n, const Scalar* x, Scalar* y)
{
    throw std::invalid_argument("Scalar type not supported");
}

template <>
inline void blas_copy<float>(int n, const float* x, float* y)
{
    int inc = 1;
    // Eigen's ?copy definition is a bit problematic -- it uses a non-const
    // pointer type for x, so we need to cast it
    BLASFUNC(scopy)(&n, const_cast<float*>(x), &inc, y, &inc);
}

template <>
inline void blas_copy<double>(int n, const double* x, double* y)
{
    int inc = 1;
    BLASFUNC(dcopy)(&n, const_cast<double*>(x), &inc, y, &inc);
}



// ============================= ?gemv =============================
// trans = 'N': y = alpha * A * x + beta * y
// trans = 'T': y = alpha * A' * x + beta * y
template <typename Scalar>
inline void blas_gemv(const char trans, const int m, const int n,
                      const Scalar alpha, const Scalar* a,
                      const Scalar* x, const Scalar beta, Scalar* y)
{
    throw std::invalid_argument("Scalar type not supported");
}

template <>
inline void blas_gemv<float>(const char trans, const int m, const int n,
                             const float alpha, const float* a,
                             const float* x, const float beta, float* y)
{
    const int inc = 1;
    BLASFUNC(sgemv)(&trans, &m, &n, &alpha, a, &m, x, &inc, &beta, y, &inc);
}

template <>
inline void blas_gemv<double>(const char trans, const int m, const int n,
                              const double alpha, const double* a,
                              const double* x, const double beta, double* y)
{
    const int inc = 1;
    BLASFUNC(dgemv)(&trans, &m, &n, &alpha, a, &m, x, &inc, &beta, y, &inc);
}



// ============================= ?ger =============================
// A = alpha * x * y' + A
template <typename Scalar>
inline void blas_ger(int m, int n, Scalar alpha,
                     const Scalar* x, const Scalar* y, Scalar* a)
{
    throw std::invalid_argument("Scalar type not supported");
}

template <>
inline void blas_ger<float>(int m, int n, float alpha,
                            const float* x, const float* y, float* a)
{
    int inc = 1;
    BLASFUNC(sger)(&m, &n, &alpha, const_cast<float*>(x), &inc, const_cast<float*>(y), &inc, a, &m);
}

template <>
inline void blas_ger<double>(int m, int n, double alpha,
                             const double* x, const double* y, double* a)
{
    int inc = 1;
    BLASFUNC(dger)(&m, &n, &alpha, const_cast<double*>(x), &inc, const_cast<double*>(y), &inc, a, &m);
}



#endif  // EIGEN_USE_BLAS

#endif  // CDTAU_BLAS_H
