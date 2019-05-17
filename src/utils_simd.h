#ifndef CDTAU_UTILS_SIMD_H
#define CDTAU_UTILS_SIMD_H

#include <RcppEigen.h>
#include <xsimd/xsimd.hpp>

// log(1 + exp(x1)) + ... + log(1 + exp(xn))
template <typename Derived>
void apply_log1exp_simd(Eigen::MatrixBase<Derived>& x)
{
    /* Eigen::ArrayXXd max0 = x.array().max(0.0);
     x.array() = 1.0 + (-x.array().abs()).exp();
     x.array() = x.array().log();
     x.array() += max0; */

    typedef xsimd::batch<double, xsimd::simd_type<double>::size> vec;

    double* xp = x.derived().data();
    const int n = x.size();
    const int simd_size = xsimd::simd_type<double>::size;
    const int vec_size = n - n % simd_size;

    vec zero;
    zero ^= zero;

    for(int i = 0; i < vec_size; i += simd_size)
    {
        vec xi = xsimd::load_aligned(xp + i);
        xi = xsimd::log(1.0 + xsimd::exp(-xsimd::abs(xi))) + xsimd::max(zero, xi);
        xi.store_aligned(xp + i);
    }
    for(int i = vec_size; i < n; i++)
    {
        xp[i] = std::log(1.0 + std::exp(-std::abs(xp[i]))) + std::max(xp[i], 0.0);
    }
}


#endif  // CDTAU_UTILS_SIMD_H
