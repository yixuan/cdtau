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

    vec zero = xsimd::set_simd(0.0);
    vec one = xsimd::set_simd(1.0);

    for(int i = 0; i < vec_size; i += simd_size)
    {
        vec xi = xsimd::load_aligned(xp + i);
        xi = xsimd::log(one + xsimd::exp(-xsimd::abs(xi))) + xsimd::max(zero, xi);
        xi.store_aligned(xp + i);
    }
    for(int i = vec_size; i < n; i++)
    {
        xp[i] = std::log(1.0 + std::exp(-std::abs(xp[i]))) + std::max(xp[i], 0.0);
    }
}

// x * log(p) + (1 - x) * log(1 - p)
inline double loglik_bernoulli_simd(const double* prob, const double* x, int n)
{
    typedef xsimd::batch<double, xsimd::simd_type<double>::size> vec;

    const int simd_size = xsimd::simd_type<double>::size;
    const int vec_size = n - n % simd_size;

    vec one = xsimd::set_simd(1.0);
    vec half = xsimd::set_simd(0.5);

    double res = 0.0;
    for(int i = 0; i < vec_size; i += simd_size)
    {
        vec probi = xsimd::load_aligned(prob + i);
        vec one_m_probi = one - probi;
        vec xi = xsimd::load_aligned(x + i);
        vec r = xsimd::log(xsimd::select(xi > half, probi, one_m_probi));
        res += xsimd::hadd(r);
    }
    for(int i = vec_size; i < n; i++)
    {
        res += (x[i] > 0.5) ? (std::log(prob[i])) : (std::log(1.0 - prob[i]));
    }

    return res;
}
inline double loglik_bernoulli_simd(const Eigen::VectorXd& prob, const Eigen::VectorXd& x)
{
    return loglik_bernoulli_simd(prob.data(), x.data(), prob.size());
}


#endif  // CDTAU_UTILS_SIMD_H
