#ifndef CDTAU_UTILS_SIMD_H
#define CDTAU_UTILS_SIMD_H

#include <RcppEigen.h>
#include <xsimd/xsimd.hpp>

// x => log(1 + exp(x))
template <typename Derived>
void apply_log1exp_simd(Eigen::MatrixBase<Derived>& x)
{
    /* Eigen::ArrayXXd max0 = x.array().max(0.0);
     x.array() = 1.0 + (-x.array().abs()).exp();
     x.array() = x.array().log();
     x.array() += max0; */

    typedef typename Derived::Scalar Scalar;
    typedef xsimd::batch<Scalar, xsimd::simd_type<Scalar>::size> vec;

    Scalar* xp = x.derived().data();
    const int n = x.size();
    const int simd_size = xsimd::simd_type<Scalar>::size;
    const int vec_size = n - n % simd_size;

    vec zero = xsimd::set_simd(Scalar(0));
    vec one = xsimd::set_simd(Scalar(1));

    for(int i = 0; i < vec_size; i += simd_size)
    {
        vec xi = xsimd::load_aligned(xp + i);
        xi = xsimd::log(one + xsimd::exp(-xsimd::abs(xi))) + xsimd::max(zero, xi);
        xi.store_aligned(xp + i);
    }
    for(int i = vec_size; i < n; i++)
    {
        xp[i] = std::log(Scalar(1) + std::exp(-std::abs(xp[i]))) + std::max(xp[i], Scalar(0));
    }
}

// x => sigmoid(x)
// x is clipped to [-10, 10]
template <typename Derived>
void apply_sigmoid_simd(Eigen::MatrixBase<Derived>& x)
{
    /* x.array() = x.array().max(Scalar(-10)).min(Scalar(10));
     x.array() = Scalar(1) / (Scalar(1) + (-x).array().exp()); */

    typedef typename Derived::Scalar Scalar;
    typedef xsimd::batch<Scalar, xsimd::simd_type<Scalar>::size> vec;

    Scalar* xp = x.derived().data();
    const int n = x.size();
    const int simd_size = xsimd::simd_type<Scalar>::size;
    const int vec_size = n - n % simd_size;

    vec one = xsimd::set_simd(Scalar(1));
    vec ten = xsimd::set_simd(Scalar(10));
    vec mten = xsimd::set_simd(Scalar(-10));

    for(int i = 0; i < vec_size; i += simd_size)
    {
        vec xi = xsimd::load_aligned(xp + i);
        xi = one / (one + xsimd::exp(-xsimd::clip(xi, mten, ten)));
        xi.store_aligned(xp + i);
    }
    for(int i = vec_size; i < n; i++)
    {
        const Scalar xi = std::max(-Scalar(10), std::min(Scalar(10), xp[i]));
        xp[i] = Scalar(1) / (Scalar(1) + std::exp(-xi));
    }
}

// x * log(p) + (1 - x) * log(1 - p)
template <typename Scalar>
Scalar loglik_bernoulli_simd(const Scalar* prob, const Scalar* x, int n)
{
    typedef xsimd::batch<Scalar, xsimd::simd_type<Scalar>::size> vec;

    const int simd_size = xsimd::simd_type<Scalar>::size;
    const int vec_size = n - n % simd_size;

    vec one = xsimd::set_simd(Scalar(1));
    vec half = xsimd::set_simd(Scalar(0.5));

    Scalar res = 0;
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
        res += (x[i] > Scalar(0.5)) ? (std::log(prob[i])) : (std::log(Scalar(1) - prob[i]));
    }

    return res;
}
template <typename Scalar>
Scalar loglik_bernoulli_simd(const Eigen::Matrix<Scalar, Eigen::Dynamic, 1>& prob, const Eigen::Matrix<Scalar, Eigen::Dynamic, 1>& x)
{
    return loglik_bernoulli_simd(prob.data(), x.data(), prob.size());
}

// res ~ Bernoulli(prob), given prob and uniform random variates
// If antithetic == true, use 1-U as the uniform random variate
template <typename Derived>
void random_bernoulli_uvar_simd(const Eigen::MatrixBase<Derived>& prob,
                                const Eigen::MatrixBase<Derived>& uvar,
                                Eigen::MatrixBase<Derived>& res,
                                bool antithetic = false)
{
    typedef typename Derived::Scalar Scalar;
    typedef xsimd::batch<Scalar, xsimd::simd_type<Scalar>::size> vec;

    const Scalar* pp = prob.derived().data();
    const Scalar* up = uvar.derived().data();
    Scalar* rp = res.derived().data();
    const int n = res.size();
    const int simd_size = xsimd::simd_type<Scalar>::size;
    const int vec_size = n - n % simd_size;

    vec one = xsimd::set_simd(Scalar(1));
    vec zero = xsimd::set_simd(Scalar(0));

    if(antithetic)
    {
        for(int i = 0; i < vec_size; i += simd_size)
        {
            vec probi = xsimd::load_aligned(pp + i);
            vec uvari = xsimd::load_aligned(up + i);
            vec resi = xsimd::select(uvari >= one - probi, one, zero);
            resi.store_aligned(rp + i);
        }
        for(int i = vec_size; i < n; i++)
        {
            rp[i] = Scalar(up[i] >= Scalar(1) - pp[i]);
        }
    } else {
        for(int i = 0; i < vec_size; i += simd_size)
        {
            vec probi = xsimd::load_aligned(pp + i);
            vec uvari = xsimd::load_aligned(up + i);
            vec resi = xsimd::select(uvari <= probi, one, zero);
            resi.store_aligned(rp + i);
        }
        for(int i = vec_size; i < n; i++)
        {
            rp[i] = Scalar(up[i] <= pp[i]);
        }
    }
}

#endif  // CDTAU_UTILS_SIMD_H
