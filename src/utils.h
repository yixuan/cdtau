#ifndef CDTAU_UTILS_H
#define CDTAU_UTILS_H

#include <RcppEigen.h>
#include <xsimd/xsimd.hpp>

// x => sigmoid(x)
template <typename Derived>
void apply_sigmoid(Eigen::MatrixBase<Derived>& x)
{
    x.array() = 1.0 / (1.0 + (-x).array().exp());
}

// log(exp(x1) + ... + exp(xn))
inline double log_sum_exp(const Eigen::VectorXd& x)
{
    const double xmax = x.maxCoeff();
    return xmax + std::log((x.array() - xmax).exp().sum());
}

// log(1 + exp(x))
// https://stackoverflow.com/a/51828104
inline double log1exp(const double& x)
{
    return std::log(1.0 + std::exp(-std::abs(x))) + std::max(x, 0.0);
}

// log(1 + exp(x1)) + ... + log(1 + exp(xn))
template <typename Derived>
void apply_log1exp_std(Eigen::MatrixBase<Derived>& x)
{
    const int n = x.size();
    double* xptr = x.derived().data();
    for(int i = 0; i < n; i++)
    {
        xptr[i] = log1exp(xptr[i]);
    }
}

template <typename Derived>
void apply_log1exp(Eigen::MatrixBase<Derived>& x)
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
        xp[i] = log1exp(xp[i]);
    }
}

// res ~ Bernoulli(prob)
inline void random_bernoulli(const Eigen::VectorXd& prob, Eigen::VectorXd& res)
{
    const int n = prob.size();
    for(int i = 0; i < n; i++)
        res[i] = double(R::unif_rand() <= prob[i]);
}

// x * log(p) + (1 - x) * log(1 - p)
inline double loglik_bernoulli(const double* prob, const double* x, int n)
{
    double res = 0.0;
    for(int i = 0; i < n; i++)
    {
        res += (x[i] > 0.5) ? (std::log(prob[i])) : (std::log(1.0 - prob[i]));
    }
    return res;
}
inline double loglik_bernoulli(const Eigen::VectorXd& prob, const Eigen::VectorXd& x)
{
    return loglik_bernoulli(prob.data(), x.data(), prob.size());
}

// Test x == y
inline bool all_equal(const Eigen::VectorXd& x, const Eigen::VectorXd& y, const double eps = 1e-12)
{
    const int n = x.size();
    for(int i = 0; i < n; i++)
    {
        if(std::abs(x[i] - y[i]) > eps)
            return false;
    }
    return true;
}

// Random shuffling
inline int random_int(int i) { return int(R::unif_rand() * i); }
inline void shuffle(Eigen::VectorXi v)
{
    std::random_shuffle(v.data(), v.data() + v.size(), random_int);
}

// x ~ N(mu, sigma^2)
inline void random_normal(double* x, int n, double mean, double sd)
{
    for(int i = 0; i < n; i++)
        x[i] = R::norm_rand() * sd + mean;
}


#endif  // CDTAU_UTILS_H