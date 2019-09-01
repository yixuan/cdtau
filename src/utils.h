#ifndef CDTAU_UTILS_H
#define CDTAU_UTILS_H

#include <RcppEigen.h>
#include <random>

// x => sigmoid(x)
template <typename Derived>
void apply_sigmoid(Eigen::MatrixBase<Derived>& x)
{
    x.array() = 1.0 / (1.0 + (-x).array().exp());
}

// log(exp(x1) + ... + exp(xn))
template <typename Scalar>
Scalar log_sum_exp(const Eigen::Matrix<Scalar, Eigen::Dynamic, 1>& x)
{
    const Scalar xmax = x.maxCoeff();
    return xmax + std::log((x.array() - xmax).exp().sum());
}

// log(1 + exp(x))
// https://stackoverflow.com/a/51828104
template <typename Scalar>
Scalar log1exp(const Scalar& x)
{
    return std::log(Scalar(1) + std::exp(-std::abs(x))) + std::max(x, Scalar(0));
}

// log(1 + exp(x1)), ..., log(1 + exp(xn))
template <typename Derived>
void apply_log1exp(Eigen::MatrixBase<Derived>& x)
{
    typedef typename Derived::Scalar Scalar;

    const int n = x.size();
    Scalar* xptr = x.derived().data();
    for(int i = 0; i < n; i++)
    {
        xptr[i] = log1exp(xptr[i]);
    }
}

// res ~ Uniform(0, 1), RNG from R
template <typename Derived>
void random_uniform(Eigen::MatrixBase<Derived>& res)
{
    typedef typename Derived::Scalar Scalar;

    const int n = res.size();
    Scalar* res_ptr = res.derived().data();
    for(int i = 0; i < n; i++)
        res_ptr[i] = R::unif_rand();
}
// RNG from C++
template <typename Derived>
void random_uniform(Eigen::MatrixBase<Derived>& res, std::mt19937& gen)
{
    typedef typename Derived::Scalar Scalar;

    const int n = res.size();
    Scalar* res_ptr = res.derived().data();
    std::uniform_real_distribution<Scalar> distr(0.0, 1.0);
    for(int i = 0; i < n; i++)
        res_ptr[i] = distr(gen);
}

// res ~ Bernoulli(prob), RNG from R
template <typename Derived>
void random_bernoulli(const Eigen::MatrixBase<Derived>& prob, Eigen::MatrixBase<Derived>& res)
{
    typedef typename Derived::Scalar Scalar;

    const int n = prob.size();
    const Scalar* prob_ptr = prob.derived().data();
    Scalar* res_ptr = res.derived().data();
    for(int i = 0; i < n; i++)
        res_ptr[i] = Scalar(R::unif_rand() <= prob_ptr[i]);
}
// RNG from C++
template <typename Derived>
void random_bernoulli(const Eigen::MatrixBase<Derived>& prob, Eigen::MatrixBase<Derived>& res, std::mt19937& gen)
{
    typedef typename Derived::Scalar Scalar;

    const int n = prob.size();
    const Scalar* prob_ptr = prob.derived().data();
    Scalar* res_ptr = res.derived().data();
    std::uniform_real_distribution<Scalar> distr(0.0, 1.0);
    for(int i = 0; i < n; i++)
        res_ptr[i] = Scalar(distr(gen) <= prob_ptr[i]);
}
// res ~ Bernoulli(prob), given prob and uniform random variates
template <typename Derived>
void random_bernoulli_uvar(const Eigen::MatrixBase<Derived>& prob,
                           const Eigen::MatrixBase<Derived>& uvar,
                           Eigen::MatrixBase<Derived>& res)
{
    typedef typename Derived::Scalar Scalar;

    res.array() = (uvar.array() <= prob.array()).template cast<Scalar>();
}

// x * log(p) + (1 - x) * log(1 - p)
template <typename Scalar>
Scalar loglik_bernoulli(const Scalar* prob, const Scalar* x, int n)
{
    Scalar res = 0.0;
    for(int i = 0; i < n; i++)
    {
        res += (x[i] > 0.5) ? (std::log(prob[i])) : (std::log(1.0 - prob[i]));
    }
    return res;
}
template <typename Scalar>
Scalar loglik_bernoulli(const Eigen::Matrix<Scalar, Eigen::Dynamic, 1>& prob, const Eigen::Matrix<Scalar, Eigen::Dynamic, 1>& x)
{
    return loglik_bernoulli(prob.data(), x.data(), prob.size());
}

// Test x == y
template <typename Scalar>
bool all_equal(const Eigen::Matrix<Scalar, Eigen::Dynamic, 1>& x, const Eigen::Matrix<Scalar, Eigen::Dynamic, 1>& y, const double eps = 1e-12)
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
template <typename Scalar>
void random_normal(Scalar* x, int n, Scalar mean, Scalar sd)
{
    for(int i = 0; i < n; i++)
        x[i] = R::norm_rand() * sd + mean;
}


#endif  // CDTAU_UTILS_H
