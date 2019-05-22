#ifndef CDTAU_UTILS_H
#define CDTAU_UTILS_H

#include <RcppEigen.h>

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
void apply_log1exp(Eigen::MatrixBase<Derived>& x)
{
    const int n = x.size();
    double* xptr = x.derived().data();
    for(int i = 0; i < n; i++)
    {
        xptr[i] = log1exp(xptr[i]);
    }
}

// res ~ Uniform(0, 1)
template <typename Derived>
void random_uniform(Eigen::MatrixBase<Derived>& res)
{
    const int n = res.size();
    double* res_ptr = res.derived().data();
    for(int i = 0; i < n; i++)
        res_ptr[i] = R::unif_rand();
}

// res ~ Bernoulli(prob)
template <typename Derived>
void random_bernoulli(const Eigen::MatrixBase<Derived>& prob, Eigen::MatrixBase<Derived>& res)
{
    const int n = prob.size();
    const double* prob_ptr = prob.derived().data();
    double* res_ptr = res.derived().data();
    for(int i = 0; i < n; i++)
        res_ptr[i] = double(R::unif_rand() <= prob_ptr[i]);
}

template <typename Derived>
void random_bernoulli_uvar(const Eigen::MatrixBase<Derived>& prob,
                           const Eigen::MatrixBase<Derived>& uvar,
                           Eigen::MatrixBase<Derived>& res)
{
    res.array() = (uvar.array() <= prob.array()).template cast<double>();
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
