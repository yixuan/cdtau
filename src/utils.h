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
inline double sum_log1exp(const double* x, int n)
{
    double res = 0.0;
    for(int i = 0; i < n; i++)
    {
        res += log1exp(x[i]);
    }
    return res;
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
