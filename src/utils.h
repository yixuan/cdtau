#ifndef CDTAU_UTILS_H
#define CDTAU_UTILS_H

#include <RcppEigen.h>
#include <random>
#include "blas.h"

// The common operation W * h + b in RBM
// Since h is a binary vector, theoretically we can make it faster
// However benchmarking results show that BLAS may still provide better performance
template <typename Scalar>
void rbm_op_v(
    const Eigen::Ref< const Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic> >& w,
    const Eigen::Matrix<Scalar, Eigen::Dynamic, 1>& h,
    const Eigen::Ref< const Eigen::Matrix<Scalar, Eigen::Dynamic, 1> >& b,
    Eigen::Matrix<Scalar, Eigen::Dynamic, 1>& v
)
{
#ifdef EIGEN_USE_BLAS

    const int m = w.rows();
    const int n = w.cols();
    blas_copy<Scalar>(m, b.data(), v.data());
    blas_gemv<Scalar>('N', m, n, Scalar(1), w.data(), h.data(), Scalar(1), v.data());

#else

    v.noalias() = w * h + b;

#endif

    /* v.noalias() = b;
    const Scalar* hptr = h.data();
    const int n = h.size();
    for(int i = 0; i < n; i++)
    {
        if(hptr[i] > Scalar(0.5))
        {
            v.noalias() += w.col(i);
        }
    } */
}

// The common operation W' * v + c in RBM
template <typename Scalar>
void rbm_op_h(
    const Eigen::Ref< const Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic> >& w,
    const Eigen::Matrix<Scalar, Eigen::Dynamic, 1>& v,
    const Eigen::Ref< const Eigen::Matrix<Scalar, Eigen::Dynamic, 1> >& c,
    Eigen::Matrix<Scalar, Eigen::Dynamic, 1>& h
)
{
#ifdef EIGEN_USE_BLAS

    const int m = w.rows();
    const int n = w.cols();
    blas_copy<Scalar>(n, c.data(), h.data());
    blas_gemv<Scalar>('T', m, n, Scalar(1), w.data(), v.data(), Scalar(1), h.data());

#else

    h.noalias() = w.transpose() * v + c;

#endif

    /* h.noalias() = c;
    const int m = w.rows();
    const int n = w.cols();

    const Scalar* vptr = v.data();
    std::vector<int> nz;
    for(int i = 0; i < m; i++)
    {
        if(vptr[i] > Scalar(0.5))
            nz.push_back(i);
    }
    const int nnz = nz.size();
    const int* nzptr = &nz[0];

    const Scalar* colptr = w.data();
    Scalar* hptr = h.data();
    for(int i = 0; i < n; i++, colptr += m)
    {
        Scalar res = Scalar(0);
        for(int j = 0; j < nnz; j++)
            res += colptr[nzptr[j]];
        hptr[i] += res;
    } */
}

// w = v1 * h1' + v2 * h2'
template <typename Scalar>
void rbm_op_rank2(
    const Eigen::Matrix<Scalar, Eigen::Dynamic, 1>& v1,
    const Eigen::Matrix<Scalar, Eigen::Dynamic, 1>& h1,
    const Eigen::Matrix<Scalar, Eigen::Dynamic, 1>& v2,
    const Eigen::Matrix<Scalar, Eigen::Dynamic, 1>& h2,
    Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic>& w
)
{
#ifdef EIGEN_USE_BLAS

    const int m = w.rows();
    const int n = w.cols();
    w.setZero();
    blas_ger<Scalar>(m, n, Scalar(1), v1.data(), h1.data(), w.data());
    blas_ger<Scalar>(m, n, Scalar(1), v2.data(), h2.data(), w.data());

#else

    w.noalias() = v1 * h1.transpose() + v2 * h2.transpose();

#endif
}

// w += v1 * h1' + v2 * h2' - v3 * h3' - v4 * h4'
template <typename Scalar>
void rbm_op_rank4(
    const Eigen::Matrix<Scalar, Eigen::Dynamic, 1>& v1,
    const Eigen::Matrix<Scalar, Eigen::Dynamic, 1>& h1,
    const Eigen::Matrix<Scalar, Eigen::Dynamic, 1>& v2,
    const Eigen::Matrix<Scalar, Eigen::Dynamic, 1>& h2,
    const Eigen::Matrix<Scalar, Eigen::Dynamic, 1>& v3,
    const Eigen::Matrix<Scalar, Eigen::Dynamic, 1>& h3,
    const Eigen::Matrix<Scalar, Eigen::Dynamic, 1>& v4,
    const Eigen::Matrix<Scalar, Eigen::Dynamic, 1>& h4,
    Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic>& w
)
{
#ifdef EIGEN_USE_BLAS

    const int m = w.rows();
    const int n = w.cols();
    blas_ger<Scalar>(m, n, Scalar(1), v1.data(), h1.data(), w.data());
    blas_ger<Scalar>(m, n, Scalar(1), v2.data(), h2.data(), w.data());
    blas_ger<Scalar>(m, n, Scalar(-1), v3.data(), h3.data(), w.data());
    blas_ger<Scalar>(m, n, Scalar(-1), v4.data(), h4.data(), w.data());

#else

    w.noalias() += v1 * h1.transpose() + v2 * h2.transpose() - v3 * h3.transpose() - v4 * h4.transpose();

#endif
}



// Test x == y
template <typename Scalar>
bool all_equal(const Eigen::Matrix<Scalar, Eigen::Dynamic, 1>& x,
               const Eigen::Matrix<Scalar, Eigen::Dynamic, 1>& y,
               const double eps = 1e-12)
{
    const int n = x.size();
    for(int i = 0; i < n; i++)
    {
        if(std::abs(x[i] - y[i]) > eps)
            return false;
    }
    return true;
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

// x => log(1 + exp(x))
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

// x => sigmoid(x)
// x is clipped to [-10, 10]
template <typename Derived>
void apply_sigmoid(Eigen::MatrixBase<Derived>& x)
{
    typedef typename Derived::Scalar Scalar;

    x.array() = x.array().max(Scalar(-10)).min(Scalar(10));
    x.array() = Scalar(1) / (Scalar(1) + (-x).array().exp());
}

// x * log(p) + (1 - x) * log(1 - p)
template <typename Scalar>
Scalar loglik_bernoulli(const Scalar* prob, const Scalar* x, int n)
{
    Scalar res = Scalar(0);
    for(int i = 0; i < n; i++)
    {
        res += (x[i] > Scalar(0.5)) ? (std::log(prob[i])) : (std::log(Scalar(1) - prob[i]));
    }
    return res;
}
template <typename Scalar>
Scalar loglik_bernoulli(const Eigen::Matrix<Scalar, Eigen::Dynamic, 1>& prob, const Eigen::Matrix<Scalar, Eigen::Dynamic, 1>& x)
{
    return loglik_bernoulli(prob.data(), x.data(), prob.size());
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
template <typename Derived, typename RNGType>
void random_uniform(Eigen::MatrixBase<Derived>& res, RNGType& gen)
{
    typedef typename Derived::Scalar Scalar;

    const int n = res.size();
    Scalar* res_ptr = res.derived().data();
    const Scalar denom = Scalar(gen.max()) + Scalar(1);
    for(int i = 0; i < n; i++)
        res_ptr[i] = gen() / denom;
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
template <typename Derived, typename RNGType>
void random_bernoulli(const Eigen::MatrixBase<Derived>& prob, Eigen::MatrixBase<Derived>& res, RNGType& gen)
{
    typedef typename Derived::Scalar Scalar;

    const int n = prob.size();
    const Scalar* prob_ptr = prob.derived().data();
    Scalar* res_ptr = res.derived().data();
    const Scalar denom = Scalar(gen.max()) + Scalar(1);
    for(int i = 0; i < n; i++)
        res_ptr[i] = Scalar(gen() / denom <= prob_ptr[i]);
}

// res ~ Bernoulli(prob), given prob and uniform random variates
// If antithetic == true, use 1-U as the uniform random variate
template <typename Derived>
void random_bernoulli_uvar(const Eigen::MatrixBase<Derived>& prob,
                           const Eigen::MatrixBase<Derived>& uvar,
                           Eigen::MatrixBase<Derived>& res,
                           bool antithetic = false)
{
    typedef typename Derived::Scalar Scalar;

    if(antithetic)
        res.array() = (uvar.array() >= (Scalar(1) - prob.array())).template cast<Scalar>();
    else
        res.array() = (uvar.array() <= prob.array()).template cast<Scalar>();
}

// Apply U to the first part of res, and (1 - U) to the second part
// prob [r x 2N], uvar [r x N], res [r x 2N]
template <typename Scalar>
void random_bernoulli_uvar_antithetic(
    const Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic>& prob,
    const Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic>& uvar,
    Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic>& res)
{
    const int n1 = uvar.cols();
    res.leftCols(n1).array() = (uvar.array() <= prob.leftCols(n1).array()).template cast<Scalar>();
    res.rightCols(n1).array() = (uvar.array() >= (Scalar(1) - prob.rightCols(n1).array())).template cast<Scalar>();
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
