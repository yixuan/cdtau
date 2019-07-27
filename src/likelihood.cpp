#include <RcppEigen.h>
#include "mcmc.h"
#include "utils.h"
#include "utils_simd.h"

using Rcpp::NumericVector;
using Rcpp::NumericMatrix;
using Eigen::VectorXd;
using Eigen::MatrixXd;
using Eigen::VectorXf;
using Eigen::MatrixXf;
typedef Eigen::Map<VectorXd> MapVec;
typedef Eigen::Map<MatrixXd> MapMat;
typedef Eigen::Map<VectorXf> MapVecf;
typedef Eigen::Map<MatrixXf> MapMatf;

// res[n x 2^n]
template <typename Scalar>
Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic> permutation(const int n)
{
    const int pn = (1 << n);  // 2^n
    Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic> res(n, pn);
    Scalar* r = res.data();
    for(int j = 0; j < pn; j++)
    {
        for(int i = 0; i < n; i++, r++)
        {
            *r = (j >> i) & 1;
        }
    }
    return res;
}

// w[m x n], b[m x 1], c[n x 1], dat[m x N]
// [[Rcpp::export]]
double loglik_rbm(MapMat w, MapVec b, MapVec c, MapMat dat)
{
    const int m = w.rows();
    const int n = w.cols();
    const int N = dat.cols();

    // Check dimension
    if(b.size() != m || c.size() != n || dat.rows() != m)
        Rcpp::stop("Dimensions do not match");

    // log(Z)
    // https://arxiv.org/pdf/1510.02255.pdf, Eqn. (5)
    MatrixXd vperm = permutation<double>(m);
    VectorXd logzv = vperm.transpose() * b;
    MatrixXd vpermwc = w.transpose() * vperm;
    vpermwc.colwise() += c;
    apply_log1exp_simd(vpermwc);
    logzv.noalias() += vpermwc.colwise().sum().transpose();
    const double logz = log_sum_exp(logzv);

    // https://arxiv.org/pdf/1510.02255.pdf, Eqn. (4)
    VectorXd loglik(N);
    VectorXd term1 = dat.transpose() * b;
    MatrixXd term2 = w.transpose() * dat;
    term2.colwise() += c;
    apply_log1exp_simd(term2);
    loglik.noalias() = term1 + term2.colwise().sum().transpose();

    return loglik.sum() - logz * N;
}

// [[Rcpp::export]]
double loglik_rbm_approx(MapMat w, MapVec b, MapVec c, MapMat dat,
                         int nsamp = 100, int nstep = 10)
{
    const int m = w.rows();
    const int n = w.cols();
    const int N = dat.cols();

    // Check dimension
    if(b.size() != m || c.size() != n || dat.rows() != m)
        Rcpp::stop("Dimensions do not match");

    RBMSampler<double> sampler(w, b, c);
    VectorXd v0(m), logp(nsamp);
    MatrixXd vmean(m, nsamp), h(n, nsamp);
    double loglik = 0.0;

    for(int i = 0; i < N; i++)
    {
        v0.noalias() = dat.col(i);
        sampler.sample_k_mc(v0, vmean, h, nstep, nsamp);
        vmean.noalias() = w * h;
        vmean.colwise() += b;
        apply_sigmoid(vmean);

        for(int j = 0; j < nsamp; j++)
        {
            logp[j] = loglik_bernoulli(&vmean(0, j), &dat(0, i), m);
        }
        loglik += log_sum_exp(logp);
    }

    return loglik - N * std::log(double(nsamp));
}

/*

 set.seed(123)
 m = 10
 n = 10
 b = rnorm(m, sd = 0.1)
 c = rnorm(n, sd = 0.1)
 w = matrix(rnorm(m * n, sd = 1.0), m, n)

 N = 100
 dat = matrix(0, m, N)
 v0 = rbinom(m, 1, 0.5)
 for(i in 1:N)
 {
     dat[, i] = rbm_sample_k(w, b, c, v0, k = 100)$v
 }

 loglik_rbm(w, b, c, dat)
 loglik_rbm_approx(w, b, c, dat, nsamp = 100, nstep = 10)
 loglik_rbm_approx(w, b, c, dat, nsamp = 100, nstep = 100)

 */



// w[m x n], b[m x 1], c[n x 1], dat[m x N]
float loglik_rbm(MapMatf w, MapVecf b, MapVecf c, MapMatf dat)
{
    const int m = w.rows();
    const int n = w.cols();
    const int N = dat.cols();

    // Check dimension
    if(b.size() != m || c.size() != n || dat.rows() != m)
        Rcpp::stop("Dimensions do not match");

    // log(Z)
    // https://arxiv.org/pdf/1510.02255.pdf, Eqn. (5)
    MatrixXf vperm = permutation<float>(m);
    VectorXf logzv = vperm.transpose() * b;
    MatrixXf vpermwc = w.transpose() * vperm;
    vpermwc.colwise() += c;
    apply_log1exp_simd(vpermwc);
    logzv.noalias() += vpermwc.colwise().sum().transpose();
    const float logz = log_sum_exp(logzv);

    // https://arxiv.org/pdf/1510.02255.pdf, Eqn. (4)
    VectorXf loglik(N);
    VectorXf term1 = dat.transpose() * b;
    MatrixXf term2 = w.transpose() * dat;
    term2.colwise() += c;
    apply_log1exp_simd(term2);
    loglik.noalias() = term1 + term2.colwise().sum().transpose();

    return loglik.sum() - logz * N;
}

float loglik_rbm_approx(MapMatf w, MapVecf b, MapVecf c, MapMatf dat,
                        int nsamp = 100, int nstep = 10)
{
    const int m = w.rows();
    const int n = w.cols();
    const int N = dat.cols();

    // Check dimension
    if(b.size() != m || c.size() != n || dat.rows() != m)
        Rcpp::stop("Dimensions do not match");

    RBMSampler<float> sampler(w, b, c);
    VectorXf v0(m), logp(nsamp);
    MatrixXf vmean(m, nsamp), h(n, nsamp);
    float loglik = 0.0;

    for(int i = 0; i < N; i++)
    {
        v0.noalias() = dat.col(i);
        sampler.sample_k_mc(v0, vmean, h, nstep, nsamp);
        vmean.noalias() = w * h;
        vmean.colwise() += b;
        apply_sigmoid(vmean);

        for(int j = 0; j < nsamp; j++)
        {
            logp[j] = loglik_bernoulli(&vmean(0, j), &dat(0, i), m);
        }
        loglik += log_sum_exp(logp);
    }

    return loglik - N * std::log(float(nsamp));
}
