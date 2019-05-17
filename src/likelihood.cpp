#include <RcppEigen.h>
#include "mcmc.h"
#include "utils.h"

using Rcpp::NumericVector;
using Rcpp::NumericMatrix;
using Eigen::VectorXd;
using Eigen::MatrixXd;
typedef Eigen::Map<VectorXd> MapVec;
typedef Eigen::Map<MatrixXd> MapMat;

// res[n x 2^n]
inline MatrixXd permutation(const int n)
{
    const int pn = (1 << n);  // 2^n
    MatrixXd res(n, pn);
    double* r = res.data();
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
    MatrixXd vperm = permutation(m);
    VectorXd logzv = vperm.transpose() * b;
    MatrixXd vpermwc = w.transpose() * vperm;
    vpermwc.colwise() += c;
    apply_log1exp(vpermwc);
    logzv.noalias() += vpermwc.colwise().sum().transpose();
    const double logz = log_sum_exp(logzv);

    // https://arxiv.org/pdf/1510.02255.pdf, Eqn. (4)
    VectorXd loglik(N);
    VectorXd term1 = dat.transpose() * b;
    MatrixXd term2 = w.transpose() * dat;
    term2.colwise() += c;
    apply_log1exp(term2);
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

    RBMSampler sampler(w, b, c);
    VectorXd v0(m), v(m), vmean(m), h(n), logp(nsamp);
    double loglik = 0.0;

    for(int i = 0; i < N; i++)
    {
        for(int j = 0; j < nsamp; j++)
        {
            v0.noalias() = dat.col(random_int(N));
            sampler.sample_k(v0, v, h, nstep);
            vmean.noalias() = w * h + b;
            apply_sigmoid(vmean);
            logp[j] = loglik_bernoulli(vmean.data(), &dat(0, i), m);
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