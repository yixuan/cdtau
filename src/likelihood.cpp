#include <RcppEigen.h>
#include "likelihood.h"

using Eigen::VectorXd;
using Eigen::MatrixXd;
typedef Eigen::Map<VectorXd> MapVec;
typedef Eigen::Map<MatrixXd> MapMat;

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

    return loglik_rbm_exact(m, n, N, w.data(), b.data(), c.data(), dat.data());
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

    return loglik_rbm_approx(m, n, N, w.data(), b.data(), c.data(), dat.data(), nsamp, nstep);
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
