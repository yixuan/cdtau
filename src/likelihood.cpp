#include "likelihood.h"

using Eigen::VectorXd;
using Eigen::MatrixXd;
typedef Eigen::Map<VectorXd> MapVec;
typedef Eigen::Map<MatrixXd> MapMat;

//' Compute Log-likelihood Value for RBM
//'
//' The two functions compute the exact and approximate log-likelihood values for RBM, respectively.
//'
//' @param w       Weight parameter of the RBM, of size \code{[m x n]}.
//' @param b       Bias parameter for the visible units, of size \code{[m x 1]}.
//' @param c       Bias parameter for the hidden units, of size \code{[n x 1]}.
//' @param dat     The observed data, of size \code{[m x N]}.
//' @param nsamp   Size of the Monte Carlo sample for approximation.
//' @param nstep   Number of steps in the Gibbs sampler.
//' @param vr      Whether to use variance reduction technique.
//' @param nthread Number of threads for parallel computing, if OpenMP is supported.
//'
//' @examples
//' set.seed(123)
//' m = 10
//' n = 10
//' b = rnorm(m, sd = 0.1)
//' c = rnorm(n, sd = 0.1)
//' w = matrix(rnorm(m * n, sd = 1.0), m, n)
//'
//' N = 100
//' dat = matrix(0, m, N)
//' v0 = rbinom(m, 1, 0.5)
//' for(i in 1:N)
//' {
//'     dat[, i] = rbm_sample_k(w, b, c, v0, k = 100)$v
//' }
//'
//' loglik_rbm(w, b, c, dat)
//' loglik_rbm_approx(w, b, c, dat, nsamp = 100, nstep = 10)
//' loglik_rbm_approx(w, b, c, dat, nsamp = 100, nstep = 100)
//'
//' @rdname loglik_rbm
//'
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

//' @rdname loglik_rbm
// [[Rcpp::export]]
double loglik_rbm_approx(MapMat w, MapVec b, MapVec c, MapMat dat,
                         int nsamp = 100, int nstep = 10, bool vr = true, int nthread = 1)
{
    const int m = w.rows();
    const int n = w.cols();
    const int N = dat.cols();

    // Check dimension
    if(b.size() != m || c.size() != n || dat.rows() != m)
        Rcpp::stop("Dimensions do not match");

    return loglik_rbm_approx(m, n, N, w.data(), b.data(), c.data(), dat.data(),
                             nsamp, nstep, vr, nthread);
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
 loglik_rbm_approx(w, b, c, dat, nsamp = 100, nstep = 10, vr = FALSE)
 loglik_rbm_approx(w, b, c, dat, nsamp = 100, nstep = 10, vr = TRUE)
 loglik_rbm_approx(w, b, c, dat, nsamp = 100, nstep = 100)

 exact = loglik_rbm(w, b, c, dat)
 est1 = c()
 est2 = c()
 for(i in 1:100)
 {
     print(i)
     est1 = c(est1, loglik_rbm_approx(w, b, c, dat, nsamp = 20, nstep = 20, vr = FALSE))
     est2 = c(est2, loglik_rbm_approx(w, b, c, dat, nsamp = 20, nstep = 20, vr = TRUE))
 }
 mean(est1) - exact
 mean(est2) - exact
 var(est1)
 var(est2)

*/
