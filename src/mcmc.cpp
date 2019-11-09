#include "mcmc.h"
#include "rng.h"

using Rcpp::NumericMatrix;
using Rcpp::NumericVector;
using Rcpp::List;
using Eigen::VectorXd;
using Eigen::MatrixXd;
typedef Eigen::Map<VectorXd> MapVec;
typedef Eigen::Map<MatrixXd> MapMat;

//' Sample from an RBM Model
//'
//' \code{rbm_sample_k()} uses Gibbs sampling with a fixed-size chain.
//' \code{rbm_sample_tau()} uses the unbiased Gibbs sampling with a chain path.
//'
//' @param w         Weight parameter of the RBM, of size \code{[m x n]}.
//' @param b         Bias parameter for the visible units, of size \code{[m x 1]}.
//' @param c         Bias parameter for the hidden units, of size \code{[n x 1]}.
//' @param v0        The initial value for Gibbs sampling, of size \code{[m x 1]}.
//' @param k         Number of steps in the Gibbs sampler.
//' @param min_steps Minimum number of steps in the Gibbs sampler.
//' @param max_steps Maximum number of steps in the Gibbs sampler.
//' @param verbose   Whether to print algorithmic information.
//'
//' @examples
//' set.seed(123)
//' m = 10
//' n = 10
//' b = rnorm(m, sd = 0.1)
//' c = rnorm(n, sd = 0.1)
//' w = matrix(rnorm(m * n, sd = 1.0), m, n)
//' v0 = rbinom(m, 1, 0.5)
//'
//' rbm_sample_k(w, b, c, v0, k = 100)
//' rbm_sample_tau(w, b, c, v0, min_steps = 1, max_steps = 100, verbose = TRUE)
//'
//' @rdname rbm_sample
//'
// [[Rcpp::export]]
List rbm_sample_k(MapMat w, MapVec b, MapVec c, NumericVector v0, int k = 10)
{
    RBMSampler<double> sampler(w, b, c);
    VectorXd v, h;
    RNGEngine gen(int(R::unif_rand() * 10000));

    sampler.sample_k(gen, Rcpp::as<MapVec>(v0), v, h, k);
    return List::create(
        Rcpp::Named("v") = v,
        Rcpp::Named("h") = h
    );
}

//' @rdname rbm_sample
// [[Rcpp::export]]
List rbm_sample_tau(
    MapMat w, MapVec b, MapVec c, NumericVector v0,
    int min_steps = 10, int max_steps = 100, bool verbose = false
)
{
    RBMSampler<double> sampler(w, b, c);
    MatrixXd vhist, vchist, hhist, hchist;
    RNGEngine gen(int(R::unif_rand() * 10000));

    sampler.sample(gen, false, Rcpp::as<MapVec>(v0), vhist, vchist, hhist, hchist,
                   min_steps, max_steps, verbose);
    return List::create(
        Rcpp::Named("v") = vhist,
        Rcpp::Named("vc") = vchist,
        Rcpp::Named("h") = hhist,
        Rcpp::Named("hc") = hchist
    );
}

/*

 set.seed(123)
 m = 10
 n = 10
 b = rnorm(m, sd = 0.1)
 c = rnorm(n, sd = 0.1)
 w = matrix(rnorm(m * n, sd = 1.0), m, n)
 v0 = rbinom(m, 1, 0.5)

 rbm_sample_k(w, b, c, v0, k = 100)
 rbm_sample_tau(w, b, c, v0, min_steps = 1, max_steps = 100, verbose = TRUE)

*/
