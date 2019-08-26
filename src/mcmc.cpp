#include "mcmc.h"

using Rcpp::NumericMatrix;
using Rcpp::NumericVector;
using Rcpp::List;
using Eigen::VectorXd;
using Eigen::MatrixXd;
typedef Eigen::Map<VectorXd> MapVec;
typedef Eigen::Map<MatrixXd> MapMat;

// [[Rcpp::export]]
List rbm_sample_k(MapMat w, MapVec b, MapVec c, VectorXd v0, int k = 10)
{
    RBMSampler<double> sampler(w, b, c);
    VectorXd v, h;

    sampler.sample_k(v0, v, h, k);
    return List::create(
        Rcpp::Named("v") = v,
        Rcpp::Named("h") = h
    );
}

// [[Rcpp::export]]
List rbm_sample_tau(
    MapMat w, MapVec b, MapVec c, VectorXd v0,
    int min_steps = 10, int max_steps = 100, bool verbose = false
)
{
    RBMSampler<double> sampler(w, b, c);
    MatrixXd vhist, vchist;
    std::mt19937 gen(int(R::unif_rand() * 10000));

    sampler.sample(gen, v0, vhist, vchist, min_steps, max_steps, verbose);
    return List::create(
        Rcpp::Named("v") = vhist,
        Rcpp::Named("vc") = vchist
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
