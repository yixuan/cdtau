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
    std::mt19937 gen(0);

    sampler.sample(gen, v0, vhist, vchist, min_steps, max_steps, verbose);
    return List::create(
        Rcpp::Named("v") = vhist,
        Rcpp::Named("vc") = vchist
    );
}
