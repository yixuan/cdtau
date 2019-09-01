#include <RcppEigen.h>
#include "rbm.h"

using Rcpp::NumericMatrix;
using Rcpp::NumericVector;
using Rcpp::List;

using Eigen::VectorXi;
using Eigen::VectorXd;
using Eigen::MatrixXd;
using Eigen::VectorXf;
using Eigen::MatrixXf;
typedef Eigen::Map<VectorXd> MapVec;
typedef Eigen::Map<MatrixXd> MapMat;

// dat [m x N]
// [[Rcpp::export]]
List rbm_cdk_warm(
    int vis_dim, int hid_dim, MapMat dat,
    MapVec b0, MapVec c0, MapMat w0,
    int batch_size = 10, double lr = 0.1, int niter = 100,
    int ngibbs = 10, int nchain = 1, bool persistent = false,
    bool eval_loglik = false, bool exact_loglik = true,
    int neval_mb = 10, int neval_dat = 1000, int neval_mcmc = 100, int neval_step = 10,
    int verbose = 0
)
{
    typedef float Scalar;
    typedef Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic> Matrix;

    const int m = vis_dim;
    const int n = hid_dim;
    const int N = dat.cols();

    if(dat.rows() != m)
        Rcpp::stop("dimensions do not match");

    // Indices of observations
    VectorXi ind = VectorXi::LinSpaced(N, 0, N - 1);

    // log-likelihood value
    std::vector<Scalar> loglik;

    // RBM model
    RBM<Scalar> rbm(m, n, nchain, b0, c0, w0);

    // Initial values for Gibbs sampler
    if(persistent)
        rbm.init_v0(dat);

    for(int k = 0; k < niter; k++)
    {
        if(verbose > 0)
            Rcpp::Rcout << "\n===== Iter " << k + 1 << " =====" << std::endl;

        // Shuffle observations
        shuffle(ind);

        // Update on mini-batches
        for(int i = 0; i < N; i += batch_size)
        {
            const int batch_id = i / batch_size + 1;
            if(verbose > 1)
                Rcpp::Rcout << "==> Mini-batch " << batch_id << std::endl;

            // Indices for this mini-batch: i, i+1, ..., i+bs-1
            const int bs = std::min(i + batch_size, N) - i;

            // Mini-batch data
            Matrix mb_dat(m, bs);
            for(int j = 0; j < bs; j++)
            {
                mb_dat.col(j).noalias() = dat.col(ind[i + j]).cast<Scalar>();
            }

            // First term
            rbm.compute_grad1(mb_dat);

            // Second term
            rbm.zero_grad2();
            if(persistent) {
                rbm.accumulate_grad2_pcdk(ngibbs);
            } else {
                // Initial values for Gibbs sampler
                rbm.init_v0(dat);
                rbm.accumulate_grad2_cdk(ngibbs);
            }

            // Update parameters
            rbm.update_param(lr, nchain);

            // Compute log-likelihood every `neval_mb` mini-batches
            if(batch_id % neval_mb == 0)
            {
                if(eval_loglik)
                {
                    const Scalar res = rbm.loglik(dat, exact_loglik, neval_dat, neval_mcmc, neval_step);
                    loglik.push_back(res);
                } else {
                    loglik.push_back(NumericVector::get_na());
                }
            }
        }
    }

    return List::create(
        Rcpp::Named("w") = rbm.get_w(),
        Rcpp::Named("b") = rbm.get_b(),
        Rcpp::Named("c") = rbm.get_c(),
        Rcpp::Named("loglik") = loglik
    );
}

// [[Rcpp::export]]
List rbm_cdk(
    int vis_dim, int hid_dim, MapMat dat,
    int batch_size = 10, double lr = 0.1, int niter = 100,
    int ngibbs = 10, int nchain = 1, bool persistent = false,
    bool eval_loglik = false, bool exact_loglik = true,
    int neval_mb = 10, int neval_dat = 1000, int neval_mcmc = 100, int neval_step = 10,
    int verbose = 0
)
{
    const int m = vis_dim;
    const int n = hid_dim;

    // Initial values
    VectorXd b0(m), c0(n);
    MatrixXd w0(m, n);

    MapVec b(b0.data(), m);
    MapVec c(c0.data(), n);
    MapMat w(w0.data(), m, n);

    random_normal(b.data(), m, 0.0, 0.1);
    random_normal(c.data(), n, 0.0, 0.1);
    random_normal(w.data(), m * n, 0.0, 0.1);

    return rbm_cdk_warm(vis_dim, hid_dim, dat, b, c, w,
                        batch_size, lr, niter, ngibbs, nchain, persistent,
                        eval_loglik, exact_loglik,
                        neval_mb, neval_dat, neval_mcmc, neval_step,
                        verbose);
}

// dat [m x N]
// [[Rcpp::export]]
List rbm_fit_warm(
    int vis_dim, int hid_dim, MapMat dat,
    MapVec b0, MapVec c0, MapMat w0,
    int batch_size = 10, double lr = 0.1, int niter = 100,
    int min_mcmc = 1, int max_mcmc = 100, int nchain = 1,
    bool eval_loglik = false, bool exact_loglik = false,
    int neval_mb = 10, int neval_dat = 1000, int neval_mcmc = 100, int neval_step = 10,
    int verbose = 0
)
{
    typedef float Scalar;
    typedef Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic> Matrix;

    const int m = vis_dim;
    const int n = hid_dim;
    const int N = dat.cols();

    if(dat.rows() != m)
        Rcpp::stop("dimensions do not match");

    // Indices of observations
    VectorXi ind = VectorXi::LinSpaced(N, 0, N - 1);

    // RBM model
    RBM<Scalar> rbm(m, n, nchain, b0, c0, w0);

    // log-likelihood value
    std::vector<Scalar> loglik;

    // Average length of Markov chains
    std::vector<Scalar> tau;

    // Average number of discarded samples in coupling
    std::vector<Scalar> disc;

    for(int k = 0; k < niter; k++)
    {
        if(verbose > 0)
            Rcpp::Rcout << "\n===== Iter " << k + 1 << " =====" << std::endl;

        // Shuffle observations
        shuffle(ind);

        // Average length of Markov chains
        Scalar tau_sum = 0.0;

        // Average number of discarded samples in coupling
        Scalar disc_sum = 0.0;

        // Update on mini-batches
        for(int i = 0; i < N; i += batch_size)
        {
            const int batch_id = i / batch_size + 1;
            if(verbose > 1)
                Rcpp::Rcout << "==> Mini-batch " << batch_id << std::endl;

            // Indices for this mini-batch: i, i+1, ..., i+bs-1
            const int bs = std::min(i + batch_size, N) - i;

            // Mini-batch data
            Matrix mb_dat(m, bs);
            for(int j = 0; j < bs; j++)
            {
                mb_dat.col(j).noalias() = dat.col(ind[i + j]).cast<Scalar>();
            }

            // First term
            rbm.compute_grad1(mb_dat);

            // Random seeds for parallel computing
            Rcpp::IntegerVector seeds = Rcpp::sample(100000, nchain);

            // Initial values for Gibbs sampler
            rbm.init_v0(dat);

            // Second term
            rbm.zero_grad2();
            #pragma omp parallel for shared(seeds, rbm) reduction(+:tau_sum, disc_sum) schedule(dynamic)
            for(int j = 0; j < nchain; j++)
            {
                Scalar tau_t = 0.0, disc_t = 0.0;
                rbm.accumulate_grad2_ucd(j, seeds[j], min_mcmc, max_mcmc, verbose, tau_t, disc_t);

                tau_sum += tau_t;
                disc_sum += disc_t;
            }

            // Update parameters
            rbm.update_param(lr, nchain);

            // Compute log-likelihood every `neval_mb` mini-batches
            if(batch_id % neval_mb == 0)
            {
                if(eval_loglik)
                {
                    const Scalar res = rbm.loglik(dat, exact_loglik, neval_dat, neval_mcmc, neval_step);
                    loglik.push_back(res);
                } else {
                    loglik.push_back(NumericVector::get_na());
                }

                // Compute average number of discarded samples in coupling
                disc.push_back(disc_sum / Scalar(neval_mb * nchain));
                disc_sum = 0.0;

                // Compute average chain length and reset taui
                tau.push_back(tau_sum / Scalar(neval_mb * nchain));
                tau_sum = 0.0;
            }
        }
    }

    return List::create(
        Rcpp::Named("w") = rbm.get_w(),
        Rcpp::Named("b") = rbm.get_b(),
        Rcpp::Named("c") = rbm.get_c(),
        Rcpp::Named("loglik") = loglik,
        Rcpp::Named("tau") = tau,
        Rcpp::Named("disc") = disc
    );
}

// [[Rcpp::export]]
List rbm_fit(
    int vis_dim, int hid_dim, MapMat dat,
    int batch_size = 10, double lr = 0.1, int niter = 100,
    int min_mcmc = 1, int max_mcmc = 100, int nchain = 1,
    bool eval_loglik = false, bool exact_loglik = false,
    int neval_mb = 10, int neval_dat = 1000, int neval_mcmc = 100, int neval_step = 10,
    int verbose = 0
)
{
    const int m = vis_dim;
    const int n = hid_dim;

    // Initial values
    VectorXd b0(m), c0(n);
    MatrixXd w0(m, n);

    MapVec b(b0.data(), m);
    MapVec c(c0.data(), n);
    MapMat w(w0.data(), m, n);

    random_normal(b.data(), m, 0.0, 0.1);
    random_normal(c.data(), n, 0.0, 0.1);
    random_normal(w.data(), m * n, 0.0, 0.1);

    return rbm_fit_warm(vis_dim, hid_dim, dat, b, c, w,
                        batch_size, lr, niter,
                        min_mcmc, max_mcmc, nchain,
                        eval_loglik, exact_loglik,
                        neval_mb, neval_dat, neval_mcmc, neval_step,
                        verbose);
}
