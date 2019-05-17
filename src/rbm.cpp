#include "mcmc.h"
#include "utils.h"

using Rcpp::NumericMatrix;
using Rcpp::NumericVector;
using Rcpp::List;
using Eigen::VectorXi;
using Eigen::VectorXd;
using Eigen::MatrixXd;
typedef Eigen::Map<VectorXd> MapVec;
typedef Eigen::Map<MatrixXd> MapMat;

double loglik_rbm(MapMat w, MapVec b, MapVec c, MapMat v);

inline double loglik_approx(
    const MatrixXd& w, const VectorXd& b, const VectorXd& c, const MatrixXd& dat,
    int nsamp = 100, int nstep = 100
)
{
    const int m = w.rows();
    const int n = w.cols();
    const int N = dat.cols();

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

// dat [m x N]
// [[Rcpp::export]]
List rbm_cdk(
    int vis_dim, int hid_dim, MapMat dat,
    int batch_size = 10, double lr = 0.1, int niter = 100,
    int ngibbs = 10, int nchain = 1,
    bool eval_loglik = false, bool exact_loglik = true, int verbose = 0
)
{
    const int m = vis_dim;
    const int n = hid_dim;
    const int N = dat.cols();

    if(dat.rows() != m)
        Rcpp::stop("dimensions do not match");

    // Indices of observations
    VectorXi ind = VectorXi::LinSpaced(N, 0, N - 1);

    // Parameters and derivatives
    VectorXd b(m), db(m), c(n), dc(n);
    MatrixXd w(m, n), dw(m, n);
    random_normal(b.data(), m, 0.0, 0.1);
    random_normal(c.data(), n, 0.0, 0.1);
    random_normal(w.data(), m * n, 0.0, 0.1);

    // log-likelihood value in each iteration
    NumericVector loglik(niter);

    VectorXd v0(m), v(m), h(n), h0mean(n);
    MatrixXd vchains(m, nchain), hmeanchains(n, nchain);
    for(int k = 0; k < niter; k++)
    {
        if(verbose > 0)
            Rcpp::Rcout << "\n===== Iter " << k << " =====" << std::endl;

        // Shuffle observations
        shuffle(ind);

        // Update on mini-batches
        for(int i = 0; i < N; i += batch_size)
        {
            if(verbose > 1)
                Rcpp::Rcout << "==> Mini-batch " << i / batch_size << std::endl;

            // Indices for this mini-batch: i, i+1, ..., i+bs-1
            const int bs = std::min(i + batch_size, N) - i;
            // Initialize gradients and the sampler
            db.setZero();
            dc.setZero();
            dw.setZero();
            RBMSampler sampler(w, b, c);

            // Compute gradient
            for(int j = 0; j < bs; j++)
            {
                v0.noalias() = dat.col(ind[i + j]);
                h0mean.noalias() = w.transpose() * v0 + c;
                apply_sigmoid(h0mean);

                for(int l = 0; l < nchain; l++)
                {
                    sampler.sample_k(v0, v, h, ngibbs);
                    vchains.col(l).noalias() = v;
                }

                hmeanchains.noalias() = w.transpose() * vchains;
                hmeanchains.colwise() += c;
                apply_sigmoid(hmeanchains);

                db.noalias() += (v0 - vchains.rowwise().mean());
                dc.noalias() += (h0mean - hmeanchains.rowwise().mean());
                dw.noalias() += (v0 * h0mean.transpose() -
                    (1.0 / nchain) * vchains * hmeanchains.transpose());
            }

            b.noalias() += lr / double(bs) * db;
            c.noalias() += lr / double(bs) * dc;
            w.noalias() += lr / double(bs) * dw;
        }

        if(exact_loglik)
        {
            MapMat mw(w.data(), m, n);
            MapVec mb(b.data(), m);
            MapVec mc(c.data(), n);
            MapMat mdat(dat.data(), m, N);
            loglik[k] = eval_loglik ? (loglik_rbm(mw, mb, mc, mdat)) : (NumericVector::get_na());
        } else {
            loglik[k] = eval_loglik ? (loglik_approx(w, b, c, dat, 100, 10)) : (NumericVector::get_na());
        }
    }

    return List::create(
        Rcpp::Named("w") = w,
        Rcpp::Named("b") = b,
        Rcpp::Named("c") = c,
        Rcpp::Named("loglik") = loglik
    );
}

// dat [m x N]
// [[Rcpp::export]]
List rbm_fit(
    int vis_dim, int hid_dim, MapMat dat,
    int batch_size = 10, double lr = 0.1, int niter = 100,
    int min_mcmc = 1, int max_mcmc = 100, int nchain = 1,
    bool eval_loglik = false, bool exact_loglik = true, int verbose = 0
)
{
    const int m = vis_dim;
    const int n = hid_dim;
    const int N = dat.cols();

    if(dat.rows() != m)
        Rcpp::stop("dimensions do not match");

    // Indices of observations
    VectorXi ind = VectorXi::LinSpaced(N, 0, N - 1);

    // Parameters and derivatives
    VectorXd b(m), db(m), c(n), dc(n);
    MatrixXd w(m, n), dw(m, n);
    random_normal(b.data(), m, 0.0, 0.1);
    random_normal(c.data(), n, 0.0, 0.1);
    random_normal(w.data(), m * n, 0.0, 0.1);

    // log-likelihood value in each iteration
    NumericVector loglik(niter);

    // Average length of Markov chains in each iteration
    NumericVector tau(niter);

    VectorXd v0(m), v1(m), h0_mean(n), h1_mean(n);
    MatrixXd vhist, vchist;
    for(int k = 0; k < niter; k++)
    {
        if(verbose > 0)
            Rcpp::Rcout << "\n===== Iter " << k << " =====" << std::endl;

        // Shuffle observations
        shuffle(ind);

        // Update on mini-batches
        for(int i = 0; i < N; i += batch_size)
        {
            if(verbose > 1)
                Rcpp::Rcout << "==> Mini-batch " << i / batch_size << std::endl;

            // Indices for this mini-batch: i, i+1, ..., i+bs-1
            const int bs = std::min(i + batch_size, N) - i;
            // Initialize gradients and the sampler
            db.setZero();
            dc.setZero();
            dw.setZero();
            RBMSampler sampler(w, b, c);

            // Compute gradient
            for(int j = 0; j < bs; j++)
            {
                v0.noalias() = dat.col(ind[i + j]);
                h0_mean.noalias() = w.transpose() * v0 + c;
                apply_sigmoid(h0_mean);

                for(int l = 0; l < nchain; l++)
                {
                    sampler.sample(v0, vhist, vchist, min_mcmc, max_mcmc);
                    const int burnin = min_mcmc - 1;
                    const int remain = vchist.cols() - burnin;
                    tau[k] += vchist.cols();

                    v1.noalias() = vhist.col(burnin);
                    h1_mean.noalias() = w.transpose() * v1 + c;
                    apply_sigmoid(h1_mean);

                    MatrixXd hhist_mean = w.transpose() * vhist.rightCols(remain);
                    hhist_mean.colwise() += c;
                    apply_sigmoid(hhist_mean);

                    MatrixXd hchist_mean = w.transpose() * vchist.rightCols(remain);
                    hchist_mean.colwise() += c;
                    apply_sigmoid(hchist_mean);

                    db.noalias() += (-v1 -
                        vhist.rightCols(remain).rowwise().sum() +
                        vchist.rightCols(remain).rowwise().sum());
                    dc.noalias() += (-h1_mean -
                        hhist_mean.rowwise().sum() +
                        hchist_mean.rowwise().sum());
                    dw.noalias() += (-v1 * h1_mean.transpose() -
                        vhist.rightCols(remain) * hhist_mean.transpose() +
                        vchist.rightCols(remain) * hchist_mean.transpose());
                }

                db.noalias() += double(nchain) * v0;
                dc.noalias() += double(nchain) * h0_mean;
                dw.noalias() += double(nchain) * v0 * h0_mean.transpose();
            }

            b.noalias() += lr / double(bs * nchain) * db;
            c.noalias() += lr / double(bs * nchain) * dc;
            w.noalias() += lr / double(bs * nchain) * dw;
        }

        if(exact_loglik)
        {
            MapMat mw(w.data(), m, n);
            MapVec mb(b.data(), m);
            MapVec mc(c.data(), n);
            MapMat mdat(dat.data(), m, N);
            loglik[k] = eval_loglik ? (loglik_rbm(mw, mb, mc, mdat)) : (NumericVector::get_na());
        } else {
            loglik[k] = eval_loglik ? (loglik_approx(w, b, c, dat, 100, 10)) : (NumericVector::get_na());
        }

        tau[k] /= (N * nchain);
    }

    return List::create(
        Rcpp::Named("w") = w,
        Rcpp::Named("b") = b,
        Rcpp::Named("c") = c,
        Rcpp::Named("loglik") = loglik,
        Rcpp::Named("tau") = tau
    );
}
