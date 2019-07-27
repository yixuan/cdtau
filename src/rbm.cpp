#include "mcmc.h"
#include "utils.h"

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
typedef Eigen::Map<VectorXf> MapVecf;
typedef Eigen::Map<MatrixXf> MapMatf;

double loglik_rbm(MapMat w, MapVec b, MapVec c, MapMat dat);
float loglik_rbm(MapMatf w, MapVecf b, MapVecf c, MapMatf dat);

double loglik_rbm_approx(MapMat w, MapVec b, MapVec c, MapMat dat,
                         int nsamp = 100, int nstep = 100);
float loglik_rbm_approx(MapMatf w, MapVecf b, MapVecf c, MapMatf dat,
                         int nsamp = 100, int nstep = 100);

// dat [m x N]
// [[Rcpp::export]]
List rbm_cdk(
    int vis_dim, int hid_dim, MapMat dat,
    int batch_size = 10, double lr = 0.1, int niter = 100,
    int ngibbs = 10, int nchain = 1,
    bool eval_loglik = false, bool exact_loglik = true,
    int neval_mb = 10, int neval_dat = 1000, int neval_mcmc = 100, int neval_step = 10,
    int verbose = 0
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

    // log-likelihood value
    std::vector<double> loglik;

    VectorXd v0(m), v(m), h(n), h0mean(n);
    MatrixXd vchains(m, nchain), hmeanchains(n, nchain);
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
            // Initialize gradients and the sampler
            db.setZero();
            dc.setZero();
            dw.setZero();
            RBMSampler<double> sampler(w, b, c);

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

            // Compute log-likelihood every `neval_mb` mini-batches
            if(batch_id % neval_mb == 0)
            {
                if(eval_loglik)
                {
                    // Get a subset of data
                    neval_dat = std::min(neval_dat, N);
                    shuffle(ind);
                    MatrixXd subdat(m, neval_dat);
                    for(int s = 0; s < neval_dat; s++)
                    {
                        subdat.col(s).noalias() = dat.col(ind[s]);
                    }

                    // Compute the loglikelihood value
                    MapMat mw(w.data(), m, n);
                    MapVec mb(b.data(), m);
                    MapVec mc(c.data(), n);
                    MapMat mdat(subdat.data(), m, neval_dat);

                    const double res = exact_loglik ?
                                       (loglik_rbm(mw, mb, mc, mdat)) :
                                       (loglik_rbm_approx(mw, mb, mc, mdat, neval_mcmc, neval_step));
                    loglik.push_back(res);
                } else {
                    loglik.push_back(NumericVector::get_na());
                }
            }
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
    bool eval_loglik = false, bool exact_loglik = false,
    int neval_mb = 10, int neval_dat = 1000, int neval_mcmc = 100, int neval_step = 10,
    int verbose = 0
)
{
    typedef float Scalar;
    typedef Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic> Matrix;
    typedef Eigen::Matrix<Scalar, Eigen::Dynamic, 1> Vector;
    typedef Eigen::Map<Matrix> MapMat;
    typedef Eigen::Map<Vector> MapVec;

    const int m = vis_dim;
    const int n = hid_dim;
    const int N = dat.cols();

    if(dat.rows() != m)
        Rcpp::stop("dimensions do not match");

    // Indices of observations
    VectorXi ind = VectorXi::LinSpaced(N, 0, N - 1);

    // Parameters and derivatives
    Vector b(m), db(m), c(n), dc(n);
    Matrix w(m, n), dw(m, n);
    random_normal(b.data(), m, Scalar(0), Scalar(0.1));
    random_normal(c.data(), n, Scalar(0), Scalar(0.1));
    random_normal(w.data(), m * n, Scalar(0), Scalar(0.1));

    // log-likelihood value
    std::vector<double> loglik;

    // Average length of Markov chains
    std::vector<double> tau;

    // Average number of discarded samples in coupling
    std::vector<double> disc;

    for(int k = 0; k < niter; k++)
    {
        if(verbose > 0)
            Rcpp::Rcout << "\n===== Iter " << k + 1 << " =====" << std::endl;

        // Shuffle observations
        shuffle(ind);

        // Compute length of Markov chains
        double tau_sum = 0.0;
        int tau_bs = 0;

        // Compute number of discarded samples in coupling
        double disc_sum = 0.0;

        // Update on mini-batches
        for(int i = 0; i < N; i += batch_size)
        {
            const int batch_id = i / batch_size + 1;
            if(verbose > 1)
                Rcpp::Rcout << "==> Mini-batch " << batch_id << std::endl;

            // Indices for this mini-batch: i, i+1, ..., i+bs-1
            const int bs = std::min(i + batch_size, N) - i;
            // Initialize gradients and the sampler
            db.setZero();
            dc.setZero();
            dw.setZero();

            Rcpp::IntegerVector seeds = Rcpp::sample(100000, bs);

            // Compute gradient
            #pragma omp parallel for shared(seeds, b, c, w, db, dc, dw) reduction(+:tau_bs, tau_sum, disc_sum) schedule(dynamic)
            for(int j = 0; j < bs; j++)
            {
                RBMSampler<Scalar> sampler(w, b, c);
                Vector v0(m), v1(m), h0_mean(n), h1_mean(n);
                Matrix vhist, vchist;
                std::mt19937 gen(seeds[j]);

                double tau_t = 0.0, disc_t = 0.0;
                Vector db_t(m), dc_t(n);
                Matrix dw_t(m, n);
                db_t.setZero();
                dc_t.setZero();
                dw_t.setZero();

                v0.noalias() = dat.col(ind[i + j]).cast<Scalar>();
                h0_mean.noalias() = w.transpose() * v0 + c;
                apply_sigmoid(h0_mean);

                for(int l = 0; l < nchain; l++)
                {
                    disc_t += sampler.sample(gen, v0, vhist, vchist, min_mcmc, max_mcmc, verbose > 2);
                    const int burnin = min_mcmc - 1;
                    const int remain = vchist.cols() - burnin;
                    tau_t += vchist.cols();

                    v1.noalias() = vhist.col(burnin);
                    h1_mean.noalias() = w.transpose() * v1 + c;
                    apply_sigmoid(h1_mean);

                    Matrix hhist_mean = w.transpose() * vhist.rightCols(remain);
                    hhist_mean.colwise() += c;
                    apply_sigmoid(hhist_mean);

                    Matrix hchist_mean = w.transpose() * vchist.rightCols(remain);
                    hchist_mean.colwise() += c;
                    apply_sigmoid(hchist_mean);

                    db_t.noalias() += (-v1 -
                        vhist.rightCols(remain).rowwise().sum() +
                        vchist.rightCols(remain).rowwise().sum());
                    dc_t.noalias() += (-h1_mean -
                        hhist_mean.rowwise().sum() +
                        hchist_mean.rowwise().sum());
                    dw_t.noalias() += (-v1 * h1_mean.transpose() -
                        vhist.rightCols(remain) * hhist_mean.transpose() +
                        vchist.rightCols(remain) * hchist_mean.transpose());
                }

                db_t.noalias() += double(nchain) * v0;
                dc_t.noalias() += double(nchain) * h0_mean;
                dw_t.noalias() += double(nchain) * v0 * h0_mean.transpose();

                tau_bs += nchain;
                tau_sum += tau_t;
                disc_sum += disc_t;

                #pragma omp critical
                {
                    db.noalias() += db_t;
                    dc.noalias() += dc_t;
                    dw.noalias() += dw_t;
                }
            }

            b.noalias() += lr / double(bs * nchain) * db;
            c.noalias() += lr / double(bs * nchain) * dc;
            w.noalias() += lr / double(bs * nchain) * dw;

            // Compute log-likelihood every `neval_mb` mini-batches
            if(batch_id % neval_mb == 0)
            {
                if(eval_loglik)
                {
                    // Get a subset of data
                    neval_dat = std::min(neval_dat, N);
                    shuffle(ind);
                    Matrix subdat(m, neval_dat);
                    for(int s = 0; s < neval_dat; s++)
                    {
                        subdat.col(s).noalias() = dat.col(ind[s]).cast<Scalar>();
                    }

                    // Compute the loglikelihood value
                    MapMat mw(w.data(), m, n);
                    MapVec mb(b.data(), m);
                    MapVec mc(c.data(), n);
                    MapMat mdat(subdat.data(), m, neval_dat);

                    const double res = exact_loglik ?
                    (loglik_rbm(mw, mb, mc, mdat)) :
                        (loglik_rbm_approx(mw, mb, mc, mdat, neval_mcmc, neval_step));
                    loglik.push_back(res);
                } else {
                    loglik.push_back(NumericVector::get_na());
                }

                // Compute average number of discarded samples in coupling
                disc.push_back(disc_sum / tau_bs);
                disc_sum = 0.0;

                // Compute average chain length and reset taui
                tau.push_back(tau_sum / tau_bs);
                tau_sum = 0.0;
                tau_bs = 0;
            }
        }
    }

    return List::create(
        Rcpp::Named("w") = w,
        Rcpp::Named("b") = b,
        Rcpp::Named("c") = c,
        Rcpp::Named("loglik") = loglik,
        Rcpp::Named("tau") = tau,
        Rcpp::Named("disc") = disc
    );
}

