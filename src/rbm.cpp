#include <RcppEigen.h>
#include "mcmc.h"
#include "utils.h"
#include "likelihood.h"

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

// dat [m x N]
// [[Rcpp::export]]
List rbm_cdk_warm(
    int vis_dim, int hid_dim, MapMat dat,
    MapVec b0, MapVec c0, MapMat w0,
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
    VectorXd b = b0, db(m), c = c0, dc(n);
    MatrixXd w = w0, dw(m, n);

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

                sampler.sample_k_mc(v0, vchains, hmeanchains, ngibbs, nchain);

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
                    const double res = exact_loglik ?
                        (loglik_rbm_exact(m, n, neval_dat, w.data(), b.data(), c.data(), subdat.data())) :
                        (loglik_rbm_approx(m, n, neval_dat, w.data(), b.data(), c.data(), subdat.data(), neval_mcmc, neval_step));
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
                        batch_size, lr, niter, ngibbs, nchain,
                        eval_loglik, exact_loglik,
                        neval_mb, neval_dat, neval_mcmc, neval_step,
                        verbose);
}

// dat [m x N]
// [[Rcpp::export]]
List rbm_pcdk_warm(
        int vis_dim, int hid_dim, MapMat dat,
        MapVec b0, MapVec c0, MapMat w0,
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
    VectorXd b = b0, db(m), c = c0, dc(n);
    MatrixXd w = w0, dw(m, n);

    // log-likelihood value
    std::vector<double> loglik;

    VectorXd v0(m), h0mean(n);
    MatrixXd vchains(m, nchain), hmeanchains(n, nchain);

    vchains.noalias() = dat.leftCols(nchain);
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

                sampler.sample_k_mc(vchains, vchains, hmeanchains, ngibbs, nchain);

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
                    const double res = exact_loglik ?
                    (loglik_rbm_exact(m, n, neval_dat, w.data(), b.data(), c.data(), subdat.data())) :
                        (loglik_rbm_approx(m, n, neval_dat, w.data(), b.data(), c.data(), subdat.data(), neval_mcmc, neval_step));
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

// [[Rcpp::export]]
List rbm_pcdk(
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

    // Initial values
    VectorXd b0(m), c0(n);
    MatrixXd w0(m, n);

    MapVec b(b0.data(), m);
    MapVec c(c0.data(), n);
    MapMat w(w0.data(), m, n);

    random_normal(b.data(), m, 0.0, 0.1);
    random_normal(c.data(), n, 0.0, 0.1);
    random_normal(w.data(), m * n, 0.0, 0.1);

    return rbm_pcdk_warm(vis_dim, hid_dim, dat, b, c, w,
                        batch_size, lr, niter, ngibbs, nchain,
                        eval_loglik, exact_loglik,
                        neval_mb, neval_dat, neval_mcmc, neval_step,
                        verbose);
}



template <typename Scalar = float>
class RBMUCD
{
private:
    typedef Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic> Matrix;
    typedef Eigen::Matrix<Scalar, Eigen::Dynamic, 1> Vector;
    typedef const Eigen::Ref<const Matrix> RefConstMat;
    typedef const Eigen::Ref<const Vector> RefConstVec;

    const int m_m;
    const int m_n;
    const int m_nchain;

    Vector    m_b;
    Vector    m_c;
    Matrix    m_w;

    Vector    m_db;
    Vector    m_dc;
    Matrix    m_dw;

    Matrix    m_vinit;

public:
    RBMUCD(int m, int n, int nchain) :
        m_m(m), m_n(n), m_nchain(nchain),
        m_b(m), m_c(n), m_w(m, n), m_db(m), m_dc(n), m_dw(m, n), m_vinit(m, nchain)
    {
        random_normal(m_b.data(), m, Scalar(0), Scalar(0.1));
        random_normal(m_c.data(), n, Scalar(0), Scalar(0.1));
        random_normal(m_w.data(), m * n, Scalar(0), Scalar(0.1));
    }

    template <typename OtherScalar>
    RBMUCD(int m, int n, int nchain,
           Eigen::Map< Eigen::Matrix<OtherScalar, Eigen::Dynamic, 1> > b0,
           Eigen::Map< Eigen::Matrix<OtherScalar, Eigen::Dynamic, 1> > c0,
           Eigen::Map< Eigen::Matrix<OtherScalar, Eigen::Dynamic, Eigen::Dynamic> > w0) :
        m_m(m), m_n(n), m_nchain(nchain),
        m_b(m), m_c(n), m_w(m, n), m_db(m), m_dc(n), m_dw(m, n), m_vinit(m, nchain)
    {
        m_b.noalias() = b0.template cast<Scalar>();
        m_c.noalias() = c0.template cast<Scalar>();
        m_w.noalias() = w0.template cast<Scalar>();
    }

    void init_v(RefConstMat& v0)
    {
        m_vinit.noalias() = v0.leftCols(m_nchain);
    }

    void zero_grad()
    {
        m_db.setZero();
        m_dc.setZero();
        m_dw.setZero();
    }

    // vi is one observation
    void accumulate_grad(
        RefConstVec& vi, int seed, int min_mcmc, int max_mcmc, int verbose,
        Scalar& tau_t, Scalar& disc_t
    )
    {
        // First term of the gradient
        // Mean of hidden units given vi
        Vector hi_mean(m_n);
        hi_mean.noalias() = m_w.transpose() * vi + m_c;
        apply_sigmoid(hi_mean);

        // Sampler
        std::mt19937 gen(seed);
        RBMSampler<Scalar> sampler(m_w, m_b, m_c);
        // Make a copy of the initial values of v
        Matrix vinit = m_vinit;

        // MCMC path
        Vector vk(m_m), hk_mean(m_n);
        Matrix vhist, vchist;

        // Gradients for this observation
        Vector db_t = Vector::Zero(m_m), dc_t = Vector::Zero(m_n);
        Matrix dw_t = Matrix::Zero(m_m, m_n);

        // Averagr chain length and # discarded samples
        tau_t = 0.0;
        disc_t = 0.0;

        // Run multiple chains
        for(int l = 0; l < m_nchain; l++)
        {
            disc_t += sampler.sample(gen, vinit.col(l), vhist, vchist, min_mcmc, max_mcmc, verbose > 2);
            const int burnin = min_mcmc - 1;
            const int remain = vchist.cols() - burnin;
            tau_t += vchist.cols();

            vk.noalias() = vhist.col(burnin);
            hk_mean.noalias() = m_w.transpose() * vk + m_c;
            apply_sigmoid(hk_mean);

            Matrix hhist_mean = m_w.transpose() * vhist.rightCols(remain);
            hhist_mean.colwise() += m_c;
            apply_sigmoid(hhist_mean);

            Matrix hchist_mean = m_w.transpose() * vchist.rightCols(remain);
            hchist_mean.colwise() += m_c;
            apply_sigmoid(hchist_mean);

            // Accumulate gradients
            db_t.noalias() += (
                vk + vhist.rightCols(remain).rowwise().sum() -
                vchist.rightCols(remain).rowwise().sum()
            );
            dc_t.noalias() += (
                hk_mean + hhist_mean.rowwise().sum() -
                hchist_mean.rowwise().sum()
            );
            dw_t.noalias() += (
                vk * hk_mean.transpose() + vhist.rightCols(remain) * hhist_mean.transpose() -
                vchist.rightCols(remain) * hchist_mean.transpose()
            );

            // Update initial value for v
            vinit.col(l).noalias() = vchist.template rightCols<1>();
        }

        tau_t /= Scalar(m_nchain);
        disc_t /= Scalar(m_nchain);

        // Add first term
        db_t.noalias() = vi - db_t / Scalar(m_nchain);
        dc_t.noalias() = hi_mean - dc_t / Scalar(m_nchain);
        dw_t.noalias() = vi * hi_mean.transpose() - dw_t / Scalar(m_nchain);

        // Update mini-batch gradients and initial values for v
        #pragma omp critical
        {
            m_db.noalias() += db_t;
            m_dc.noalias() += dc_t;
            m_dw.noalias() += dw_t;
            m_vinit.noalias() = vinit;
        }
    }

    void update_param(Scalar lr, int batch_size)
    {
        m_b.noalias() += lr / Scalar(batch_size) * m_db;
        m_c.noalias() += lr / Scalar(batch_size) * m_dc;
        m_w.noalias() += lr / Scalar(batch_size) * m_dw;
    }

    Scalar loglikelihood(RefConstMat dat, bool exact, int nsamp, int nstep) const
    {
        return exact ?
            (loglik_rbm_exact(m_m, m_n, dat.cols(), m_w.data(), m_b.data(), m_c.data(), dat.data())) :
            (loglik_rbm_approx(m_m, m_n, dat.cols(), m_w.data(), m_b.data(), m_c.data(), dat.data(), nsamp, nstep));
    }

    const Matrix& get_w() const { return m_w; }
    const Vector& get_b() const { return m_b; }
    const Vector& get_c() const { return m_c; }
};


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
    typedef Eigen::Matrix<Scalar, Eigen::Dynamic, 1> Vector;

    const int m = vis_dim;
    const int n = hid_dim;
    const int N = dat.cols();

    if(dat.rows() != m)
        Rcpp::stop("dimensions do not match");

    // Indices of observations
    VectorXi ind = VectorXi::LinSpaced(N, 0, N - 1);

    // RBM model
    RBMUCD<Scalar> rbm(m, n, nchain, b0, c0, w0);
    rbm.init_v(dat.leftCols(nchain).cast<Scalar>());

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
        int bs_sum = 0;

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

            // Random seeds for parallel computing
            Rcpp::IntegerVector seeds = Rcpp::sample(100000, bs);

            // Initialize gradients
            rbm.zero_grad();

            // Compute gradients
            #pragma omp parallel for shared(seeds, rbm, dat) reduction(+:tau_sum, disc_sum) schedule(dynamic)
            for(int j = 0; j < bs; j++)
            {
                Vector vi = dat.col(ind[i + j]).cast<Scalar>();
                Scalar tau_t = 0.0, disc_t = 0.0;

                // rbm.init_v(vi.replicate(1, nchain));
                rbm.accumulate_grad(vi, seeds[j], min_mcmc, max_mcmc, verbose, tau_t, disc_t);

                tau_sum += tau_t;
                disc_sum += disc_t;
            }

            bs_sum += bs;

            // Update parameters
            rbm.update_param(lr, bs);

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
                    loglik.push_back(rbm.loglikelihood(subdat, exact_loglik, neval_mcmc, neval_step));
                } else {
                    loglik.push_back(NumericVector::get_na());
                }

                // Compute average number of discarded samples in coupling
                disc.push_back(disc_sum / bs_sum);
                disc_sum = 0.0;

                // Compute average chain length and reset taui
                tau.push_back(tau_sum / bs_sum);
                tau_sum = 0.0;
                bs_sum = 0;
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
