#include <RcppEigen.h>
#include "mcmc.h"
#include "utils.h"
#include "likelihood.h"

template <typename Scalar = float>
class RBM
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

    Vector    m_db1;
    Vector    m_dc1;
    Matrix    m_dw1;

    Vector    m_db2;
    Vector    m_dc2;
    Matrix    m_dw2;

    Matrix    m_v0;
    Matrix    m_vchains;
    Matrix    m_hchains;

public:
    RBM(int m, int n, int nchain) :
        m_m(m), m_n(n), m_nchain(nchain),
        m_b(m), m_c(n), m_w(m, n),
        m_db1(m), m_dc1(n), m_dw1(m, n), m_db2(m), m_dc2(n), m_dw2(m, n),
        m_v0(m, nchain), m_vchains(m, nchain), m_hchains(n, nchain)
    {
        random_normal(m_b.data(), m, Scalar(0), Scalar(0.1));
        random_normal(m_c.data(), n, Scalar(0), Scalar(0.1));
        random_normal(m_w.data(), m * n, Scalar(0), Scalar(0.1));
    }

    template <typename OtherScalar>
    RBM(int m, int n, int nchain,
        Eigen::Map< Eigen::Matrix<OtherScalar, Eigen::Dynamic, 1> > b0,
        Eigen::Map< Eigen::Matrix<OtherScalar, Eigen::Dynamic, 1> > c0,
        Eigen::Map< Eigen::Matrix<OtherScalar, Eigen::Dynamic, Eigen::Dynamic> > w0) :
        m_m(m), m_n(n), m_nchain(nchain),
        m_b(m), m_c(n), m_w(m, n),
        m_db1(m), m_dc1(n), m_dw1(m, n), m_db2(m), m_dc2(n), m_dw2(m, n),
        m_v0(m, nchain), m_vchains(m, nchain), m_hchains(n, nchain)
    {
        m_b.noalias() = b0.template cast<Scalar>();
        m_c.noalias() = c0.template cast<Scalar>();
        m_w.noalias() = w0.template cast<Scalar>();
    }

    // Log-likelihood value
    template <typename Derived>
    Scalar loglik(Eigen::MatrixBase<Derived>& dat, bool exact = false, int nobs = 100, int nmc = 30, int nstep = 10) const
    {
        // Get a subset of data
        const int N = dat.cols();
        nobs = std::min(nobs, N);
        Matrix subdat(m_m, nobs);
        if(exact)
        {
            subdat.noalias() = dat.leftCols(nobs).template cast<Scalar>();
        } else {
            for(int i = 0; i < nobs; i++)
            {
                subdat.col(i).noalias() = dat.col(int(R::unif_rand() * N)).template cast<Scalar>();
            }
        }

        const Scalar res = exact ?
            (loglik_rbm_exact(m_m, m_n, nobs, m_w.data(), m_b.data(), m_c.data(), subdat.data())) :
            (loglik_rbm_approx(m_m, m_n, nobs, m_w.data(), m_b.data(), m_c.data(), subdat.data(), nmc, nstep));

        return res;
    }

    // First term of the gradient
    // Mini-batch vmb [m x b]
    void compute_grad1(RefConstMat& vmb)
    {
        const int bs = vmb.cols();

        // Mean of hidden units given vmb
        Matrix hmean(m_n, bs);
        hmean.noalias() = m_w.transpose() * vmb;
        hmean.colwise() += m_c;
        apply_sigmoid(hmean);

        m_db1.noalias() = vmb.rowwise().mean();
        m_dc1.noalias() = hmean.rowwise().mean();
        m_dw1.noalias() = (1.0 / bs) * vmb * hmean.transpose();
    }

    // Zero out gradients
    void zero_grad2()
    {
        m_db2.setZero();
        m_dc2.setZero();
        m_dw2.setZero();
    }

    // Initialize Gibbs sampler using ramdomly selected observations
    // dat [m x N]
    template <typename Derived>
    void init_v0(Eigen::MatrixBase<Derived>& dat)
    {
        const int N = dat.cols();
        for(int i = 0; i < m_nchain; i++)
        {
            m_v0.col(i).noalias() = dat.col(int(R::unif_rand() * N)).template cast<Scalar>();
        }
    }

    // Compute the second term of gradient using CD-k
    void accumulate_grad2_cdk(int k)
    {
        // Gibbs samples
        RBMSampler<Scalar> sampler(m_w, m_b, m_c);
        sampler.sample_k_mc(m_v0, m_vchains, m_hchains, k, m_nchain);

        // Second term of gradient
        m_hchains.noalias() = m_w.transpose() * m_vchains;
        m_hchains.colwise() += m_c;
        apply_sigmoid(m_hchains);

        m_db2.noalias() = m_vchains.rowwise().sum();
        m_dc2.noalias() = m_hchains.rowwise().sum();
        m_dw2.noalias() = m_vchains * m_hchains.transpose();
    }

    // Compute the second term of gradient using PCD-k
    void accumulate_grad2_pcdk(int k)
    {
        // Gibbs samples
        RBMSampler<Scalar> sampler(m_w, m_b, m_c);
        Matrix& vchains = m_v0;
        // vchains will be updated to the last state of the Markov chain
        sampler.sample_k_mc(vchains, vchains, m_hchains, k, m_nchain);

        // Second term of gradient
        m_hchains.noalias() = m_w.transpose() * vchains;
        m_hchains.colwise() += m_c;
        apply_sigmoid(m_hchains);

        m_db2.noalias() = vchains.rowwise().sum();
        m_dc2.noalias() = m_hchains.rowwise().sum();
        m_dw2.noalias() = vchains * m_hchains.transpose();
    }

    // Compute the second term of gradient using unbiased CD
    void accumulate_grad2_ucd(int id, int seed, int min_mcmc, int max_mcmc, int verbose, Scalar& tau_t, Scalar& disc_t)
    {
        // Sampler
        std::mt19937 gen(seed);
        RBMSampler<Scalar> sampler(m_w, m_b, m_c);

        // MCMC path
        Vector vk(m_m), hk_mean(m_n);
        Matrix vhist, vchist;

        // # discarded samples
        disc_t = sampler.sample(gen, m_v0.col(id), vhist, vchist, min_mcmc, max_mcmc, verbose > 2);
        const int burnin = min_mcmc - 1;
        const int remain = vchist.cols() - burnin;
        // Chain length
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

        // Compute the second term of gradient
        Vector db_t = Vector::Zero(m_m), dc_t = Vector::Zero(m_n);
        Matrix dw_t = Matrix::Zero(m_m, m_n);

        db_t.noalias() = vk + vhist.rightCols(remain).rowwise().sum() -
            vchist.rightCols(remain).rowwise().sum();
        dc_t.noalias() = hk_mean + hhist_mean.rowwise().sum() -
            hchist_mean.rowwise().sum();
        dw_t.noalias() = vk * hk_mean.transpose() +
            vhist.rightCols(remain) * hhist_mean.transpose() -
            vchist.rightCols(remain) * hchist_mean.transpose();

        // Accumulate gradients
        #pragma omp critical
        {
            m_db2.noalias() += db_t;
            m_dc2.noalias() += dc_t;
            m_dw2.noalias() += dw_t;
        }
    }

    void update_param(Scalar lr, int n2)
    {
        m_b.noalias() += lr * (m_db1 - m_db2 / n2);
        m_c.noalias() += lr * (m_dc1 - m_dc2 / n2);
        m_w.noalias() += lr * (m_dw1 - m_dw2 / n2);
    }

    const Matrix& get_w() const { return m_w; }
    const Vector& get_b() const { return m_b; }
    const Vector& get_c() const { return m_c; }
};


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
    int ngibbs = 10, int nchain = 1,
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

            // Initial values for Gibbs sampler
            rbm.init_v0(dat);

            // Second term
            rbm.zero_grad2();
            rbm.accumulate_grad2_cdk(ngibbs);

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
            rbm.accumulate_grad2_pcdk(ngibbs);

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
