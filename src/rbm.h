#ifndef CDTAU_RBM_H
#define CDTAU_RBM_H

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
        std::mt19937 gen(int(R::unif_rand() * 10000));
        RBMSampler<Scalar> sampler(m_w, m_b, m_c);
        sampler.sample_k_mc(gen, m_v0, m_vchains, m_hchains, k);

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
        std::mt19937 gen(int(R::unif_rand() * 10000));
        RBMSampler<Scalar> sampler(m_w, m_b, m_c);
        Matrix& vchains = m_v0;
        // vchains will be updated to the last state of the Markov chain
        sampler.sample_k_mc(gen, vchains, vchains, m_hchains, k);

        // Second term of gradient
        m_hchains.noalias() = m_w.transpose() * vchains;
        m_hchains.colwise() += m_c;
        apply_sigmoid(m_hchains);

        m_db2.noalias() = vchains.rowwise().sum();
        m_dc2.noalias() = m_hchains.rowwise().sum();
        m_dw2.noalias() = vchains * m_hchains.transpose();
    }

    // Compute the second term of gradient using unbiased CD
    void accumulate_grad2_ucd(int id, int seed, bool antithetic, int min_mcmc, int max_mcmc, int verbose, Scalar& tau_t, Scalar& disc_t)
    {
        // Sampler
        std::mt19937 gen(seed);
        RBMSampler<Scalar> sampler(m_w, m_b, m_c);

        // MCMC path
        Vector vk(m_m), vk_mean(m_m), hk(m_n), hk_mean(m_n);
        Matrix vhist, vchist, hhist, hchist;

        // # discarded samples
        disc_t = sampler.sample(gen, antithetic, m_v0.col(id), vhist, vchist, hhist, hchist,
                                min_mcmc, max_mcmc, verbose > 2);
        const int burnin = min_mcmc - 1;
        const int remain = vchist.cols() - burnin;
        // Chain length
        tau_t += vchist.cols();

        vk.noalias() = vhist.col(burnin);
        hk.noalias() = hhist.col(burnin);
        vk_mean.noalias() = m_w * hk + m_b;
        apply_sigmoid(vk_mean);
        hk_mean.noalias() = m_w.transpose() * vk + m_c;
        apply_sigmoid(hk_mean);

        Matrix vhist_mean = m_w * hhist.rightCols(remain);
        vhist_mean.colwise() += m_b;
        apply_sigmoid(vhist_mean);

        Matrix hhist_mean = m_w.transpose() * vhist.rightCols(remain);
        hhist_mean.colwise() += m_c;
        apply_sigmoid(hhist_mean);

        Matrix vchist_mean = m_w * hchist.rightCols(remain);
        vchist_mean.colwise() += m_b;
        apply_sigmoid(vchist_mean);

        Matrix hchist_mean = m_w.transpose() * vchist.rightCols(remain);
        hchist_mean.colwise() += m_c;
        apply_sigmoid(hchist_mean);

        // Compute the second term of gradient
        Vector db_t = Vector::Zero(m_m), dc_t = Vector::Zero(m_n);
        Matrix dw_t = Matrix::Zero(m_m, m_n);

        db_t.noalias() = vk + vhist.rightCols(remain).rowwise().sum() -
            vchist.rightCols(remain).rowwise().sum();
        dc_t.noalias() = hk + hhist.rightCols(remain).rowwise().sum() -
            hchist.rightCols(remain).rowwise().sum();
        dw_t.noalias() = vk * hk_mean.transpose() +
            vhist.rightCols(remain) * hhist_mean.transpose() -
            vchist.rightCols(remain) * hchist_mean.transpose();

        db_t.noalias() += vk_mean + vhist_mean.rowwise().sum() -
            vchist_mean.rowwise().sum();
        dc_t.noalias() += hk_mean + hhist_mean.rowwise().sum() -
            hchist_mean.rowwise().sum();
        dw_t.noalias() += vk_mean * hk.transpose() +
            vhist_mean * hhist.rightCols(remain).transpose() -
            vchist_mean * hchist.rightCols(remain).transpose();

        dw_t.noalias() += vk * hk.transpose() +
            vhist.rightCols(remain) * hhist.rightCols(remain).transpose() -
            vchist.rightCols(remain) * hchist.rightCols(remain).transpose();

        // Accumulate gradients
        #pragma omp critical
        {
            m_db2.noalias() += 0.5 * db_t;
            m_dc2.noalias() += 0.5 * dc_t;
            m_dw2.noalias() += dw_t / 3.0;
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


#endif  // CDTAU_RBM_H
