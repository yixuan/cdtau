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

    // First term of the gradient, computed from the data
    Vector    m_db1;
    Vector    m_dc1;
    Matrix    m_dw1;

    // Second term of the gradient, computed from MCMC
    Vector    m_db2;
    Vector    m_dc2;
    // There are two estimators for dw2, x and y
    // The final estimator is a weighted average, p*x + (1-p)*y
    // We use the following quantities to compute the optimal weight p
    Matrix    m_dw2_1;    // sum x_i
    Matrix    m_dw2_11;   // sum x_i^2
    Matrix    m_dw2_2;    // sum y_i
    Matrix    m_dw2_22;   // sum y_i^2
    Matrix    m_dw2_12;   // sum x_i*y_i

    Vector    m_db;
    Vector    m_dc;
    Matrix    m_dw;

    Matrix    m_v0;
    Matrix    m_vchains;
    Matrix    m_hchains;

public:
    RBM(int m, int n, int nchain) :
        m_m(m), m_n(n), m_nchain(nchain),
        m_b(m), m_c(n), m_w(m, n),
        m_db1(m), m_dc1(n), m_dw1(m, n), m_db2(m), m_dc2(n),
        m_dw2_1(m, n), m_dw2_11(m, n), m_dw2_2(m, n), m_dw2_22(m, n), m_dw2_12(m, n),
        m_db(m), m_dc(n), m_dw(m, n),
        m_v0(m, nchain), m_vchains(m, nchain), m_hchains(n, nchain)
    {
        random_normal(m_b.data(), m, Scalar(0), Scalar(0.1));
        random_normal(m_c.data(), n, Scalar(0), Scalar(0.1));
        random_normal(m_w.data(), m * n, Scalar(0), Scalar(0.1));

        m_db.setZero();
        m_dc.setZero();
        m_dw.setZero();
    }

    template <typename OtherScalar>
    RBM(int m, int n, int nchain,
        Eigen::Map< Eigen::Matrix<OtherScalar, Eigen::Dynamic, 1> > b0,
        Eigen::Map< Eigen::Matrix<OtherScalar, Eigen::Dynamic, 1> > c0,
        Eigen::Map< Eigen::Matrix<OtherScalar, Eigen::Dynamic, Eigen::Dynamic> > w0) :
        m_m(m), m_n(n), m_nchain(nchain),
        m_b(m), m_c(n), m_w(m, n),
        m_db1(m), m_dc1(n), m_dw1(m, n), m_db2(m), m_dc2(n),
        m_dw2_1(m, n), m_dw2_11(m, n), m_dw2_2(m, n), m_dw2_22(m, n), m_dw2_12(m, n),
        m_db(m), m_dc(n), m_dw(m, n),
        m_v0(m, nchain), m_vchains(m, nchain), m_hchains(n, nchain)
    {
        m_b.noalias() = b0.template cast<Scalar>();
        m_c.noalias() = c0.template cast<Scalar>();
        m_w.noalias() = w0.template cast<Scalar>();

        m_db.setZero();
        m_dc.setZero();
        m_dw.setZero();
    }

    // Log-likelihood value
    template <typename Derived>
    Scalar loglik(Eigen::MatrixBase<Derived>& dat, bool exact = false, int nobs = 100, int nmc = 30, int nstep = 10) const
    {
        // Get a subset of data
        const int N = dat.cols();
        nobs = std::min(nobs, N);
        Matrix subdat(m_m, nobs);
        if(nobs == N)
        {
            subdat.noalias() = dat.template cast<Scalar>();
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
        m_dw1.noalias() = (Scalar(1) / bs) * vmb * hmean.transpose();
    }

    // Zero out gradients
    void zero_grad2()
    {
        m_db2.setZero();
        m_dc2.setZero();
        m_dw2_1.setZero();
        m_dw2_11.setZero();
        m_dw2_2.setZero();
        m_dw2_22.setZero();
        m_dw2_12.setZero();
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
        m_dw2_1.noalias() = m_vchains * m_hchains.transpose();
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
        m_dw2_1.noalias() = vchains * m_hchains.transpose();
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
        Vector db_t(m_m), dc_t(m_n);
        Matrix dw_t1(m_m, m_n), dw_t2(m_m, m_n);

        db_t.noalias() = vk_mean + vhist_mean.rowwise().sum() -
            vchist_mean.rowwise().sum();
        dc_t.noalias() = hk_mean + hhist_mean.rowwise().sum() -
            hchist_mean.rowwise().sum();
        dw_t1.noalias() = vk * hk_mean.transpose() +
            vhist.rightCols(remain) * hhist_mean.transpose() -
            vchist.rightCols(remain) * hchist_mean.transpose();
        dw_t2.noalias() = vk_mean * hk.transpose() +
            vhist_mean * hhist.rightCols(remain).transpose() -
            vchist_mean * hchist.rightCols(remain).transpose();

        // Accumulate gradients
        #pragma omp critical
        {
            m_db2.noalias() += db_t;
            m_dc2.noalias() += dc_t;

            m_dw2_1.noalias() += dw_t1;
            m_dw2_11.noalias() += dw_t1.cwiseAbs2();
            m_dw2_2.noalias() += dw_t2;
            m_dw2_22.noalias() += dw_t2.cwiseAbs2();
            m_dw2_12.noalias() += dw_t1.cwiseProduct(dw_t2);
        }
    }

    void update_param(Scalar lr, Scalar momentum, int n2)
    {
        m_db.noalias() = momentum * m_db + lr * (m_db1 - m_db2 / n2);
        m_dc.noalias() = momentum * m_dc + lr * (m_dc1 - m_dc2 / n2);
        m_dw.noalias() = momentum * m_dw + lr * (m_dw1 - m_dw2_1 / n2);

        m_b.noalias() += m_db;
        m_c.noalias() += m_dc;
        m_w.noalias() += m_dw;
    }

    void update_param_ucd(Scalar lr, Scalar momentum, int n2)
    {
        m_db.noalias() = momentum * m_db + lr * (m_db1 - m_db2 / n2);
        m_dc.noalias() = momentum * m_dc + lr * (m_dc1 - m_dc2 / n2);
        // m_dw.noalias() = momentum * m_dw + lr * (m_dw1 - (Scalar(0.5) * m_dw2_1 + Scalar(0.5) * m_dw2_2) / n2);

        // mu1 = w1 / n2
        // mu2 = w2 / n2
        // s11 = (w11 - w1^2 / n2) / (n2 - 1)
        // s22 = (w22 - w2^2 / n2) / (n2 - 1)
        // s12 = (w12 - w1 * w2 / n2) / (n2 - 1)
        // p = (s22 - s12) / (s11 + s22 - 2 * s12)
        // est = p * mu1 + (1 - p) * mu2 = p * (mu1 - mu2) + mu2

        //        w22 - w12 + (w1 - w2) * w2 / n2
        // p = --------------------------------------
        //     w11 + w22 - 2 * w12 - (w1 - w2)^2 / n2

        // Matrix weight = (m_dw2_22 - m_dw2_12 + (m_dw2_1 - m_dw2_2).cwiseProduct(m_dw2_2) / n2).cwiseQuotient(
        //     m_dw2_11 + m_dw2_22 - Scalar(2) * m_dw2_12 - (m_dw2_1 - m_dw2_2).cwiseAbs2() / n2
        // );

        Matrix weight(m_m, m_n);
        const Scalar *w1 = m_dw2_1.data(), *w11 = m_dw2_11.data(), *w2 = m_dw2_2.data(),
                     *w22 = m_dw2_22.data(), *w12 = m_dw2_12.data();
        Scalar* p = weight.data();
        for(int i = 0; i < m_m * m_n; i++)
        {
            const Scalar numer = w22[i] - w12[i] + (w1[i] - w2[i]) * w2[i] / n2;
            const Scalar denom = w11[i] - w12[i] - (w1[i] - w2[i]) * w1[i] / n2 + numer;
            const Scalar denom_sign = (denom >= Scalar(0)) - (denom < Scalar(0));
            p[i] = numer / (std::abs(denom) + Scalar(1e-3)) * denom_sign;
        }

        m_dw.noalias() = momentum * m_dw + lr * (m_dw1 - (weight.cwiseProduct(m_dw2_1 - m_dw2_2) + m_dw2_2) / n2);

        // Matrix numer = m_dw2_22 - m_dw2_12 + (m_dw2_1 - m_dw2_2).cwiseProduct(m_dw2_2) / n2;
        // Matrix denom = m_dw2_11 + m_dw2_22 - m_dw2_12 + ((m_dw2_1 - m_dw2_2).cwiseProduct(m_dw2_2) - m_dw2_1.cwiseAbs2()) / n2;
        // Matrix weight = numer.cwiseQuotient(denom);

        // Rcpp::Rcout << "numer: " << numer.maxCoeff() << ", " << numer.minCoeff() << ", " << numer.cwiseAbs().minCoeff() << std::endl;
        // Rcpp::Rcout << "denom: " << denom.maxCoeff() << ", " << denom.minCoeff() << ", " << denom.cwiseAbs().minCoeff() << std::endl;
        // Rcpp::Rcout << "weight: " << weight.maxCoeff() << ", " << weight.minCoeff() << std::endl << std::endl;

        m_b.noalias() += m_db;
        m_c.noalias() += m_dc;
        m_w.noalias() += m_dw;
    }

    const Matrix& get_w() const { return m_w; }
    const Vector& get_b() const { return m_b; }
    const Vector& get_c() const { return m_c; }
};


#endif  // CDTAU_RBM_H
