#ifndef CDTAU_MCMC_H
#define CDTAU_MCMC_H

#include <RcppEigen.h>
#include "utils.h"

class RBMSampler
{
private:
    typedef Eigen::MatrixXd Matrix;
    typedef Eigen::VectorXd Vector;
    typedef const Eigen::Ref<const Matrix> RefConstMat;
    typedef const Eigen::Ref<const Vector> RefConstVec;

    const int   m_m;
    const int   m_n;
    RefConstVec m_b;
    RefConstVec m_c;
    RefConstMat m_w;

    // (xi1, eta0) -> (xi2, eta1) -> ...
    // xi = (v, h), eta = (vc, hc)
    void maxcoup(
        const Vector& v1, const Vector& h1, const Vector& vc0, const Vector& hc0,
        Vector& v2, Vector& h2, Vector& vc1, Vector& hc1,
        int max_try = 10, bool verbose = false
    ) const
    {
        // Sample the xi chain
        // p1(v | h1)
        Vector v2mean = m_w * h1 + m_b;
        apply_sigmoid(v2mean);
        random_bernoulli(v2mean, v2);
        // p2(h | v)
        Vector h2mean = m_w.transpose() * v2 + m_c;
        apply_sigmoid(h2mean);
        Vector uvar(m_n);
        random_uniform(uvar);
        random_bernoulli_uvar(h2mean, uvar, h2);

        // If xi1 == eta0, also make xi2 == eta1 and early exit
        if(all_equal(v1, vc0) && all_equal(h1, hc0))
        {
            if(verbose)
                Rcpp::Rcout << "[ maxcoup ]: -1" << std::endl;
            vc1.noalias() = v2;
            hc1.noalias() = h2;
            return;
        }

        // Let the two chains meet with a positive probability
        // p((v, h) | xi1) = p1(v | h1) * p2(h | v)
        // p((v, h) | eta0) = q1(v | hc0) * q2(h | v)
        // p2 = q2, so p((v, h) | xi1) / p((v, h) | eta0) = p1(v | h1) / q1(v | hc0)
        Vector vc1mean = m_w * hc0 + m_b;
        apply_sigmoid(vc1mean);
        double logpxi1 = loglik_bernoulli(v2mean, v2);
        double logpeta0 = loglik_bernoulli(vc1mean, v2);
        double u = R::exp_rand();
        if(u >= logpxi1 - logpeta0)
        {
            if(verbose)
                Rcpp::Rcout << "[ maxcoup ]: 0" << std::endl;
            vc1.noalias() = v2;
            hc1.noalias() = h2;
            return;
        }

        // Otherwise, sample the second chain
        for(int i = 0; i < max_try; i++)
        {
            random_bernoulli(vc1mean, vc1);
            logpxi1 = loglik_bernoulli(v2mean, vc1);
            logpeta0 = loglik_bernoulli(vc1mean, vc1);
            u = R::exp_rand();
            if(u < logpeta0 - logpxi1)
            {
                if(verbose)
                    Rcpp::Rcout << "[ maxcoup ]: " << i + 1 << std::endl;

                Vector hc1mean = m_w.transpose() * vc1 + m_c;
                apply_sigmoid(hc1mean);
                random_bernoulli_uvar(hc1mean, uvar, hc1);
                return;
            }
        }
        Vector hc1mean = m_w.transpose() * vc1 + m_c;
        apply_sigmoid(hc1mean);
        random_bernoulli_uvar(hc1mean, uvar, hc1);

        if(verbose)
            Rcpp::Rcout << "[ maxcoup ]: max" << std::endl;
    }

    // Sample couplings of v given h
    // p1 = Bernoulli(sigmoid(w * h1 + b))
    // p2 = Bernoulli(sigmoid(w * h2 + b))
    void maxcoup_v(
        const Vector& h1, const Vector& h2, Vector& v1, Vector& v2,
        int max_try = 10, bool verbose = false
    ) const
    {
        // Sample the first chain
        Vector v1mean = m_w * h1 + m_b;
        apply_sigmoid(v1mean);
        random_bernoulli(v1mean, v1);

        // If h1 == h2, also make v1 == v2 and early exit
        if(all_equal(h1, h2))
        {
            if(verbose)
                Rcpp::Rcout << "[ maxcoup_v ]: -1" << std::endl;
            v2.noalias() = v1;
            return;
        }

        // Let the two chains meet with a positive probability
        Vector v2mean = m_w * h2 + m_b;
        apply_sigmoid(v2mean);
        double logp1 = loglik_bernoulli(v1mean, v1);
        double logp2 = loglik_bernoulli(v2mean, v1);
        double u = R::exp_rand();
        if(u >= logp1 - logp2)
        {
            if(verbose)
                Rcpp::Rcout << "[ maxcoup_v ]: 0" << std::endl;
            v2.noalias() = v1;
            return;
        }

        // Otherwise, sample the second chain
        for(int i = 0; i < max_try; i++)
        {
            random_bernoulli(v2mean, v2);
            logp1 = loglik_bernoulli(v1mean, v2);
            logp2 = loglik_bernoulli(v2mean, v2);
            u = R::exp_rand();
            if(u < logp2 - logp1)
            {
                if(verbose)
                    Rcpp::Rcout << "[ maxcoup_v ]: " << i + 1 << std::endl;
                return;
            }
        }

        if(verbose)
            Rcpp::Rcout << "[ maxcoup_v ]: max" << std::endl;
    }

    // Sample couplings of h given v
    // p1 = Bernoulli(sigmoid(w' * v1 + c))
    // p2 = Bernoulli(sigmoid(w' * v2 + c))
    void maxcoup_h(
        const Vector& v1, const Vector& v2, Vector& h1, Vector& h2,
        int max_try = 10, bool verbose = false
    ) const
    {
        // Sample the first chain
        Vector h1mean = m_w.transpose() * v1 + m_c;
        apply_sigmoid(h1mean);
        random_bernoulli(h1mean, h1);

        // If v1 == v2, also make h1 == h2 and early exit
        if(all_equal(v1, v2))
        {
            if(verbose)
                Rcpp::Rcout << "[ maxcoup_h ]: -1" << std::endl;
            h2.noalias() = h1;
            return;
        }

        // Let the two chains meet with a positive probability
        Vector h2mean = m_w.transpose() * v2 + m_c;
        apply_sigmoid(h2mean);
        double logp1 = loglik_bernoulli(h1mean, h1);
        double logp2 = loglik_bernoulli(h2mean, h1);
        double u = R::exp_rand();
        if(u >= logp1 - logp2)
        {
            if(verbose)
                Rcpp::Rcout << "[ maxcoup_h ]: 0" << std::endl;
            h2.noalias() = h1;
            return;
        }

        // Otherwise, sample the second chain
        for(int i = 0; i < max_try; i++)
        {
            random_bernoulli(h2mean, h2);
            logp1 = loglik_bernoulli(h1mean, h2);
            logp2 = loglik_bernoulli(h2mean, h2);
            u = R::exp_rand();
            if(u < logp2 - logp1)
            {
                if(verbose)
                    Rcpp::Rcout << "[ maxcoup_h ]: " << i + 1 << std::endl;
                return;
            }
        }

        if(verbose)
            Rcpp::Rcout << "[ maxcoup_h ]: max" << std::endl;
    }

public:
    RBMSampler(RefConstMat& w, RefConstVec& b, RefConstVec& c) :
        m_m(w.rows()), m_n(w.cols()), m_b(b), m_c(c), m_w(w)
    {
        if(b.size() != m_m || c.size() != m_n)
            throw std::invalid_argument("dimensions do not match");
    }

    // Sample k steps
    void sample_k(const Vector& v0, Vector& v, Vector& h, int k = 10) const
    {
        v.resize(m_m);
        h.resize(m_n);

        Vector hmean = m_w.transpose() * v0 + m_c;
        apply_sigmoid(hmean);
        random_bernoulli(hmean, h);

        Vector vmean(m_m);
        for(int i = 0; i < k; i++)
        {
            vmean.noalias() = m_w * h + m_b;
            apply_sigmoid(vmean);
            random_bernoulli(vmean, v);

            hmean.noalias() = m_w.transpose() * v + m_c;
            apply_sigmoid(hmean);
            random_bernoulli(hmean, h);
        }
    }

    // Sample k steps on multiple chains
    void sample_k_mc(const Vector& v0, Matrix& v, Matrix& h, int k = 10, int nchain = 10) const
    {
        v.resize(m_m, nchain);
        h.resize(m_n, nchain);

        h.noalias() = m_w.transpose() * v0.replicate(1, nchain);
        h.colwise() += m_c;
        apply_sigmoid(h);
        random_bernoulli(h, h);

        for(int i = 0; i < k; i++)
        {
            v.noalias() = m_w * h;
            v.colwise() += m_b;
            apply_sigmoid(v);
            random_bernoulli(v, v);

            h.noalias() = m_w.transpose() * v;
            h.colwise() += m_c;
            apply_sigmoid(h);
            random_bernoulli(h, h);
        }
    }

    // Unbiased sampling
    void sample(
        const Vector& v0, Matrix& vhist, Matrix& vchist,
        int min_steps = 1, int max_steps = 100, bool verbose = false
    ) const
    {
        Vector vmean(m_m), hmean(m_n);
        Vector v(m_m), h(m_n), vc(m_m), hc(m_n);
        Vector v_next(m_m), h_next(m_n), vc_next(m_m), hc_next(m_n);

        hmean.noalias() = m_w.transpose() * v0 + m_c;
        apply_sigmoid(hmean);
        random_bernoulli(hmean, h);  // h0
        vc.noalias() = v0;           // vc0
        hc.noalias() = h;            // hc0
        // random_bernoulli(hmean, hc);

        vmean.noalias() = m_w * h + m_b;
        apply_sigmoid(vmean);
        random_bernoulli(vmean, v);  // v1
        hmean.noalias() = m_w.transpose() * v + m_c;
        apply_sigmoid(hmean);
        random_bernoulli(hmean, h);  // h1

        std::vector<Vector> vs;
        vs.push_back(v);
        std::vector<Vector> vcs;

        for(int i = 0; i < max_steps; i++)
        {
            if(verbose)
                Rcpp::Rcout << "===== Gibbs iteration " << i << " =====" << std::endl;

            maxcoup(v, h, vc, hc, v_next, h_next, vc_next, hc_next, 10, verbose);

            vs.push_back(v_next);
            vcs.push_back(vc_next);

            v.swap(v_next);
            h.swap(h_next);
            vc.swap(vc_next);
            hc.swap(hc_next);

            if((i >= min_steps - 1) &&
               (all_equal(v, vc)) &&
               (all_equal(h, hc)))
            {
                if(verbose)
                    Rcpp::Rcout << "===== Exit with meeting time = " << i + 1 << " =====" << std::endl;
                break;
            }
        }

        const int tau = vcs.size();
        vhist.resize(m_m, tau + 1);
        vchist.resize(m_m, tau);
        for(int i = 0; i < tau; i++)
        {
            vhist.col(i).noalias() = vs[i];
            vchist.col(i).noalias() = vcs[i];
        }
        vhist.col(tau).noalias() = vs[tau];
    }
};


#endif  // CDTAU_MCMC_H
