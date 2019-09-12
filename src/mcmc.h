#ifndef CDTAU_MCMC_H
#define CDTAU_MCMC_H

#include <RcppEigen.h>
#include "utils.h"
#include "utils_simd.h"

template <typename Scalar = double>
class RBMSampler
{
private:
    typedef Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic> Matrix;
    typedef Eigen::Matrix<Scalar, Eigen::Dynamic, 1> Vector;
    typedef const Eigen::Ref<const Matrix> RefConstMat;
    typedef const Eigen::Ref<const Vector> RefConstVec;
    typedef Eigen::Ref<Vector> RefVec;

    const int   m_m;
    const int   m_n;
    RefConstVec m_b;
    RefConstVec m_c;
    RefConstMat m_w;

    // (xi1, eta0) -> (xi2, eta1) -> ...
    // xi = (v, h), eta = (vc, hc)
    int maxcoup(
        std::mt19937& gen,
        const Vector& v1, const Vector& h1, const Vector& vc0, const Vector& hc0,
        Vector& v2, Vector& h2, Vector& vc1, Vector& hc1,
        int max_try = 10, bool verbose = false
    ) const
    {
        // Sample the xi chain
        // p1(v | h1)
        Vector v2mean = m_w * h1 + m_b;
        apply_sigmoid(v2mean);
        Vector uv(m_m);
        random_uniform(uv, gen);
        random_bernoulli_uvar(v2mean, uv, v2);
        // p2(h | v)
        Vector h2mean = m_w.transpose() * v2 + m_c;
        apply_sigmoid(h2mean);
        Vector uh(m_n);
        random_uniform(uh, gen);
        random_bernoulli_uvar(h2mean, uh, h2);

        // If xi1 == eta0, also make xi2 == eta1 and early exit
        if(all_equal(v1, vc0) && all_equal(h1, hc0))
        {
            if(verbose)
                Rcpp::Rcout << "[ maxcoup ]: -1" << std::endl;
            vc1.noalias() = v2;
            hc1.noalias() = h2;
            return 0;
        }

        // Let the two chains meet with a positive probability
        // p((v, h) | xi1) = p1(v | h1) * p2(h | v)
        // p((v, h) | eta0) = q1(v | hc0) * q2(h | v)
        // p2 = q2, so p((v, h) | xi1) / p((v, h) | eta0) = p1(v | h1) / q1(v | hc0)
        Vector vc1mean = m_w * hc0 + m_b;
        apply_sigmoid(vc1mean);
        Scalar logpxi1 = loglik_bernoulli_simd(v2mean, v2);
        Scalar logpeta0 = loglik_bernoulli_simd(vc1mean, v2);
        std::exponential_distribution<Scalar> exp_distr(1.0);
        Scalar u = exp_distr(gen);
        if(u >= logpxi1 - logpeta0)
        {
            if(verbose)
                Rcpp::Rcout << "[ maxcoup ]: 0" << std::endl;
            vc1.noalias() = v2;
            hc1.noalias() = h2;
            return 0;
        }

        // Otherwise, sample the second chain
        for(int i = 0; i < max_try; i++)
        {
            if(i == 0)
                random_bernoulli_uvar(vc1mean, uv, vc1);
            else
                random_bernoulli(vc1mean, vc1, gen);
            logpxi1 = loglik_bernoulli_simd(v2mean, vc1);
            logpeta0 = loglik_bernoulli_simd(vc1mean, vc1);
            u = exp_distr(gen);
            if(u < logpeta0 - logpxi1)
            {
                if(verbose)
                    Rcpp::Rcout << "[ maxcoup ]: " << i + 1 << std::endl;

                Vector hc1mean = m_w.transpose() * vc1 + m_c;
                apply_sigmoid(hc1mean);
                random_bernoulli_uvar(hc1mean, uh, hc1);
                return i;
            }
        }
        Vector hc1mean = m_w.transpose() * vc1 + m_c;
        apply_sigmoid(hc1mean);
        random_bernoulli_uvar(hc1mean, uh, hc1);

        if(verbose)
            Rcpp::Rcout << "[ maxcoup ]: max" << std::endl;
        return max_try;
    }

public:
    RBMSampler(RefConstMat& w, RefConstVec& b, RefConstVec& c) :
        m_m(w.rows()), m_n(w.cols()), m_b(b), m_c(c), m_w(w)
    {
        if(b.size() != m_m || c.size() != m_n)
            throw std::invalid_argument("dimensions do not match");
    }

    // Sample k steps
    void sample_k(std::mt19937& gen, RefConstVec& v0, Vector& v, Vector& h, int k = 10) const
    {
        v.resize(m_m);
        h.resize(m_n);

        Vector hmean = m_w.transpose() * v0 + m_c;
        apply_sigmoid(hmean);
        random_bernoulli(hmean, h, gen);

        Vector vmean(m_m);
        for(int i = 0; i < k; i++)
        {
            vmean.noalias() = m_w * h + m_b;
            apply_sigmoid(vmean);
            random_bernoulli(vmean, v, gen);

            hmean.noalias() = m_w.transpose() * v + m_c;
            apply_sigmoid(hmean);
            random_bernoulli(hmean, h, gen);
        }
    }

    // Sample k steps on multiple chains, using multiple initial values
    void sample_k_mc(std::mt19937& gen, const Matrix& v0, Matrix& v, Matrix& h, int k = 10) const
    {
        const int nchain = v0.cols();
        v.resize(m_m, nchain);
        h.resize(m_n, nchain);

        h.noalias() = m_w.transpose() * v0;
        h.colwise() += m_c;
        apply_sigmoid(h);
        random_bernoulli(h, h, gen);

        for(int i = 0; i < k; i++)
        {
            v.noalias() = m_w * h;
            v.colwise() += m_b;
            apply_sigmoid(v);
            random_bernoulli(v, v, gen);

            h.noalias() = m_w.transpose() * v;
            h.colwise() += m_c;
            apply_sigmoid(h);
            random_bernoulli(h, h, gen);
        }
    }

    // Sample k steps on multiple chains, using the same initial vector
    void sample_k_mc(std::mt19937& gen, const Vector& v0, Matrix& v, Matrix& h, int k = 10, int nchain = 10) const
    {
        v.resize(m_m, nchain);
        h.resize(m_n, nchain);

        h.noalias() = m_w.transpose() * v0.replicate(1, nchain);
        h.colwise() += m_c;
        apply_sigmoid(h);
        random_bernoulli(h, h, gen);

        for(int i = 0; i < k; i++)
        {
            v.noalias() = m_w * h;
            v.colwise() += m_b;
            apply_sigmoid(v);
            random_bernoulli(v, v, gen);

            h.noalias() = m_w.transpose() * v;
            h.colwise() += m_c;
            apply_sigmoid(h);
            random_bernoulli(h, h, gen);
        }
    }

    // Unbiased sampling
    // vc0, hc0, v1, and h1 will be updated
    int sample(
        std::mt19937& gen,
        RefVec vc0, RefVec hc0, RefVec v1, RefVec h1,
        Matrix& vhist, Matrix& vchist, Matrix& hhist, Matrix& hchist,
        int min_steps = 1, int max_steps = 100, bool verbose = false
    ) const
    {
        // (v0, h0)   -> (v1, h1)   -> (v2, h2)   -> ... -> (vt, ht)
        //               (vc0, hc0) -> (vc1, hc1) -> ... -> (vct, hct)
        // Init: (v0, h0) = (vc0, hc0)
        // Iter: (v1, h1, vc0, hc0) -> (v2, h2, vc1, hc1) -> ...
        // Stop: (vt, ht) = (vct, hct)
        Vector v(m_m), h(m_n), vc(m_m), hc(m_n);
        Vector v_next(m_m), h_next(m_n), vc_next(m_m), hc_next(m_n);

        vc.noalias() = vc0;
        hc.noalias() = hc0;
        v.noalias() = v1;
        h.noalias() = h1;

        std::vector<Vector> vs;
        vs.push_back(v);
        std::vector<Vector> vcs;

        std::vector<Vector> hs;
        hs.push_back(h);
        std::vector<Vector> hcs;

        int discard = 0;

        for(int i = 0; i < max_steps; i++)
        {
            if(verbose)
                Rcpp::Rcout << "===== Gibbs iteration " << i << " =====" << std::endl;

            discard += maxcoup(gen, v, h, vc, hc, v_next, h_next, vc_next, hc_next, 10, verbose);

            vs.push_back(v_next);
            vcs.push_back(vc_next);
            hs.push_back(h_next);
            hcs.push_back(hc_next);

            if((i >= min_steps - 1) &&
               (all_equal(v_next, vc_next)) &&
               (all_equal(h_next, hc_next)))
            {
                if(verbose)
                    Rcpp::Rcout << "===== Exit with meeting time = " << i + 1 << " =====" << std::endl;
                break;
            }

            v.swap(v_next);
            h.swap(h_next);
            vc.swap(vc_next);
            hc.swap(hc_next);
        }

        // Reset initial values
        vc0.noalias() = vc;
        hc0.noalias() = hc;
        v1.noalias() = v;
        h1.noalias() = h;

        const int tau = vcs.size();
        vhist.resize(m_m, tau + 1);
        vchist.resize(m_m, tau);
        hhist.resize(m_n, tau + 1);
        hchist.resize(m_n, tau);
        for(int i = 0; i < tau; i++)
        {
            vhist.col(i).noalias() = vs[i];
            vchist.col(i).noalias() = vcs[i];
            hhist.col(i).noalias() = hs[i];
            hchist.col(i).noalias() = hcs[i];
        }
        vhist.col(tau).noalias() = vs[tau];
        hhist.col(tau).noalias() = hs[tau];

        return discard;
    }

    int sample(
        std::mt19937& gen,
        RefConstVec& v0, Matrix& vhist, Matrix& vchist, Matrix& hhist, Matrix& hchist,
        int min_steps = 1, int max_steps = 100, bool verbose = false
    ) const
    {
        // (v0, h0)   -> (v1, h1)   -> (v2, h2)   -> ... -> (vt, ht)
        //               (vc0, hc0) -> (vc1, hc1) -> ... -> (vct, hct)
        // Init: (v0, h0) = (vc0, hc0)
        // Iter: (v1, h1, vc0, hc0) -> (v2, h2, vc1, hc1) -> ...
        // Stop: (vt, ht) = (vct, hct)
        Vector v(m_m), h(m_n), vc(m_m), hc(m_n);

        vc.noalias() = v0;              // vc0 = v0
        hc.noalias() = m_w.transpose() * vc + m_c;
        apply_sigmoid(hc);
        random_bernoulli(hc, hc, gen);  // hc0 = h0

        v.noalias() = m_w * hc + m_b;
        apply_sigmoid(v);
        random_bernoulli(v, v, gen);    // v1
        h.noalias() = m_w.transpose() * v + m_c;
        apply_sigmoid(h);
        random_bernoulli(h, h, gen);    // h1

        return sample(gen, vc, hc, v, h, vhist, vchist, hhist, hchist, min_steps, max_steps, verbose);
    }
};


#endif  // CDTAU_MCMC_H
