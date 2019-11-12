#ifndef CDTAU_MCMC_H
#define CDTAU_MCMC_H

#include <RcppEigen.h>
#include "utils.h"
#include "utils_simd.h"
#include "rng.h"

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

    // (v1, vc0) -> (h1, hc0)
    // gen is used to generate antithetic uniform random variables
    // gen2 is used for other purposes, such as rejection
    int maxcoup_h_update(
        RNGEngine& gen, RNGEngine& gen2, bool antithetic,
        const Vector& v1, const Vector& vc0, const Vector& h1mean_v1, const Vector& hc0mean_vc0,
        Vector& h1, Vector& hc0,
        int max_try = 10, bool verbose = false
    ) const
    {
        Vector uh(m_n);

        // Sample the main chain, p(h|v1)
        random_uniform(uh, gen);
        random_bernoulli_uvar_simd(h1mean_v1, uh, h1, antithetic);

        // If v1 == vc0, also make h1 == hc0 and early exit
        if(all_equal(v1, vc0))
        {
            if(verbose)
                Rcpp::Rcout << "[ maxcoup_h_update ]: -1" << std::endl;
            hc0.noalias() = h1;
            return 0;
        }

        // Let the two chains meet with a positive probability
        // p(h) / q(h) = p(h|v1) / p(h|vc0)
        const Scalar logph1 = loglik_bernoulli_simd(h1mean_v1, h1);
        const Scalar logqh1 = loglik_bernoulli_simd(hc0mean_vc0, h1);
        std::exponential_distribution<Scalar> exp_distr(Scalar(1));
        const Scalar u = exp_distr(gen2);
        if(u >= logph1 - logqh1)
        {
            if(verbose)
                Rcpp::Rcout << "[ maxcoup_h_update ]: 0" << std::endl;
            hc0.noalias() = h1;
            return 0;
        }

        // Otherwise, sample the two chains conditional on no-meet
        // Use the same RNG to generate proposals to make the two chains correlated
        bool h1_set = false, hc0_set = false;
        for(int i = 0; i < max_try; i++)
        {
            // Common RNG
            random_uniform(uh, gen);

            // Sample h1
            if(!h1_set)
            {
                random_bernoulli_uvar_simd(h1mean_v1, uh, h1, antithetic);
                // Accept h1 with probability 1-q(h1)/p(h1)
                // <=> Exp(1) < log[p(h1)] - log[q(h1)]
                const Scalar logph1 = loglik_bernoulli_simd(h1mean_v1, h1);
                const Scalar logqh1 = loglik_bernoulli_simd(hc0mean_vc0, h1);
                const Scalar u1 = exp_distr(gen2);
                h1_set = (u1 < logph1 - logqh1);
            }

            // Sample hc0
            if(!hc0_set)
            {
                random_bernoulli_uvar_simd(hc0mean_vc0, uh, hc0, antithetic);
                // Accept hc0 with probability 1-p(hc0)/q(hc0)
                // <=> Exp(1) < log[q(hc0)] - log[p(hc0)]
                const Scalar logphc0 = loglik_bernoulli_simd(h1mean_v1, hc0);
                const Scalar logqhc0 = loglik_bernoulli_simd(hc0mean_vc0, hc0);
                const Scalar u2 = exp_distr(gen2);
                hc0_set = (u2 < logqhc0 - logphc0);
            }

            // Exit if h1 and hc0 have been set
            if(h1_set && hc0_set)
                return i;
        }

        if(verbose)
            Rcpp::Rcout << "[ maxcoup_h_update ]: max" << std::endl;
        return max_try;
    }

    // (h1, hc0) -> (v2, vc1)
    int maxcoup_v_update(
        RNGEngine& gen, RNGEngine& gen2, bool antithetic,
        const Vector& h1, const Vector& hc0, const Vector& v2mean_h1, const Vector& vc1mean_hc0,
        Vector& v2, Vector& vc1,
        int max_try = 10, bool verbose = false
    ) const
    {
        Vector uv(m_m);

        // Sample the main chain, p(v|h1)
        random_uniform(uv, gen);
        random_bernoulli_uvar_simd(v2mean_h1, uv, v2, antithetic);

        // If h1 == hc0, also make v2 == vc1 and early exit
        if(all_equal(h1, hc0))
        {
            if(verbose)
                Rcpp::Rcout << "[ maxcoup_v_update ]: -1" << std::endl;
            vc1.noalias() = v2;
            return 0;
        }

        // Let the two chains meet with a positive probability
        // p(h) / q(h) = p(v|h1) / p(v|hc0)
        const Scalar logpv2 = loglik_bernoulli_simd(v2mean_h1, v2);
        const Scalar logqv2 = loglik_bernoulli_simd(vc1mean_hc0, v2);
        std::exponential_distribution<Scalar> exp_distr(1.0);
        const Scalar u = exp_distr(gen2);
        if(u >= logpv2 - logqv2)
        {
            if(verbose)
                Rcpp::Rcout << "[ maxcoup_v_update ]: 0" << std::endl;
            vc1.noalias() = v2;
            return 0;
        }

        // Otherwise, sample the two chains conditional on no-meet
        // Use the same RNG to generate proposals to make the two chains correlated
        bool v2_set = false, vc1_set = false;
        for(int i = 0; i < max_try; i++)
        {
            // Common RNG
            random_uniform(uv, gen);

            // Sample v2
            if(!v2_set)
            {
                random_bernoulli_uvar_simd(v2mean_h1, uv, v2, antithetic);
                // Accept v2 with probability 1-q(v2)/p(v2)
                // <=> Exp(1) < log[p(v2)] - log[q(v2)]
                const Scalar logpv2 = loglik_bernoulli_simd(v2mean_h1, v2);
                const Scalar logqv2 = loglik_bernoulli_simd(vc1mean_hc0, v2);
                const Scalar u1 = exp_distr(gen2);
                v2_set = (u1 < logpv2 - logqv2);
            }

            // Sample vc1
            if(!vc1_set)
            {
                random_bernoulli_uvar_simd(vc1mean_hc0, uv, vc1, antithetic);
                // Accept vc1 with probability 1-p(vc1)/q(vc1)
                // <=> Exp(1) < log[q(vc1)] - log[p(vc1)]
                const Scalar logpvc1 = loglik_bernoulli_simd(v2mean_h1, vc1);
                const Scalar logqvc1 = loglik_bernoulli_simd(vc1mean_hc0, vc1);
                const Scalar u2 = exp_distr(gen2);
                vc1_set = (u2 < logqvc1 - logpvc1);
            }

            // Exit if v2 and vc1 have been set
            if(v2_set && vc1_set)
                return i;
        }

        if(verbose)
            Rcpp::Rcout << "[ maxcoup_v_update ]: max" << std::endl;
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
    void sample_k(RNGEngine& gen, RefConstVec& v0, Vector& v, Vector& h, int k = 10) const
    {
        v.resize(m_m);
        h.resize(m_n);

        Vector hmean = m_w.transpose() * v0 + m_c;
        apply_sigmoid_simd(hmean);
        random_bernoulli(hmean, h, gen);

        Vector vmean(m_m);
        for(int i = 0; i < k; i++)
        {
            vmean.noalias() = m_w * h + m_b;
            apply_sigmoid_simd(vmean);
            random_bernoulli(vmean, v, gen);

            hmean.noalias() = m_w.transpose() * v + m_c;
            apply_sigmoid_simd(hmean);
            random_bernoulli(hmean, h, gen);
        }
    }

    // Sample k steps on multiple chains, using multiple initial values
    void sample_k_mc(RNGEngine& gen, const Matrix& v0, Matrix& v, Matrix& h, int k = 10) const
    {
        const int nchain = v0.cols();
        v.resize(m_m, nchain);
        h.resize(m_n, nchain);

        h.noalias() = m_w.transpose() * v0;
        h.colwise() += m_c;
        apply_sigmoid_simd(h);
        random_bernoulli(h, h, gen);

        for(int i = 0; i < k; i++)
        {
            v.noalias() = m_w * h;
            v.colwise() += m_b;
            apply_sigmoid_simd(v);
            random_bernoulli(v, v, gen);

            h.noalias() = m_w.transpose() * v;
            h.colwise() += m_c;
            apply_sigmoid_simd(h);
            random_bernoulli(h, h, gen);
        }
    }

    // Sample k steps on multiple chains, using the same initial vector
    void sample_k_mc(RNGEngine& gen, const Vector& v0, Matrix& v, Matrix& h, int k = 10, int nchain = 10) const
    {
        v.resize(m_m, nchain);
        h.resize(m_n, nchain);

        h.noalias() = m_w.transpose() * v0.replicate(1, nchain);
        h.colwise() += m_c;
        apply_sigmoid_simd(h);
        random_bernoulli(h, h, gen);

        for(int i = 0; i < k; i++)
        {
            v.noalias() = m_w * h;
            v.colwise() += m_b;
            apply_sigmoid_simd(v);
            random_bernoulli(v, v, gen);

            h.noalias() = m_w.transpose() * v;
            h.colwise() += m_c;
            apply_sigmoid_simd(h);
            random_bernoulli(h, h, gen);
        }
    }

    // Sample k steps on multiple chains, using the same initial vector
    // Generate antithetic pairs
    void sample_k_mc_pair(RNGEngine& gen, const Vector& v0,
                          Matrix& v, Matrix& h, int k = 10, int npair = 5) const
    {
        const int nchain = npair * 2;
        v.resize(m_m, nchain);
        h.resize(m_n, nchain);

        // Mean of h
        h.noalias() = m_w.transpose() * v0.replicate(1, nchain);
        h.colwise() += m_c;
        apply_sigmoid_simd(h);

        // Common RNG
        Matrix uv(m_m, npair), uh(m_n, npair);
        random_uniform(uh, gen);

        // Antithetic h
        random_bernoulli_uvar_antithetic(h, uh, h);

        for(int i = 0; i < k; i++)
        {
            // Antithetic v
            random_uniform(uv, gen);

            v.noalias() = m_w * h;
            v.colwise() += m_b;
            apply_sigmoid_simd(v);
            random_bernoulli_uvar_antithetic(v, uv, v);

            // Antithetic h
            random_uniform(uh, gen);

            h.noalias() = m_w.transpose() * v;
            h.colwise() += m_c;
            apply_sigmoid_simd(h);
            random_bernoulli_uvar_antithetic(h, uh, h);
        }
    }

    // Unbiased sampling
    int sample(
        RNGEngine& gen, bool antithetic, RefConstVec& v0,
        Matrix& vhist, Matrix& vchist, Matrix& hhist, Matrix& hchist,
        int min_steps = 1, int max_steps = 100, bool verbose = false
    ) const
    {
        // We want to create a pair of antithetic chains in two sample() calls
        // To do this, we need to make sure that the uniform random vector sequences
        // are the same in the two calls
        // 1. We create a second RNG for other purposes, such as generating
        //    exponential random variables.
        // 2. Since there are rejection steps in maxcoup_h_update() and maxcoup_v_update(),
        //    the number of uniform random variables used in each iteration is not
        //    deterministic. Therefore, we reset the random seeds in each iteration.

        // gen() gives a random integer, used as the seed for gen2
        RNGEngine gen2(gen());

        // Seeds for gen in each iteration
        typedef RNGEngine::result_type SeedType;
        std::vector<SeedType> seeds(2 * max_steps);
        for(int i = 0; i < 2 * max_steps; i++)
            seeds[i] = gen2();

        Vector v(m_m), h(m_n), vc(m_m), hc(m_n);
        Vector vmean_h(m_m), hmean_v(m_n), vcmean_hc(m_m), hcmean_vc(m_n);
        Vector uv(m_m), uh(m_n);

        hc.setZero();
        vc.noalias() = v0;                                         // vc0 = v0

        rbm_op_h(m_w, vc, m_c, hcmean_vc);
        apply_sigmoid_simd(hcmean_vc);                             // E(h|v)
        random_uniform(uh, gen);
        random_bernoulli_uvar_simd(hcmean_vc, uh, h, antithetic);  // h1

        rbm_op_v(m_w, h, m_b, vmean_h);
        apply_sigmoid_simd(vmean_h);                               // E(v|h)
        random_uniform(uv, gen);
        random_bernoulli_uvar_simd(vmean_h, uv, v, antithetic);    // v1

        // E(h|v)
        rbm_op_h(m_w, v, m_c, hmean_v);
        apply_sigmoid_simd(hmean_v);

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

            gen = RNGEngine(seeds[i]);
            discard += maxcoup_h_update(gen, gen2, antithetic, v, vc, hmean_v, hcmean_vc,
                                        h, hc, 10, verbose);

            if(i >= min_steps && all_equal(h, hc))
            {
                if(verbose)
                    Rcpp::Rcout << "===== Exit with meeting time = " << i + 1 << " =====" << std::endl;
                break;
            }

            // E(v|h)
            rbm_op_v(m_w, h, m_b, vmean_h);
            apply_sigmoid_simd(vmean_h);
            rbm_op_v(m_w, hc, m_b, vcmean_hc);
            apply_sigmoid_simd(vcmean_hc);

            gen = RNGEngine(seeds[max_steps + i]);
            discard += maxcoup_v_update(gen, gen2, antithetic, h, hc, vmean_h, vcmean_hc,
                                        v, vc, 10, verbose);

            // E(h|v)
            rbm_op_h(m_w, v, m_c, hmean_v);
            apply_sigmoid_simd(hmean_v);
            rbm_op_h(m_w, vc, m_c, hcmean_vc);
            apply_sigmoid_simd(hcmean_vc);

            vs.push_back(v);
            vcs.push_back(vc);
            hs.push_back(h);
            hcs.push_back(hc);

            if(i >= min_steps - 1 && all_equal(v, vc))
            {
                if(verbose)
                    Rcpp::Rcout << "===== Exit with meeting time = " << i + 1 << " =====" << std::endl;
                break;
            }
        }

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



    // Unbiased sampling
    void sample_and_update_grad(
        RNGEngine& gen, bool antithetic, RefConstVec& v0,
        Vector& db, Vector& dc, Matrix& dw, Scalar& tau, Scalar& discard,
        int min_steps = 1, int max_steps = 100, bool verbose = false
    ) const
    {
        // We want to create a pair of antithetic chains in two sample() calls
        // To do this, we need to make sure that the uniform random vector sequences
        // are the same in the two calls
        // 1. We create a second RNG for other purposes, such as generating
        //    exponential random variables.
        // 2. Since there are rejection steps in maxcoup_h_update() and maxcoup_v_update(),
        //    the number of uniform random variables used in each iteration is not
        //    deterministic. Therefore, we reset the random seeds in each iteration.

        // gen() gives a random integer, used as the seed for gen2
        RNGEngine gen2(gen());

        // Seeds for gen in each iteration
        typedef RNGEngine::result_type SeedType;
        std::vector<SeedType> seeds(2 * max_steps);
        for(int i = 0; i < 2 * max_steps; i++)
            seeds[i] = gen2();

        Vector v(m_m), h(m_n), vc(m_m), hc(m_n);
        Vector vmean_h(m_m), hmean_v(m_n), vcmean_hc(m_m), hcmean_vc(m_n);
        Vector uv(m_m), uh(m_n);

        vc.noalias() = v0;                                         // vc0 = v0

        rbm_op_h(m_w, vc, m_c, hcmean_vc);
        apply_sigmoid_simd(hcmean_vc);                             // E(h|v)
        random_uniform(uh, gen);
        random_bernoulli_uvar_simd(hcmean_vc, uh, h, antithetic);  // h1

        rbm_op_v(m_w, h, m_b, vmean_h);
        apply_sigmoid_simd(vmean_h);                               // E(v|h)
        random_uniform(uv, gen);
        random_bernoulli_uvar_simd(vmean_h, uv, v, antithetic);    // v1

        // E(h|v)
        rbm_op_h(m_w, v, m_c, hmean_v);
        apply_sigmoid_simd(hmean_v);

        const int burnin = min_steps - 1;
        int i = 0;
        discard = 0;

        for(i = 0; i < max_steps; i++)
        {
            if(verbose)
                Rcpp::Rcout << "===== Gibbs iteration " << i << " =====" << std::endl;

            if(i == burnin)
            {
                db.noalias() = v + vmean_h;
                dc.noalias() = h + hmean_v;
                // dw.noalias() = v * hmean_v.transpose() + vmean_h * h.transpose();
                rbm_op_rank2(v, hmean_v, vmean_h, h, dw);
            }

            gen = RNGEngine(seeds[i]);
            discard += maxcoup_h_update(gen, gen2, antithetic, v, vc, hmean_v, hcmean_vc,
                                        h, hc, 10, verbose);

            // If h == hc, v will be equal to vc in this iteration, making f(xi) - f(eta) = 0
            // Hence we can directly exit the loop
            if(i >= burnin && all_equal(h, hc))
            {
                if(verbose)
                    Rcpp::Rcout << "===== Exit with meeting time = " << i + 1 << " =====" << std::endl;
                break;
            }

            // E(v|h)
            rbm_op_v(m_w, h, m_b, vmean_h);
            apply_sigmoid_simd(vmean_h);
            rbm_op_v(m_w, hc, m_b, vcmean_hc);
            apply_sigmoid_simd(vcmean_hc);

            gen = RNGEngine(seeds[max_steps + i]);
            discard += maxcoup_v_update(gen, gen2, antithetic, h, hc, vmean_h, vcmean_hc,
                                        v, vc, 10, verbose);

            // E(h|v)
            rbm_op_h(m_w, v, m_c, hmean_v);
            apply_sigmoid_simd(hmean_v);
            rbm_op_h(m_w, vc, m_c, hcmean_vc);
            apply_sigmoid_simd(hcmean_vc);

            if(i >= burnin)
            {
                db.noalias() += (v + vmean_h - vc - vcmean_hc);
                dc.noalias() += (h + hmean_v - hc - hcmean_vc);
                /* dw.noalias() += (
                    v * hmean_v.transpose() + vmean_h * h.transpose() -
                    vc * hcmean_vc.transpose() - vcmean_hc * hc.transpose()
                ); */
                rbm_op_rank4_simd(v, hmean_v, vmean_h, h, vc, hcmean_vc, vcmean_hc, hc, dw);
            }

            if(i >= burnin && all_equal(v, vc))
            {
                if(verbose)
                    Rcpp::Rcout << "===== Exit with meeting time = " << i + 1 << " =====" << std::endl;
                break;
            }

        }

        tau = i + 1;
        db *= Scalar(0.5);
        dc *= Scalar(0.5);
        dw *= Scalar(0.5);
    }
};


#endif  // CDTAU_MCMC_H
