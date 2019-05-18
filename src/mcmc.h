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

    void sample(
        const Vector& v0, Matrix& vhist, Matrix& vchist,
        int min_steps = 1, int max_steps = 100, bool verbose = false
    ) const
    {
        Vector vmean(m_m), v(m_m), vc(m_m), hmean(m_n), h(m_n), hc(m_n);
        hmean.noalias() = m_w.transpose() * v0 + m_c;
        apply_sigmoid(hmean);
        random_bernoulli(hmean, h);
        random_bernoulli(hmean, hc);

        vmean.noalias() = m_w * h + m_b;
        apply_sigmoid(vmean);
        random_bernoulli(vmean, v);
        hmean.noalias() = m_w.transpose() * v + m_c;
        apply_sigmoid(hmean);
        random_bernoulli(hmean, h);

        std::vector<Vector> vs;
        vs.push_back(v);
        std::vector<Vector> vcs;

        for(int i = 0; i < max_steps; i++)
        {
            if(verbose)
                Rcpp::Rcout << "===== Gibbs iteration " << i << " =====" << std::endl;

            maxcoup_v(h, hc, v, vc, 10, verbose);
            maxcoup_h(v, vc, h, hc, 10, verbose);

            vs.push_back(v);
            vcs.push_back(vc);

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
