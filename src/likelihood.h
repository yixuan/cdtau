#ifndef CDTAU_LIKELIHOOD_H
#define CDTAU_LIKELIHOOD_H

#include <RcppEigen.h>
#include "mcmc.h"
#include "utils.h"
#include "utils_simd.h"
#include "rng.h"

// ============================= OpenBLAS specific functions =============================
#ifdef USE_OPENBLAS
extern "C" {
    // Set the number of threads on runtime.
    void openblas_set_num_threads(int num_threads);

    // Get the number of threads on runtime.
    int openblas_get_num_threads(void);
}
#endif

// res[n x 2^n]
template <typename Scalar>
Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic> permutation(const int n)
{
    const int pn = (1 << n);  // 2^n
    Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic> res(n, pn);
    Scalar* r = res.data();
    for(int j = 0; j < pn; j++)
    {
        for(int i = 0; i < n; i++, r++)
        {
            *r = (j >> i) & 1;
        }
    }
    return res;
}

// w[m x n], b[m x 1], c[n x 1], dat[m x N]
template <typename Scalar>
Scalar loglik_rbm_exact(
    int m, int n, int N,
    const Scalar* wp, const Scalar* bp, const Scalar* cp, const Scalar* datp
)
{
    typedef Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic> Matrix;
    typedef Eigen::Matrix<Scalar, Eigen::Dynamic, 1> Vector;
    typedef Eigen::Map<const Matrix> MapMat;
    typedef Eigen::Map<const Vector> MapVec;

    MapMat w(wp, m, n), dat(datp, m, N);
    MapVec b(bp, m), c(cp, n);

    // log(Z)
    Scalar logz = Scalar(0);
    if(m <= n)
    {
        // If m <= n, https://arxiv.org/pdf/1510.02255.pdf, Eqn. (5)
        Matrix vperm = permutation<Scalar>(m);
        Vector logzv = vperm.transpose() * b;
        Matrix vpermwc = w.transpose() * vperm;
        vpermwc.colwise() += c;
        apply_log1exp_simd(vpermwc);
        logzv.noalias() += vpermwc.colwise().sum().transpose();
        logz = log_sum_exp(logzv);
    } else {
        // If m > n, https://arxiv.org/pdf/1510.02255.pdf, Eqn. (6)
        Matrix hperm = permutation<Scalar>(n);
        Vector logzh = hperm.transpose() * c;
        Matrix hpermwb = w * hperm;
        hpermwb.colwise() += b;
        apply_log1exp_simd(hpermwb);
        logzh.noalias() += hpermwb.colwise().sum().transpose();
        logz = log_sum_exp(logzh);
    }

    // https://arxiv.org/pdf/1510.02255.pdf, Eqn. (4)
    Scalar term1 = dat.rowwise().sum().dot(b);
    Matrix term2 = w.transpose() * dat;
    term2.colwise() += c;
    apply_log1exp_simd(term2);

    return term1 + term2.sum() - logz * N;
}

template <typename Scalar>
Scalar loglik_rbm_approx(
    int m, int n, int N,
    const Scalar* wp, const Scalar* bp, const Scalar* cp, const Scalar* datp,
    int nsamp = 100, int nstep = 10, bool vr = true
)
{
    typedef Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic> Matrix;
    typedef Eigen::Matrix<Scalar, Eigen::Dynamic, 1> Vector;
    typedef Eigen::Map<const Matrix> MapMat;
    typedef Eigen::Map<const Vector> MapVec;

    MapMat w(wp, m, n), dat(datp, m, N);
    MapVec b(bp, m), c(cp, n);

    // Half of nsamp
    const int npair = (nsamp + 1) / 2;
    if(vr)
        nsamp = 2 * npair;
    Scalar loglik = 0.0;

    // Random seeds for C++ RNG
    Rcpp::IntegerVector seeds = Rcpp::sample(100000, N, true);

    // Inside OpenMP only use one thread for OpenBLAS
#ifdef USE_OPENBLAS
    const int nthread = openblas_get_num_threads();
    openblas_set_num_threads(1);
#endif

    #pragma omp parallel for shared(seeds, dat, w, b, c, m, n, N, nsamp, nstep) reduction(+:loglik) schedule(dynamic)
    for(int i = 0; i < N; i++)
    {
        RBMSampler<Scalar> sampler(w, b, c);
        RNGEngine gen(seeds[i]);

        Vector v0 = dat.col(i);
        Matrix vmean(m, nsamp), h(n, nsamp);
        // Variance reduction
        if(vr)
            sampler.sample_k_mc_pair(gen, v0, vmean, h, nstep, npair);
        else
            sampler.sample_k_mc(gen, v0, vmean, h, nstep, nsamp);
        vmean.noalias() = w * h;
        vmean.colwise() += b;
        apply_sigmoid(vmean);

        Vector logp(nsamp);
        for(int j = 0; j < nsamp; j++)
        {
            logp[j] = loglik_bernoulli(&vmean(0, j), &dat.coeffRef(0, i), m);
        }
        loglik += log_sum_exp(logp);
    }

#ifdef USE_OPENBLAS
    openblas_set_num_threads(nthread);
#endif

    return loglik - N * std::log(Scalar(nsamp));
}


#endif  // CDTAU_LIKELIHOOD_H
