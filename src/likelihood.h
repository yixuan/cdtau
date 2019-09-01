#ifndef CDTAU_LIKELIHOOD_H
#define CDTAU_LIKELIHOOD_H

#include <RcppEigen.h>
#include "mcmc.h"
#include "utils.h"
#include "utils_simd.h"

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
    // https://arxiv.org/pdf/1510.02255.pdf, Eqn. (5)
    Matrix vperm = permutation<Scalar>(m);
    Vector logzv = vperm.transpose() * b;
    Matrix vpermwc = w.transpose() * vperm;
    vpermwc.colwise() += c;
    apply_log1exp_simd(vpermwc);
    logzv.noalias() += vpermwc.colwise().sum().transpose();
    const Scalar logz = log_sum_exp(logzv);

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
    int nsamp = 100, int nstep = 10
)
{
    typedef Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic> Matrix;
    typedef Eigen::Matrix<Scalar, Eigen::Dynamic, 1> Vector;
    typedef Eigen::Map<const Matrix> MapMat;
    typedef Eigen::Map<const Vector> MapVec;

    MapMat w(wp, m, n), dat(datp, m, N);
    MapVec b(bp, m), c(cp, n);

    RBMSampler<Scalar> sampler(w, b, c);
    Vector v0(m), logp(nsamp);
    Matrix vmean(m, nsamp), h(n, nsamp);
    Scalar loglik = 0.0;

    for(int i = 0; i < N; i++)
    {
        v0.noalias() = dat.col(i);
        sampler.sample_k_mc(v0, vmean, h, nstep, nsamp);
        vmean.noalias() = w * h;
        vmean.colwise() += b;
        apply_sigmoid(vmean);

        for(int j = 0; j < nsamp; j++)
        {
            logp[j] = loglik_bernoulli(&vmean(0, j), &dat.coeffRef(0, i), m);
        }
        loglik += log_sum_exp(logp);
    }

    return loglik - N * std::log(Scalar(nsamp));
}


#endif  // CDTAU_LIKELIHOOD_H
