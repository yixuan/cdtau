#ifndef CDTAU_UTILS_SIMD_H
#define CDTAU_UTILS_SIMD_H

#include <RcppEigen.h>
#include <xsimd/xsimd.hpp>

// The common operation W * h + b in RBM
// h is a binary vector
template <typename Scalar>
void rbm_op_v_simd(
    const Eigen::Ref< const Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic> >& w,
    const Eigen::Matrix<Scalar, Eigen::Dynamic, 1>& h,
    const Eigen::Ref< const Eigen::Matrix<Scalar, Eigen::Dynamic, 1> >& b,
    Eigen::Matrix<Scalar, Eigen::Dynamic, 1>& v
)
{
    typedef xsimd::batch<Scalar, xsimd::simd_type<Scalar>::size> vec;

    const int m = w.rows();
    const int n = w.cols();

    const int simd_size = xsimd::simd_type<Scalar>::size;
    const int vec_size = m - m % simd_size;

    // Fall back to default implementation
    if(vec_size != m)
    {
        v.noalias() = w * h + b;
        return;
    }

    const Scalar* hp = h.data();
    const Scalar* bp = b.data();
    Scalar* vp = v.data();

    // Copy b to v
    for(int i = 0; i < vec_size; i += simd_size)
    {
        vec bi = xsimd::load_aligned(bp + i);
        bi.store_aligned(vp + i);
    }

    for(int j = 0; j < n; j++)
    {
        if(hp[j] > Scalar(0.5))
        {
            const Scalar* colp = w.col(j).data();

            for(int i = 0; i < vec_size; i += simd_size)
            {
                vec vi = xsimd::load_aligned(vp + i);
                vec wi = xsimd::load_aligned(colp + i);
                vec res = vi + wi;
                res.store_aligned(vp + i);
            }
        }
    }
}

// w += v1 * h1' + v2 * h2' - v3 * h3' - v4 * h4'
// h2 and h4 are binary vectors
template <typename Scalar>
void rbm_op_rank4_simd(
    const Eigen::Matrix<Scalar, Eigen::Dynamic, 1>& v1,
    const Eigen::Matrix<Scalar, Eigen::Dynamic, 1>& h1,
    const Eigen::Matrix<Scalar, Eigen::Dynamic, 1>& v2,
    const Eigen::Matrix<Scalar, Eigen::Dynamic, 1>& h2,
    const Eigen::Matrix<Scalar, Eigen::Dynamic, 1>& v3,
    const Eigen::Matrix<Scalar, Eigen::Dynamic, 1>& h3,
    const Eigen::Matrix<Scalar, Eigen::Dynamic, 1>& v4,
    const Eigen::Matrix<Scalar, Eigen::Dynamic, 1>& h4,
    Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic>& w
)
{
    typedef xsimd::batch<Scalar, xsimd::simd_type<Scalar>::size> vec;

    const int m = w.rows();
    const int n = w.cols();

    const int simd_size = xsimd::simd_type<Scalar>::size;
    const int vec_size = m - m % simd_size;

    // Fall back to default implementation
    if(vec_size != m)
    {
        w.noalias() += v1 * h1.transpose() + v2 * h2.transpose() - v3 * h3.transpose() - v4 * h4.transpose();
        return;
    }

    const Scalar* v1p = v1.data();
    const Scalar* v2p = v2.data();
    const Scalar* v3p = v3.data();
    const Scalar* v4p = v4.data();

    for(int j = 0; j < n; j++)
    {
        Scalar* colp = w.col(j).data();
        vec h1j = xsimd::set_simd(h1[j]);
        vec h3j = xsimd::set_simd(h3[j]);

        if(h2[j] <= Scalar(0.5) && h4[j] <= Scalar(0.5))
        {
            // w += v1 * h1 - v3 * h3
            for(int i = 0; i < vec_size; i += simd_size)
            {
                vec wi = xsimd::load_aligned(colp + i);
                vec v1i = xsimd::load_aligned(v1p + i);
                vec v3i = xsimd::load_aligned(v3p + i);
                // wi += v1i * h1j - v3i * h3j;
                wi += xsimd::fms(v1i, h1j, v3i * h3j);
                wi.store_aligned(colp + i);
            }
        } else if(h2[j] <= Scalar(0.5) && h4[j] > Scalar(0.5))
        {
            // w += v1 * h1 - v3 * h3 - v4
            for(int i = 0; i < vec_size; i += simd_size)
            {
                vec wi = xsimd::load_aligned(colp + i);
                vec v1i = xsimd::load_aligned(v1p + i);
                vec v3i = xsimd::load_aligned(v3p + i);
                vec v4i = xsimd::load_aligned(v4p + i);
                // wi += v1i * h1j - v3i * h3j - v4i;
                wi += v1i * h1j - xsimd::fma(v3i, h3j, v4i);
                wi.store_aligned(colp + i);
            }
        } else if(h2[j] > Scalar(0.5) && h4[j] <= Scalar(0.5))
        {
            // w += v1 * h1 + v2 - v3 * h3
            for(int i = 0; i < vec_size; i += simd_size)
            {
                vec wi = xsimd::load_aligned(colp + i);
                vec v1i = xsimd::load_aligned(v1p + i);
                vec v2i = xsimd::load_aligned(v2p + i);
                vec v3i = xsimd::load_aligned(v3p + i);
                // wi += v1i * h1j + v2i - v3i * h3j;
                wi += xsimd::fma(v1i, h1j, v2i) - v3i * h3j;
                wi.store_aligned(colp + i);
            }
        } else {
            // w += v1 * h1 + v2 - v3 * h3 - v4
            for(int i = 0; i < vec_size; i += simd_size)
            {
                vec wi = xsimd::load_aligned(colp + i);
                vec v1i = xsimd::load_aligned(v1p + i);
                vec v2i = xsimd::load_aligned(v2p + i);
                vec v3i = xsimd::load_aligned(v3p + i);
                vec v4i = xsimd::load_aligned(v4p + i);
                // wi += v1i * h1j + v2i - v3i * h3j - v4i;
                wi += xsimd::fma(v1i, h1j, v2i) - xsimd::fma(v3i, h3j, v4i);
                wi.store_aligned(colp + i);
            }
        }
    }
}

// x => log(1 + exp(x))
template <typename Derived>
void apply_log1exp_simd(Eigen::MatrixBase<Derived>& x)
{
    /* Eigen::ArrayXXd max0 = x.array().max(0.0);
     x.array() = 1.0 + (-x.array().abs()).exp();
     x.array() = x.array().log();
     x.array() += max0; */

    typedef typename Derived::Scalar Scalar;
    typedef xsimd::batch<Scalar, xsimd::simd_type<Scalar>::size> vec;

    Scalar* xp = x.derived().data();
    const int n = x.size();
    const int simd_size = xsimd::simd_type<Scalar>::size;
    const int vec_size = n - n % simd_size;

    vec zero = xsimd::set_simd(Scalar(0));
    vec one = xsimd::set_simd(Scalar(1));

    for(int i = 0; i < vec_size; i += simd_size)
    {
        vec xi = xsimd::load_aligned(xp + i);
        xi = xsimd::log(one + xsimd::exp(-xsimd::abs(xi))) + xsimd::max(zero, xi);
        xi.store_aligned(xp + i);
    }
    for(int i = vec_size; i < n; i++)
    {
        xp[i] = std::log(Scalar(1) + std::exp(-std::abs(xp[i]))) + std::max(xp[i], Scalar(0));
    }
}

// x => sigmoid(x)
// x is clipped to [-10, 10]
template <typename Derived>
void apply_sigmoid_simd(Eigen::MatrixBase<Derived>& x)
{
    /* x.array() = x.array().max(Scalar(-10)).min(Scalar(10));
     x.array() = Scalar(1) / (Scalar(1) + (-x).array().exp()); */

    typedef typename Derived::Scalar Scalar;
    typedef xsimd::batch<Scalar, xsimd::simd_type<Scalar>::size> vec;

    Scalar* xp = x.derived().data();
    const int n = x.size();
    const int simd_size = xsimd::simd_type<Scalar>::size;
    const int vec_size = n - n % simd_size;

    vec one = xsimd::set_simd(Scalar(1));
    vec ten = xsimd::set_simd(Scalar(10));
    vec mten = xsimd::set_simd(Scalar(-10));

    for(int i = 0; i < vec_size; i += simd_size)
    {
        vec xi = xsimd::load_aligned(xp + i);
        xi = one / (one + xsimd::exp(-xsimd::clip(xi, mten, ten)));
        xi.store_aligned(xp + i);
    }
    for(int i = vec_size; i < n; i++)
    {
        const Scalar xi = std::max(-Scalar(10), std::min(Scalar(10), xp[i]));
        xp[i] = Scalar(1) / (Scalar(1) + std::exp(-xi));
    }
}

// x * log(p) + (1 - x) * log(1 - p)
template <typename Scalar>
Scalar loglik_bernoulli_simd(const Scalar* prob, const Scalar* x, int n)
{
    typedef xsimd::batch<Scalar, xsimd::simd_type<Scalar>::size> vec;

    const int simd_size = xsimd::simd_type<Scalar>::size;
    const int vec_size = n - n % simd_size;

    vec one = xsimd::set_simd(Scalar(1));
    vec half = xsimd::set_simd(Scalar(0.5));

    Scalar res = 0;
    for(int i = 0; i < vec_size; i += simd_size)
    {
        vec probi = xsimd::load_aligned(prob + i);
        vec one_m_probi = one - probi;
        vec xi = xsimd::load_aligned(x + i);
        vec r = xsimd::log(xsimd::select(xi > half, probi, one_m_probi));
        res += xsimd::hadd(r);
    }
    for(int i = vec_size; i < n; i++)
    {
        res += (x[i] > Scalar(0.5)) ? (std::log(prob[i])) : (std::log(Scalar(1) - prob[i]));
    }

    return res;
}
template <typename Scalar>
Scalar loglik_bernoulli_simd(const Eigen::Matrix<Scalar, Eigen::Dynamic, 1>& prob, const Eigen::Matrix<Scalar, Eigen::Dynamic, 1>& x)
{
    return loglik_bernoulli_simd(prob.data(), x.data(), prob.size());
}

// res ~ Bernoulli(prob), given prob and uniform random variates
// If antithetic == true, use 1-U as the uniform random variate
template <typename Derived>
void random_bernoulli_uvar_simd(const Eigen::MatrixBase<Derived>& prob,
                                const Eigen::MatrixBase<Derived>& uvar,
                                Eigen::MatrixBase<Derived>& res,
                                bool antithetic = false)
{
    typedef typename Derived::Scalar Scalar;
    typedef xsimd::batch<Scalar, xsimd::simd_type<Scalar>::size> vec;

    const Scalar* pp = prob.derived().data();
    const Scalar* up = uvar.derived().data();
    Scalar* rp = res.derived().data();
    const int n = res.size();
    const int simd_size = xsimd::simd_type<Scalar>::size;
    const int vec_size = n - n % simd_size;

    vec one = xsimd::set_simd(Scalar(1));
    vec zero = xsimd::set_simd(Scalar(0));

    if(antithetic)
    {
        for(int i = 0; i < vec_size; i += simd_size)
        {
            vec probi = xsimd::load_aligned(pp + i);
            vec uvari = xsimd::load_aligned(up + i);
            vec resi = xsimd::select(uvari >= one - probi, one, zero);
            resi.store_aligned(rp + i);
        }
        for(int i = vec_size; i < n; i++)
        {
            rp[i] = Scalar(up[i] >= Scalar(1) - pp[i]);
        }
    } else {
        for(int i = 0; i < vec_size; i += simd_size)
        {
            vec probi = xsimd::load_aligned(pp + i);
            vec uvari = xsimd::load_aligned(up + i);
            vec resi = xsimd::select(uvari <= probi, one, zero);
            resi.store_aligned(rp + i);
        }
        for(int i = vec_size; i < n; i++)
        {
            rp[i] = Scalar(up[i] <= pp[i]);
        }
    }
}

#endif  // CDTAU_UTILS_SIMD_H
