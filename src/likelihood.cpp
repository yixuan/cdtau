#include <RcppEigen.h>

using Rcpp::NumericVector;
using Rcpp::NumericMatrix;
using Eigen::VectorXd;
using Eigen::MatrixXd;
typedef Eigen::Map<VectorXd> MapVec;
typedef Eigen::Map<MatrixXd> MapMat;

inline MatrixXd permutation(const int n)
{
    const int pn = (1 << n);  // 2^n
    MatrixXd res(pn, n);
    double* r = res.data();
    for(int j = 0; j < n; j++)
    {
        for(int i = 0; i < pn; i++, r++)
        {
            *r = (i >> j) & 1;
        }
    }
    return res;
}

// w[m x n], b[m x 1], c[n x 1], v[m x N]
// [[Rcpp::export]]
double loglik_rbm(MapMat w, MapVec b, MapVec c, MapMat v)
{
    const int m = w.rows();
    const int n = w.cols();
    const int pm = (1 << m);  // 2^m
    const int pn = (1 << n);  // 2^n
    const int N = v.cols();

    // Check dimension
    if(b.size() != m || c.size() != n || v.rows() != m)
        Rcpp::stop("Dimensions do not match");

    MatrixXd vperm = permutation(m);
    MatrixXd hperm = permutation(n);
    VectorXd vbperm = vperm * b;
    VectorXd hcperm = hperm * c;
    MatrixXd joint_prob(pm, pn);
    // -E = b'v + c'h + v'wh
    joint_prob.noalias() = vperm * w * hperm.transpose();
    joint_prob.colwise() += vbperm;
    joint_prob.rowwise() += hcperm.transpose();
    // p = exp(-E)
    joint_prob.array() = joint_prob.array().exp();
    // Reuse memory
    VectorXd& vdist = vbperm;
    vdist.noalias() = joint_prob.rowwise().sum();
    const double z = vdist.sum();
    vdist /= z;
    // Free memory
    vperm.resize(1, 1);
    hperm.resize(1, 1);
    joint_prob.resize(1, 1);
    hcperm.resize(1);

    NumericVector loglik(N);
    for(int j = 0; j < N; j++)
    {
        int ind = 0;
        for(int i = 0; i < m; i++)
        {
            ind += int(v.coeff(i, j) > 0.5) << i;
        }
        loglik[j] = std::log(vdist[ind]);
    }

    return Rcpp::sum(loglik);
}
