#include <RcppEigen.h>
#include "utils.h"

using Rcpp::NumericVector;
using Rcpp::NumericMatrix;
using Eigen::VectorXd;
using Eigen::MatrixXd;
typedef Eigen::Map<VectorXd> MapVec;
typedef Eigen::Map<MatrixXd> MapMat;

// res[n x 2^n]
inline MatrixXd permutation(const int n)
{
    const int pn = (1 << n);  // 2^n
    MatrixXd res(n, pn);
    double* r = res.data();
    for(int j = 0; j < pn; j++)
    {
        for(int i = 0; i < n; i++, r++)
        {
            *r = (j >> i) & 1;
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
    const int N = v.cols();

    // Check dimension
    if(b.size() != m || c.size() != n || v.rows() != m)
        Rcpp::stop("Dimensions do not match");

    // log(Z)
    // https://arxiv.org/pdf/1510.02255.pdf, Eqn. (5)
    MatrixXd vperm = permutation(m);
    VectorXd logzv = vperm.transpose() * b;
    MatrixXd vpermwc = w.transpose() * vperm;
    vpermwc.colwise() += c;
    apply_log1exp(vpermwc);
    logzv.noalias() += vpermwc.colwise().sum().transpose();
    const double logz = log_sum_exp(logzv);

    // https://arxiv.org/pdf/1510.02255.pdf, Eqn. (4)
    VectorXd loglik(N);
    VectorXd term1 = v.transpose() * b;
    MatrixXd term2 = w.transpose() * v;
    term2.colwise() += c;
    apply_log1exp(term2);
    loglik.noalias() = term1 + term2.colwise().sum().transpose();

    return loglik.sum() - logz * N;
}
