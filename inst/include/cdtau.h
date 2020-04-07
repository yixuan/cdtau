#ifndef CDTAU_PKG_H
#define CDTAU_PKG_H

#ifdef USE_OPENBLAS
#define EIGEN_USE_BLAS
#endif

#include <RcppEigen.h>

using Rcpp::NumericVector;
using Rcpp::NumericMatrix;
using Eigen::VectorXd;
using Eigen::MatrixXd;
typedef Eigen::Map<VectorXd> MapVec;
typedef Eigen::Map<MatrixXd> MapMat;


#endif  // CDTAU_PKG_H
