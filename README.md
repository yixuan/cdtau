## CD-τ <img src="https://statr.me/images/sticker-cdtau.png" alt="cdtau" height="150px" align="right" />

The `cdtau` R package implements the unbiased contrastive divergence (UCD) algorithm based on the paper
[Unbiased Contrastive Divergence Algorithm for Training Energy-Based Latent Variable Models](https://openreview.net/forum?id=r1eyceSYPr) (ICLR 2020) by Yixuan Qiu, Lingsong Zhang, and Xiao Wang.

### Installation

Currently `cdtau` has not been submitted to CRAN, but it can be installed just like any other R
package. For `devtools` users, the following command should work on most platforms:

```r
library(devtools)
install_github("yixuan/cdtau")
```

Note that a C++ compiler that supports the C++14 standard is needed.

For best performance, it is **strongly suggested** linking your R to the
[OpenBLAS](https://www.openblas.net/) library for matrix computation. You can achieve this with the
help of the [ropenblas](https://prdm0.github.io/ropenblas/) package.

After setting up OpenBLAS, the following steps will optimize `cdtau` to make the best use of OpenBLAS.
First download this repository to a folder called `cdtau`, and then uncomment the following line of
`cdtau/src/Makevars`.

```
# Uncomment the following line if you have linked R to OpenBLAS
# BLAS_FLAGS = -DUSE_OPENBLAS -DEIGEN_USE_BLAS
```

Now `cdtau/src/Makevars` should look like

```
CXX_STD = CXX14

# Uncomment the following line if you have linked R to OpenBLAS
BLAS_FLAGS = -DUSE_OPENBLAS -DEIGEN_USE_BLAS

PKG_CPPFLAGS = -march=native -I. -DEIGEN_MAX_ALIGN_BYTES=64 $(BLAS_FLAGS)
PKG_CXXFLAGS = $(SHLIB_OPENMP_CXXFLAGS)
PKG_LIBS = $(LAPACK_LIBS) $(BLAS_LIBS) $(FLIBS) $(SHLIB_OPENMP_CXXFLAGS)

```

Finally, install the package using the following command:


```bash
cd cdtau
R CMD INSTALL .
```

### Example

The following code demonstrates the training of restricted boltzmann machine (RBM) on the
bars-and-stripes data set.

```r
library(cdtau)
# Bars-and-stripes data, Schulz et al. (2010), Fischer and Igel (2010)
d = 4
n = m = d^2
N = 2^d
dat = matrix(0, 2 * N, m)

for(i in 1:N)
{
    bits = as.integer(rev(intToBits(i - 1)[1:d]))
    mat = tcrossprod(bits, rep(1, d))
    dat[2 * i - 1, ] = as.numeric(mat)
    dat[2 * i, ] = as.numeric(t(mat))
}

N = nrow(dat)

# Persistent contrastive divergence
set.seed(123)
pcd = rbm_cdk(m, n, t(dat), batch_size = N, lr = 0.1, niter = 10000,
              ngibbs = 1, nchain = 1000, persistent = TRUE,
              eval_loglik = TRUE, exact_loglik = TRUE,
              eval_freq = 1, eval_size = N, verbose = 1)

# Unbiased contrastive divergence
set.seed(123)
ucd = rbm_ucd(m, n, t(dat), batch_size = N, lr = 0.1, niter = 10000,
              min_mcmc = 1, max_mcmc = 100, nchain = 1000,
              eval_loglik = TRUE, exact_loglik = TRUE,
              eval_freq = 1, eval_size = N, verbose = 1)

plot(pcd$loglik, type = "l", xlab = "Iterations", ylab = "Log-likelihood Value")
lines(ucd$loglik, col = "blue")
```

### Citation

Please consider to cite this work if the article or the package has helped you.


```
@inproceedings{qiu2020unbiased,
    title={Unbiased Contrastive Divergence Algorithm for Training Energy-Based Latent Variable Models},
    author={Yixuan Qiu and Lingsong Zhang and Xiao Wang},
    booktitle={International Conference on Learning Representations},
    year={2020},
    url={https://openreview.net/forum?id=r1eyceSYPr}
}
```