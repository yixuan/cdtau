#' Train RBM Models
#'
#' \code{rbm_cdk()} uses the traditional CD-k algorithm to train RBM.
#' \code{rbm_ucd()} uses the unbiased CD algorithm.
#'
#' @param m             Dimension of the visible units.
#' @param n             Dimension of the hidden units.
#' @param dat           The observed data, of size \code{[m x N]}.
#' @param w0            Initial value for the weight parameter of the RBM, of size \code{[m x n]}.
#' @param b0            Initial value for the bias parameter for the visible units, of size \code{[m x 1]}.
#' @param c0            Initial value for the bias parameter for the hidden units, of size \code{[n x 1]}.
#' @param batch_size    Size of the mini-batch.
#' @param lr            Learning rate.
#' @param momentum      Momentum coefficient in SGD.
#' @param niter         Number of iterations.
#' @param ngibbs        The "k" in the CD-k algorithm.
#' @param nchain        Number of independent Markov chains to approximate gradient.
#' @param persistent    Whether to use PCD instead of CD.
#' @param eval_loglik   Whether to compute log-likelihood values during training.
#' @param exact_loglik  Compute exact or approximate log-likelihood values.
#' @param eval_freq     Evaluate log-likelihood every \code{eval_freq} mini-batches.
#' @param eval_size     Size of sub-sampled data to evaluate log-likelihood.
#' @param eval_nmc      Size of the Monte Carlo sample for approximating the log-likelihood.
#' @param eval_nstep    Number of steps in the Gibbs sampler for approximating the log-likelihood.
#' @param verbose       Level of verbosity.
#'
#' @examples \dontrun{
#' # Bars-and-stripes data, Schulz et al. (2010), Fischer and Igel (2010)
#' d = 4
#' n = m = d^2
#' N = 2^d
#' dat = matrix(0, 2 * N, m)
#'
#' for(i in 1:N)
#' {
#'     bits = as.integer(rev(intToBits(i - 1)[1:d]))
#'     mat = tcrossprod(bits, rep(1, d))
#'     dat[2 * i - 1, ] = as.numeric(mat)
#'     dat[2 * i, ] = as.numeric(t(mat))
#' }
#'
#' N = nrow(dat)
#'
#' # Persistent contrastive divergence
#' set.seed(123)
#' pcd = rbm_cdk(m, n, t(dat), batch_size = N, lr = 0.1, niter = 1000,
#'               ngibbs = 1, nchain = 1000, persistent = TRUE,
#'               eval_loglik = TRUE, exact_loglik = TRUE,
#'               eval_freq = 1, eval_size = N, verbose = 1)
#'
#' # Unbiased contrastive divergence
#' set.seed(123)
#' ucd = rbm_ucd(m, n, t(dat), batch_size = N, lr = 0.1, niter = 1000,
#'               min_mcmc = 1, max_mcmc = 100, nchain = 1000,
#'               eval_loglik = TRUE, exact_loglik = TRUE,
#'               eval_freq = 1, eval_size = N, verbose = 1)
#'
#' plot(pcd$loglik, type = "l")
#' lines(ucd$loglik, col = "blue")
#' }
#'
#' @rdname rbm_cd
#'
rbm_cdk = function(
    m, n, dat, b0 = NULL, c0 = NULL, w0 = NULL,
    batch_size = 10L, lr = 0.1, momentum = 0.0, niter = 100L, ngibbs = 10L, nchain = 1L, persistent = FALSE,
    eval_loglik = FALSE, exact_loglik = FALSE,
    eval_freq = 10L, eval_size = 100L, eval_nmc = 100L, eval_nstep = 10L, verbose = 0L
)
{
    if(is.null(b0) || is.null(c0) || is.null(w0))
    {
        .Call(`_cdtau_rbm_cdk_`, m, n, dat, batch_size, lr, momentum, niter, ngibbs, nchain, persistent, eval_loglik, exact_loglik, eval_freq, eval_size, eval_nmc, eval_nstep, verbose)
    } else {
        .Call(`_cdtau_rbm_cdk_warm_`, m, n, dat, b0, c0, w0, batch_size, lr, momentum, niter, ngibbs, nchain, persistent, eval_loglik, exact_loglik, eval_freq, eval_size, eval_nmc, eval_nstep, verbose)
    }
}

#' @rdname rbm_cd
#'
rbm_ucd = function(
    m, n, dat, b0 = NULL, c0 = NULL, w0 = NULL,
    batch_size = 10L, lr = 0.1, momentum = 0.0, niter = 100L, min_mcmc = 1L, max_mcmc = 100L, nchain = 1L,
    eval_loglik = FALSE, exact_loglik = FALSE,
    eval_freq = 10L, eval_size = 100L, eval_nmc = 100L, eval_nstep = 10L, verbose = 0L
)
{
    if(is.null(b0) || is.null(c0) || is.null(w0))
    {
        .Call(`_cdtau_rbm_ucd_`, m, n, dat, batch_size, lr, momentum, niter, min_mcmc, max_mcmc, nchain, eval_loglik, exact_loglik, eval_freq, eval_size, eval_nmc, eval_nstep, verbose)
    } else {
        .Call(`_cdtau_rbm_ucd_warm_`, m, n, dat, b0, c0, w0, batch_size, lr, momentum, niter, min_mcmc, max_mcmc, nchain, eval_loglik, exact_loglik, eval_freq, eval_size, eval_nmc, eval_nstep, verbose)
    }
}
