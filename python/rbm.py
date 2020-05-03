import math
import numpy as np
import mxnet as mx
from mxnet import nd
from sampler import RBMSampler, UnbiasedRBMSampler

def log_sum_exp(x):
    c = nd.max(x).asscalar()
    return math.log(nd.sum(nd.exp(x - c)).asscalar()) + c

# https://stackoverflow.com/a/47521145
def vec_bin_array(arr, m):
    """
    Arguments:
    arr: Numpy array of positive integers
    m: Number of bits of each integer to retain

    Returns a copy of arr with every element replaced with a bit vector.
    Bits encoded as int8's.
    """
    to_str_func = np.vectorize(lambda x: np.binary_repr(x).zfill(m))
    strs = to_str_func(arr)
    ret = np.zeros(list(arr.shape) + [m], dtype=np.int8)
    for bit_ix in range(0, m):
        fetch_bit_func = np.vectorize(lambda x: x[bit_ix] == '1')
        ret[..., bit_ix] = fetch_bit_func(strs).astype("int8")

    return ret

class RBM:
    # Constructor
    def __init__(self, m, n, w0=None, b0=None, c0=None, ctx=mx.cpu()):
        self.ctx = ctx

        # Dimensions
        self.m = m
        self.n = n

        # Parameters
        if w0 is not None and b0 is not None and c0 is not None:
            if w0.shape != (m, n):
                raise ValueError("w0 must be an [m x n] array")
            if b0.shape != (m, ):
                raise ValueError("b0 must be an [m] array")
            if c0.shape != (n, ):
                raise ValueError("c0 must be an [n] array")
            self.w = w0.as_in_context(self.ctx)
            self.b = b0.as_in_context(self.ctx)
            self.c = c0.as_in_context(self.ctx)
        else:
            self.w = nd.random.normal(scale=0.1, shape=(m, n), ctx=self.ctx)
            self.b = nd.random.normal(scale=0.1, shape=m, ctx=self.ctx)
            self.c = nd.random.normal(scale=0.1, shape=n, ctx=self.ctx)

        # Gradients
        self.dw1 = nd.empty(shape=self.w.shape, ctx=self.ctx)
        self.db1 = nd.empty(shape=self.b.shape, ctx=self.ctx)
        self.dc1 = nd.empty(shape=self.c.shape, ctx=self.ctx)

        self.dw2 = nd.empty(shape=self.w.shape, ctx=self.ctx)
        self.db2 = nd.empty(shape=self.b.shape, ctx=self.ctx)
        self.dc2 = nd.empty(shape=self.c.shape, ctx=self.ctx)

    # Approximate log-likelihood value
    def loglik(self, dat, nobs=100, nmc=30, nstep=10):
        if nobs > dat.shape[0]:
            nobs = dat.shape[0]
        ind = np.random.choice(dat.shape[0], nobs, replace=False)
        samp = RBMSampler(self.w, self.b, self.c, ctx=self.ctx)
        loglik = 0.0
        for i in range(nobs):
            vi = dat[ind[i]]
            v, h = samp.sample_k(vi.reshape(1, -1).repeat(nmc, axis=0), nstep)
            vmean = nd.sigmoid(nd.dot(h, self.w.T) + self.b)
            logp = nd.log(vmean) * vi + nd.log(1 - vmean) * (1 - vi)
            logp = nd.sum(logp, axis=1)
            loglik += log_sum_exp(logp)
        return loglik - nobs * math.log(nmc)

    # Exact log-likelihood value
    # dat[N x m]
    def loglik_exact(self, dat):
        N = dat.shape[0]

        # log(Z)
        # https://arxiv.org/pdf/1510.02255.pdf, Eqn. (5)
        vperm = nd.array(vec_bin_array(np.arange(2 ** self.m), self.m), ctx=self.ctx)
        vpermwc = nd.dot(vperm, self.w) + self.c
        logzv = nd.dot(vperm, self.b) + nd.sum(nd.log(1 + nd.exp(vpermwc)), axis=1)
        logz = log_sum_exp(logzv)

        # https://arxiv.org/pdf/1510.02255.pdf, Eqn. (4)
        term1 = nd.dot(dat, self.b)
        term2 = nd.log(1 + nd.exp(nd.dot(dat, self.w) + self.c))
        loglik = nd.sum(term1) + nd.sum(term2) - logz * N
        return loglik.asscalar()

    # First term of the gradient
    # Mini-batch vmb [b x m]
    def compute_grad1(self, vmb):
        # Mean of hidden units given vmb
        hmean = nd.sigmoid(nd.dot(vmb, self.w) + self.c)

        self.db1 = nd.mean(vmb, axis=0, out=self.db1)
        self.dc1 = nd.mean(hmean, axis=0, out=self.dc1)
        self.dw1 = nd.dot(vmb.T, hmean, out=self.dw1)
        self.dw1 /= vmb.shape[0]

    # Zero out gradients
    def zero_grad2(self):
        self.db2 = nd.zeros_like(self.db2, out=self.db2)
        self.dc2 = nd.zeros_like(self.dc2, out=self.dc2)
        self.dw2 = nd.zeros_like(self.dw2, out=self.dw2)

    # Compute the second term of gradient using CD-k
    # dat [N x m]
    def accumulate_grad2_cdk(self, dat, k=1, nchain=1):
        # Initial values for Gibbs sampling
        N = dat.shape[0]
        ind = np.random.choice(N, nchain)
        v0 = dat[ind, :]

        # Gibbs samples
        samp = RBMSampler(self.w, self.b, self.c, ctx=self.ctx)
        v, h = samp.sample_k(v0, k=k)

        # Second term
        hmean = nd.sigmoid(nd.dot(v, self.w) + self.c)
        self.db2 = nd.sum(v, axis=0, out=self.db2)
        self.dc2 = nd.sum(hmean, axis=0, out=self.dc2)
        self.dw2 = nd.dot(v.T, hmean, out=self.dw2)

    # Compute the second term of gradient using unbiased CD
    # dat [N x m]
    def accumulate_grad2_ucd(self, dat, min_mcmc=1, max_mcmc=100):
        # Initial value for Gibbs sampling
        N = dat.shape[0]
        ind = np.random.choice(N, 1)[0]
        v0 = dat[ind, :]

        # Gibbs samples
        samp = UnbiasedRBMSampler(self.w, self.b, self.c, ctx=self.ctx)
        vhist, vchist, disc = samp.sample(v0, min_steps=min_mcmc, max_steps=max_mcmc)

        burnin = min_mcmc - 1
        tau = vchist.shape[0]
        remain = tau - burnin

        vk = vhist[burnin, :]
        hk_mean = nd.sigmoid(nd.dot(self.w.T, vk) + self.c)

        hhist_mean = nd.sigmoid(nd.dot(vhist[-remain:, :], self.w) + self.c)
        hchist_mean = nd.sigmoid(nd.dot(vchist[-remain:, :], self.w) + self.c)

        # Second term
        self.db2 += vk + nd.sum(vhist[-remain:, :], axis=0) -\
                    nd.sum(vchist[-remain:, :], axis=0)
        self.dc2 += hk_mean + nd.sum(hhist_mean, axis=0) -\
                    nd.sum(hchist_mean, axis=0)
        self.dw2 += nd.dot(vk.reshape(-1, 1), hk_mean.reshape(1, -1)) +\
                    nd.dot(vhist[-remain:, :].T, hhist_mean) -\
                    nd.dot(vchist[-remain:, :].T, hchist_mean)

        return tau, disc

    # Update parameters
    def update_param(self, lr, nchain):
        self.b += lr * (self.db1 - self.db2 / nchain)
        self.c += lr * (self.dc1 - self.dc2 / nchain)
        self.w += lr * (self.dw1 - self.dw2 / nchain)

    # Train RBM using CD-k
    def train_cdk(self, dat, batch_size, epochs, lr=0.01, k=1, nchain=1, report_freq=1, exact_loglik=False):
        N = dat.shape[0]
        ind = np.arange(N)
        loglik = []

        for epoch in range(epochs):
            np.random.shuffle(ind)

            for i in range(0, N, batch_size):
                ib = i // batch_size + 1
                batchid = ind[i:(i + batch_size)]
                vmb = dat[batchid, :]

                self.compute_grad1(vmb)
                self.zero_grad2()
                self.accumulate_grad2_cdk(dat, k, nchain)
                self.update_param(lr, nchain)

                if ib % report_freq == 0:
                    if exact_loglik:
                        ll = self.loglik_exact(dat)
                    else:
                        ll = self.loglik(dat, nobs=100)
                    loglik.append(ll)
                    print("epoch = {}, batch = {}, loglik = {}".format(epoch, ib, ll))

        return loglik

    # Train RBM using Unbiased CD
    def train_ucd(self, dat, batch_size, epochs, lr=0.01, min_mcmc=1, max_mcmc=100, nchain=1, report_freq=1, exact_loglik=False):
        N = dat.shape[0]
        ind = np.arange(N)
        loglik = []
        tau = []
        disc = []

        for epoch in range(epochs):
            np.random.shuffle(ind)

            tt = 0.0
            dd = 0.0
            for i in range(0, N, batch_size):
                ib = i // batch_size + 1
                batchid = ind[i:(i + batch_size)]
                bs = batchid.size
                vmb = dat[batchid, :]

                self.compute_grad1(vmb)
                self.zero_grad2()
                for j in range(nchain):
                    tau_t, disc_t = self.accumulate_grad2_ucd(dat, min_mcmc=min_mcmc, max_mcmc=max_mcmc)
                    tt += tau_t
                    dd += disc_t
                self.update_param(lr, nchain)

                if ib % report_freq == 0:
                    if exact_loglik:
                        ll = self.loglik_exact(dat)
                    else:
                        ll = self.loglik(dat, nobs=100)
                    loglik.append(ll)
                    tau.append(tt / nchain / report_freq)
                    disc.append(dd / nchain / report_freq)
                    tt = 0.0
                    dd = 0.0
                    print("epoch = {}, batch = {}, loglik = {}".format(epoch, ib, ll))

        return loglik, tau, disc



if __name__ == "__main__":
    # Simulate an RBM model
    mx.random.seed(123)
    np.random.seed(123)
    ctx = mx.cpu()
    m = 10
    n = 5
    w = nd.random.normal(shape=(m, n), ctx=ctx)
    b = nd.random.normal(shape=m, ctx=ctx)
    c = nd.random.normal(shape=n, ctx=ctx)

    # Test CD-k sampler and use it to generate initial values
    samp = RBMSampler(w, b, c, ctx=ctx)
    N = 100
    v0 = nd.random.randint(0, 2, shape=(N, m), ctx=ctx).astype(w.dtype)
    v, h = samp.sample_k(v0, 30)

    # Test log-likelihood value on the sampled data
    rbm = RBM(m, n, w, b, c, ctx=ctx)
    print(rbm.loglik(v, nobs=N))
    print(rbm.loglik_exact(v))

    # Train RBM using CD-k
    mx.random.seed(123)
    np.random.seed(123)
    rbm = RBM(m, n, ctx=ctx)
    res = rbm.train_cdk(v, batch_size=25, epochs=5, lr=0.1,
                        k=1, nchain=100,
                        report_freq=1, exact_loglik=True)

    # Train RBM using unbiased CD
    mx.random.seed(123)
    np.random.seed(123)
    rbm = RBM(m, n, ctx=ctx)
    res = rbm.train_ucd(v, batch_size=25, epochs=5, lr=0.1,
                        min_mcmc=1, max_mcmc=100, nchain=100,
                        report_freq=1, exact_loglik=True)
