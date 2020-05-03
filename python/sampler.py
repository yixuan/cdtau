import random
import mxnet as mx
from mxnet import nd

class RBMSampler:
    # Constructor
    # w [m x n], b [m], c [n]
    def __init__(self, w, b, c, ctx=mx.cpu()):
        if len(w.shape) != 2:
            raise ValueError("w must be a 2-d array")
        if len(b.shape) != 1 or len(c.shape) != 1:
            raise ValueError("b and c must be 1-d arrays")

        self.ctx = ctx
        self.m = w.shape[0]
        self.n = w.shape[1]
        self.w = w.as_in_context(self.ctx)
        self.b = b.as_in_context(self.ctx)
        self.c = c.as_in_context(self.ctx)

    # Sample v given h
    # h [N x n]
    def sample_v_given_h(self, h):
        vmean = nd.sigmoid(nd.dot(h, self.w.T) + self.b)
        v = nd.random.uniform(shape=vmean.shape, ctx=self.ctx) <= vmean
        return v

    # Sample h given v
    # v [N x m]
    def sample_h_given_v(self, v):
        hmean = nd.sigmoid(nd.dot(v, self.w) + self.c)
        h = nd.random.uniform(shape=hmean.shape, ctx=self.ctx) <= hmean
        return h

    # Gibbs with k steps
    # v0 [N x m]
    def sample_k(self, v0, k):
        h = self.sample_h_given_v(v0)
        for i in range(k):
            v = self.sample_v_given_h(h)
            h = self.sample_h_given_v(v)
        return v, h



class UnbiasedRBMSampler:
    # Constructor
    # w [m x n], b [m], c [n]
    def __init__(self, w, b, c, ctx=mx.cpu()):
        if len(w.shape) != 2:
            raise ValueError("w must be a 2-d array")
        if len(b.shape) != 1 or len(c.shape) != 1:
            raise ValueError("b and c must be 1-d arrays")

        self.ctx = ctx
        self.m = w.shape[0]
        self.n = w.shape[1]
        self.w = w.as_in_context(self.ctx)
        self.b = b.as_in_context(self.ctx)
        self.c = c.as_in_context(self.ctx)

    def clip_sigmoid(self, x):
        return nd.sigmoid(nd.clip(x, -10.0, 10.0))

    # Sample from Bernoulli distribution
    def sample_bernoulli(self, prob):
        return nd.random.uniform(shape=prob.shape, ctx=self.ctx) <= prob

    # Sample v given h
    # h [N x n]
    def sample_v_given_h(self, h):
        vmean = nd.sigmoid(nd.dot(h, self.w.T) + self.b)
        return self.sample_bernoulli(vmean)

    # Sample h given v
    # v [N x m]
    def sample_h_given_v(self, v):
        hmean = nd.sigmoid(nd.dot(v, self.w) + self.c)
        return self.sample_bernoulli(hmean)

    # (xi1, eta0) -> (xi2, eta1) -> ...
    # xi = (v, h), eta = (vc, hc)
    def max_coup(self, vc0, hc0, v1, h1, max_try=10):
        # Sample the xi chain
        # p(v | h1)
        v2mean = self.clip_sigmoid(nd.dot(self.w, h1) + self.b)
        v2 = self.sample_bernoulli(v2mean)
        # p(h | v)
        h2 = self.sample_h_given_v(v2)

        # If xi1 == eta0, also make xi2 == eta1 and early exit
        if nd.norm(v1 - vc0).asscalar() == 0 and nd.norm(h1 - hc0).asscalar() == 0:
            vc1 = v2.copy()
            hc1 = h2.copy()
            return vc1, hc1, v2, h2, 0

        # Let the two chains meet with a positive probability
        # p((v, h) | xi1) = p1(v | h1) * p2(h | v)
        # p((v, h) | eta0) = q1(v | hc0) * q2(h | v)
        # p2 = q2, so p((v, h) | xi1) / p((v, h) | eta0) = p1(v | h1) / q1(v | hc0)
        vc1mean = self.clip_sigmoid(nd.dot(self.w, hc0) + self.b)
        logpxi1 = nd.sum(nd.log(v2mean) * v2 + nd.log(1 - v2mean) * (1 - v2)).asscalar()
        logpeta0 = nd.sum(nd.log(vc1mean) * v2 + nd.log(1 - vc1mean) * (1 - v2)).asscalar()
        u = nd.random.exponential().asscalar()
        if u >= logpxi1 - logpeta0:
            vc1 = v2.copy()
            hc1 = h2.copy()
            return vc1, hc1, v2, h2, 0

        # Otherwise, sample the two chains conditional on no-meet
        v2 = None
        vc1 = None
        for i in range(max_try):
            # Common RNG
            uv = nd.random.uniform(shape=self.m, ctx=self.ctx)
            # Sample v2
            if v2 is None:
                v2 = uv <= v2mean
                # Accept v2 with probability 1-q(v2)/p(v2)
                # <=> Exp(1) < log[p(v2)] - log[q(v2)]
                logpv2 = nd.sum(nd.log(v2mean) * v2 + nd.log(1 - v2mean) * (1 - v2)).asscalar()
                logqv2 = nd.sum(nd.log(vc1mean) * v2 + nd.log(1 - vc1mean) * (1 - v2)).asscalar()
                u1 = nd.random.exponential().asscalar()
                if i < max_try - 1 and u1 >= logpv2 - logqv2:
                    v2 = None
            # Sample vc1
            if vc1 is None:
                vc1 = uv <= vc1mean
                # Accept vc1 with probability 1-p(vc1)/q(vc1)
                # <=> Exp(1) < log[q(vc1)] - log[p(vc1)]
                logpvc1 = nd.sum(nd.log(v2mean) * vc1 + nd.log(1 - v2mean) * (1 - vc1)).asscalar()
                logqvc1 = nd.sum(nd.log(vc1mean) * vc1 + nd.log(1 - vc1mean) * (1 - vc1)).asscalar()
                u2 = nd.random.exponential().asscalar()
                if i < max_try - 1 and u2 >= logqvc1 - logpvc1:
                    vc1 = None
            # Exit if v2 and vc1 have been set
            if v2 is not None and vc1 is not None:
                break

        # Sample h
        uh = nd.random.uniform(shape=self.n, ctx=self.ctx)
        h2mean = self.clip_sigmoid(nd.dot(self.w.T, v2) + self.c)
        hc1mean = self.clip_sigmoid(nd.dot(self.w.T, vc1) + self.c)
        h2 = uh <= h2mean
        hc1 = uh <= hc1mean

        return vc1, hc1, v2, h2, i

    # Unbiased sampling
    def sample(self, v0, min_steps=1, max_steps=100):
        # (v0, h0)   -> (v1, h1)   -> (v2, h2)   -> ... -> (vt, ht)
        # (vc0, hc0) -> (vc1, hc1) -> ... -> (vct, hct)
        # Init: (v0, h0) = (vc0, hc0)
        # Iter: (v1, h1, vc0, hc0) -> (v2, h2, vc1, hc1) -> ...
        # Stop: (vt, ht) = (vct, hct)
        vc = v0
        hc = self.sample_h_given_v(vc)
        v = self.sample_v_given_h(hc)
        h = self.sample_h_given_v(v)

        discarded = 0
        vhist = [v]
        vchist = []

        for i in range(max_steps):
            vc, hc, v, h, disc = self.max_coup(vc, hc, v, h, max_try=10)
            discarded += disc
            vhist.append(v)
            vchist.append(vc)
            if i >= min_steps - 1 and nd.norm(v - vc).asscalar() == 0 and nd.norm(h - hc).asscalar() == 0:
                break

        return nd.stack(*vhist), nd.stack(*vchist), discarded



if __name__ == "__main__":
    # Simulate an RBM model
    mx.random.seed(123)
    random.seed(123)
    ctx = mx.cpu()
    m = 100
    n = 50
    w = nd.random.normal(shape=(m, n), ctx=ctx)
    b = nd.random.normal(shape=m, ctx=ctx)
    c = nd.random.normal(shape=n, ctx=ctx)

    # Test CD-k sampler
    sampler0 = RBMSampler(w, b, c, ctx=ctx)
    nchain = 3
    v0 = nd.random.randint(0, 2, shape=(nchain, m), ctx=ctx).astype(w.dtype)
    res = sampler0.sample_k(v0, k=100)
    print(res)

    # Test unbiased MCMC sampler
    mx.random.seed(123)
    sampler = UnbiasedRBMSampler(w, b, c, ctx=ctx)
    v0 = nd.random.randint(0, 2, shape=m, ctx=ctx).astype(w.dtype)
    res = sampler.sample(v0, min_steps=1, max_steps=100)
    print(res)
