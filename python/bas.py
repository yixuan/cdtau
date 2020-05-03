import numpy as np
import mxnet as mx
from mxnet import nd
from rbm import RBM
import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
import seaborn as sns

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


# The bars-and-stripes data studied by Schulz et al. (2010)
mx.random.seed(123)
np.random.seed(123)
d = 4
m = d * d
N = 2 ** d
dat = nd.empty(shape=(2 * N, m))
bits = vec_bin_array(np.arange(N), d).astype(np.float)
for i in range(N):
    mat = bits[i, :].reshape(1, -1).T.dot(np.ones(shape=(1, d)))
    dat[2 * i, :] = mat.flatten()
    dat[2 * i + 1, :] = mat.T.flatten()

ctx = mx.cpu()
dat = nd.array(dat, ctx=ctx)
N = dat.shape[0]
m = d * d
n = m

# Train RBM using CD-k
mx.random.seed(123)
np.random.seed(123)
cd1 = RBM(m, n, ctx=ctx)
res_cd1 = cd1.train_cdk(dat, batch_size=N, epochs=2000, lr=0.1,
                        k=1, nchain=1000,
                        report_freq=1, exact_loglik=True)
# np.savetxt("cd1.txt", np.array(res_cd1))

mx.random.seed(123)
np.random.seed(123)
ucd = RBM(m, n, ctx=ctx)
res_ucd = ucd.train_ucd(dat, batch_size=N, epochs=2000, lr=0.1,
                        min_mcmc=1, max_mcmc=100, nchain=1000,
                        report_freq=1, exact_loglik=True)
# np.savetxt("ucd_loglik.txt", np.array(res_ucd[0]))
# np.savetxt("ucd_tau.txt", np.array(res_ucd[1]))
# np.savetxt("ucd_disc.txt", np.array(res_ucd[2]))



fig = plt.figure()
sub = fig.add_subplot(131)
n = len(res_cd1)
sns.lineplot(np.arange(n), res_ucd[0], label="Unbiased CD")
sns.lineplot(np.arange(n), res_cd1, label="CD")
sub.set_xlabel("Iteration")
sub.set_ylabel("Log-likelihood Function Value")

sub = fig.add_subplot(132)
sns.lineplot(np.arange(n), res_ucd[1])
sub.set_xlabel("Iteration")
sub.set_ylabel("Average Stopping Time")

sub = fig.add_subplot(133)
sns.lineplot(np.arange(n), res_ucd[2])
sub.set_xlabel("Iteration")
sub.set_ylabel("# Rejected Samples per Chain")

fig.show()
