import numpy as np
import numpy.random as rng
import torch as th
from scipy.special import logsumexp
from scipy.stats import multivariate_normal

try:
    from itertools import izip as zip
except ImportError:  # will be 3.x series
    pass


class Gaussian:
    """Implements a gaussian pdf. Focus is on efficient multiplication, division
    and sampling.
    """

    def __init__(self, m=None, P=None, U=None, S=None, Pm=None, L=None):
        """Initialize a gaussian pdf given a valid combination of its
        parameters. Valid combinations are: m-P, m-U, m-S, Pm-P, Pm-U, Pm-S.

        :param m: mean
        :param P: precision
        :param U: upper triangular precision factor such that U'U = P
        :param S: covariance
        :param C : upper or lower triangular covariance factor, in any case
        S = C'C
        :param Pm: precision times mean such that P*m = Pm
        :param L: lower triangular covariance factor given as 1D array such that
        LL' = S
        """
        if m is not None:
            m = np.asarray(m)
            self.m = m
            self.ndim = m.size

            if P is not None:
                P = np.asarray(P)
                L = np.linalg.cholesky(P)
                self.P = P
                self.C = np.linalg.inv(L)
                self.S = np.dot(self.C.T, self.C)
                self.Pm = np.dot(P, m)
                self.logdetP = 2.0 * np.sum(np.log(np.diagonal(L)))

            elif U is not None:
                U = np.asarray(U)
                self.P = np.dot(U.T, U)
                self.C = np.linalg.inv(U.T)
                self.S = np.dot(self.C.T, self.C)
                self.Pm = np.dot(self.P, m)
                self.logdetP = 2.0 * np.sum(np.log(np.diagonal(U)))

            elif L is not None:
                L = np.asarray(L)
                Lm = np.zeros((self.ndim, self.ndim))
                idx_l = np.tril_indices(self.ndim, -1)
                idx_d = np.diag_indices(self.ndim)

                Lm[idx_l] = L[self.ndim :]
                Lm[idx_d] = L[0 : self.ndim]
                self.C = Lm.T
                self.S = np.dot(self.C.T, self.C)
                self.P = np.linalg.inv(self.S)
                self.Pm = np.dot(self.P, m)
                self.logdetP = -2.0 * np.sum(np.log(np.diagonal(self.C)))

            elif S is not None:
                S = np.asarray(S)
                self.P = np.linalg.inv(S)
                self.C = np.linalg.cholesky(S).T
                self.S = S
                self.Pm = np.dot(self.P, m)
                self.logdetP = -2.0 * np.sum(np.log(np.diagonal(self.C)))

            else:
                raise ValueError("Precision information missing.")

        elif Pm is not None:
            Pm = np.asarray(Pm)
            self.Pm = Pm
            self.ndim = Pm.size

            if P is not None:
                P = np.asarray(P)
                L = np.linalg.cholesky(P)
                self.P = P
                self.C = np.linalg.inv(L)
                self.S = np.dot(self.C.T, self.C)
                self.m = np.linalg.solve(P, Pm)
                self.logdetP = 2.0 * np.sum(np.log(np.diagonal(L)))

            elif U is not None:
                U = np.asarray(U)
                self.P = np.dot(U.T, U)
                self.C = np.linalg.inv(U.T)
                self.S = np.dot(self.C.T, self.C)
                self.m = np.linalg.solve(self.P, Pm)
                self.logdetP = 2.0 * np.sum(np.log(np.diagonal(U)))

            elif S is not None:
                S = np.asarray(S)
                self.P = np.linalg.inv(S)
                self.C = np.linalg.cholesky(S).T
                self.S = S
                self.m = np.dot(S, Pm)
                self.logdetP = -2.0 * np.sum(np.log(np.diagonal(self.C)))

            else:
                raise ValueError("Precision information missing.")

        else:
            raise ValueError("Mean information missing.")

    @property
    def mean(self):
        """Returns the Gaussian mean."""
        return self.m

    @property
    def covariance(self):
        """Returns the Gaussian covariance."""
        return self.S

    def gen(self, n_samples=1):
        """Returns independent samples from the gaussian."""
        z = rng.randn(n_samples, self.ndim)
        samples = np.dot(z, self.C) + self.m
        return samples

    def sample(self, n_samples=1):
        """Returns `n_samples` from the Gaussian in a vector."""
        samples = self.gen(n_samples)
        return samples

    def eval(self, x, ii=None, log=True):
        """Evaluates the gaussian pdf.

        :param x: rows are inputs to evaluate at
        :param ii: a list of indices specifying which marginal to evaluate. If
        None, the joint pdf is evaluated
        :param log: if True, the log pdf is evaluated
        :return: pdf or log pdf
        """
        if ii is None:
            xm = x - self.m
            lp = -np.sum(np.dot(xm, self.P) * xm, axis=1)
            lp += self.logdetP - self.ndim * np.log(2.0 * np.pi)
            lp *= 0.5

        else:
            m = self.m[ii]
            S = self.S[ii][:, ii]
            lp = multivariate_normal.logpdf(x, m, S)
            lp = np.array([lp]) if x.shape[0] == 1 else lp

        res = lp if log else np.exp(lp)
        return res

    def __mul__(self, other):
        """Multiply with another gaussian."""
        assert isinstance(other, Gaussian)
        P = self.P + other.P
        Pm = self.Pm + other.Pm
        return Gaussian(P=P, Pm=Pm)

    def __imul__(self, other):
        """Incrementally multiply with another gaussian."""
        assert isinstance(other, Gaussian)
        res = self * other
        self.m = res.m
        self.P = res.P
        self.C = res.C
        self.S = res.S
        self.Pm = res.Pm
        self.logdetP = res.logdetP
        return res

    def __truediv__(self, other):
        """Divide by another gaussian. Note that the resulting gaussian might be
        improper."""
        assert isinstance(other, Gaussian)
        P = self.P - other.P
        Pm = self.Pm - other.Pm
        return Gaussian(P=P, Pm=Pm)

    def __div__(self, other):
        """For backwards compatibility with Python 2.7"""
        return self.__truediv__(other)

    def __idiv__(self, other):
        """Incrementally divide by another gaussian. Note that the resulting
        gaussian might be improper."""
        assert isinstance(other, Gaussian)
        res = self / other
        self.m = res.m
        self.P = res.P
        self.C = res.C
        self.S = res.S
        self.Pm = res.Pm
        self.logdetP = res.logdetP
        return res

    def __pow__(self, power, modulo=None):
        """Raise gaussian to a power and get another gaussian."""
        P = power * self.P
        Pm = power * self.Pm
        return Gaussian(P=P, Pm=Pm)

    def __ipow__(self, power):
        """Incrementally raise gaussian to a power."""
        res = self ** power
        self.m = res.m
        self.P = res.P
        self.C = res.C
        self.S = res.S
        self.Pm = res.Pm
        self.logdetP = res.logdetP
        return res

    def kl(self, other):
        """Calculates the kl divergence from this to another gaussian, i.e.
        KL(this | other)."""
        assert isinstance(other, Gaussian)
        assert self.ndim == other.ndim
        t1 = np.sum(other.P * self.S)
        m = other.m - self.m
        t2 = np.dot(m, np.dot(other.P, m))
        t3 = self.logdetP - other.logdetP
        t = 0.5 * (t1 + t2 + t3 - self.ndim)
        return t


class MoG:
    """Implements a mixture of Gaussians."""

    def __init__(self, a, ms=None, Ps=None, Us=None, Ss=None, xs=None, Ls=None):
        """Creates a mog with a valid combination of parameters or an already
        given list of gaussian variables.
        :param a: mixing coefficients
        :param ms: means
        :param Ps: precisions
        :param Us: precision factors such that U'U = P
        :param Ss: covariances
        :param xs: list of gaussian variables
        :param Ls: lower-triangular covariance factor such that L*L' = S
        """
        if ms is not None:

            if Ps is not None:
                self.xs = [Gaussian(m=m, P=P) for m, P in zip(ms, Ps)]

            elif Us is not None:
                self.xs = [Gaussian(m=m, U=U) for m, U in zip(ms, Us)]

            elif Ss is not None:
                self.xs = [Gaussian(m=m, S=S) for m, S in zip(ms, Ss)]

            elif Ls is not None:
                self.xs = [Gaussian(m=m, L=L) for m, L in zip(ms, Ls)]

            else:
                raise ValueError("Precision information missing.")

        elif xs is not None:
            self.xs = xs

        else:
            raise ValueError("Mean information missing.")

        self.a = np.asarray(a)
        self.ndim = self.xs[0].ndim
        self.n_components = len(self.xs)
        self.ncomp = self.n_components
        self.__gaussian = self.project_to_gaussian()

    @property
    def mean(self):
        """Returns the mean of the projected Gaussian."""
        return self.__gaussian.m

    @property
    def covariance(self):
        """Returns the covariance of the projected Gaussian"""
        return self.__gaussian.S

    def gen(self, n_samples=1):
        """Generates independent samples from mog."""
        ii = discrete_sample(self.a, n_samples)
        ns = [np.sum((ii == i).astype(int)) for i in range(self.n_components)]
        samples = [x.gen(n_samples=n) for x, n in zip(self.xs, ns)]
        samples = np.concatenate(samples, axis=0)
        # Uncomment the lines below another possible way of sampling from MoG
        # weighted_sample = [sample * weight for sample, weight in zip(samples,
        # self.a)]
        # sample = sum(weighted_sample)
        return samples

    def sample(self, sample_shape=(1,)):
        """Returns `n_samples` from the MoG in a tensor."""
        samples = self.gen(sample_shape[0])
        return th.as_tensor(samples, dtype=th.float)

    def eval(self, x, ii=None, log=True):
        """Evaluates the mog pdf.

        :param x: rows are inputs to evaluate at
        :param ii: a list of indices specifying which marginal to evaluate. if
        None, the joint pdf is evaluated
        :param log: if True, the log pdf is evaluated
        :return: pdf or log pdf
        """
        ps = np.array(
            [self.xs[ix].eval(x, ii, log) for ix in range(len(self.a))]
        ).T
        res = (
            logsumexp(ps + np.log(self.a), axis=1)
            if log
            else np.dot(ps, self.a)
        )
        return res

    def __mul__(self, other):
        """Multiply with a single gaussian."""
        assert isinstance(other, Gaussian)
        ys = [x * other for x in self.xs]
        lcs = np.empty_like(self.a)
        for i, (x, y) in enumerate(zip(self.xs, ys)):
            lcs[i] = x.logdetP + other.logdetP - y.logdetP
            lcs[i] -= (
                np.dot(x.m, np.dot(x.P, x.m))
                + np.dot(other.m, np.dot(other.P, other.m))
                - np.dot(y.m, np.dot(y.P, y.m))
            )
            lcs[i] *= 0.5
        la = np.log(self.a) + lcs
        la -= logsumexp(la)
        a = np.exp(la)
        return MoG(a=a, xs=ys)

    def __imul__(self, other):
        """Incrementally multiply with a single gaussian."""
        assert isinstance(other, Gaussian)
        res = self * other
        self.a = res.a
        self.xs = res.xs
        return res

    def __truediv__(self, other):
        """Divide by a single gaussian."""
        assert isinstance(other, Gaussian)
        ys = [x / other for x in self.xs]
        lcs = np.empty_like(self.a)
        for i, (x, y) in enumerate(zip(self.xs, ys)):
            lcs[i] = x.logdetP - other.logdetP - y.logdetP
            lcs[i] -= (
                np.dot(x.m, np.dot(x.P, x.m))
                - np.dot(other.m, np.dot(other.P, other.m))
                - np.dot(y.m, np.dot(y.P, y.m))
            )
            lcs[i] *= 0.5
        la = np.log(self.a) + lcs
        la -= logsumexp(la)
        a = np.exp(la)
        return MoG(a=a, xs=ys)

    def __div__(self, other):
        """For backwards compatibility with Python 2.7."""
        return self.__truediv__(other)

    def __idiv__(self, other):
        """Incrementally divide by a single gaussian."""
        assert isinstance(other, Gaussian)
        res = self / other
        self.a = res.a
        self.xs = res.xs
        return res

    def _get_summary_stats_from_mog(self):
        """Computes the mean and covariance of a MoG.

        :returns (mu, S): x is a tensor of size [n] with the grand mean and P a
        tensor of size [n, n] with the covariances.
        """
        alpha = self.a
        res_mean = np.zeros((self.ncomp, self.ndim))
        mu = np.zeros(self.ndim)
        S = np.zeros((self.ndim, self.ndim))
        # We'll decompose the MoG into x = y + N(0, S), where y will be the
        # weighted mean and the covariance is given by the law of total variance
        # S = E(var(x∣y))+var(E(x∣y))
        for idx, gauss in enumerate(self.xs):
            mu += alpha[idx] * gauss.m
            S += alpha[idx] * gauss.C  # first term of covariance matrix
            res_mean[idx] = gauss.m
        res_mean -= mu
        S += np.matmul(
            (alpha * res_mean.T), res_mean
        )  # add second term of covariance
        return mu, S

    def project_to_gaussian(self):
        """Returns a gaussian with the same mean and precision as self."""
        m, S = self._get_summary_stats_from_mog()
        return Gaussian(m, S)

    def prune_negligible_components(self, threshold):
        """Removes all the components whose mixing coefficient is less than a
        threshold."""
        ii = np.nonzero((self.a < threshold).astype(int))[0]
        total_del_a = np.sum(self.a[ii])
        del_count = ii.size
        self.n_components -= del_count
        self.a = np.delete(self.a, ii)
        self.a += total_del_a / self.n_components
        self.xs = [x for i, x in enumerate(self.xs) if i not in ii]

    def kl(self, other, n_samples=10000):
        """Estimates the kl from this to another pdf, i.e. KL(this | other),
        using monte carlo."""
        x = self.gen(n_samples)
        lp = self.eval(x, log=True)
        lq = other.eval(x, log=True)
        t = lp - lq
        res = np.mean(t)
        err = np.std(t, ddof=1) / np.sqrt(n_samples)
        return res, err


def discrete_sample(p, n_samples=1):
    """Samples from a discrete distribution.

    :param p: a distribution with N elements
    :param n_samples: number of samples
    :return: vector of samples """
    # cumulative distribution
    c = np.cumsum(p[:-1])[np.newaxis, :]
    # get the samples
    r = rng.rand(n_samples, 1)
    return np.sum((r > c).astype(int), axis=1)
