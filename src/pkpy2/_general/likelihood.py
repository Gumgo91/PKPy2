"""Observation log-likelihood of one subject for many particles.

Combined residuals: y ~ N(f, sp^2 f^2 + sa^2). Lognormal residuals: log y ~
N(log f, sl^2) (log-transform-both-sides; the Jacobian of the transform is not
included, as in NONMEM LTBS). Censored observations (CENS = 1: below DV; CENS = -1:
above DV; LIMIT: the other end of the interval) contribute log P(censoring
interval) (Beal's M3, or M4 with a LIMIT).
"""
import math
import numpy as np
from scipy.special import log_ndtr

LOG2PI = math.log(2. * math.pi)
VAR_FLOOR = 1e-12


def _log_phi(z):
    return -.5 * (LOG2PI + z * z)


def _log_diff(a, b):
    """log(Phi(a) - Phi(b)) for a > b, computed stably."""
    la, lb = log_ndtr(a), log_ndtr(b)
    with np.errstate(divide='ignore', invalid='ignore'):
        return la + np.log1p(-np.exp(np.minimum(lb - la, 0.)))


class ObservationModel:
    """Precomputed per-subject observation arrays and the residual SDs in force."""

    def __init__(self, subject, pop):
        used = subject.obs_used
        self.index = np.nonzero(used)[0]
        out = subject.obs_out[used]
        self.y = subject.obs_dv[used]
        self.cens = subject.obs_cens[used]
        self.limit = subject.obs_limit[used]
        self.form = pop.form[out]
        self.out = out
        self.sp = pop.sigma_prop[out]
        self.sa = pop.sigma_add[out]
        self.sl = pop.sigma_ln[out]
        self.lognormal = self.form == 1
        self.logy = np.where(self.lognormal & (self.y > 0), np.log(np.where(self.y > 0, self.y, 1.)), 0.)
        if np.any(self.lognormal & (self.y <= 0)):
            raise ValueError('lognormal residuals need positive observations (and positive LLOQs)')
        self.uncensored = self.cens == 0
        self.left = self.cens == 1
        self.right = self.cens == -1
        self.has_limit = np.isfinite(self.limit) & (self.cens != 0)
        if np.any(self.lognormal & self.has_limit):
            ok = self.limit > 0
            self.limit_t = np.where(self.lognormal, np.log(np.where(ok, self.limit, 1.)), self.limit)
            self.has_limit = self.has_limit & (~self.lognormal | ok)
        else:
            self.limit_t = self.limit

    @property
    def m(self):
        return len(self.y)

    def moments(self, f):
        """Mean and variance on the modeling scale for predictions f (K, m)."""
        if np.any(self.lognormal):
            safe = np.where(f > 0, f, np.nan)
            lf = np.log(safe)
            mean = np.where(self.lognormal, lf, f)
        else:
            mean = f
        v = np.where(self.lognormal, self.sl ** 2, (self.sp * f) ** 2 + self.sa ** 2)
        return mean, np.maximum(v, VAR_FLOOR)

    def loglik(self, pred, *, scores=False):
        """ll (K,) and, if scores, dll/dv (K, m) and dv/dlogsigma for (prop, add, lognormal) (K, m, 3)."""
        f = pred[:, self.index]
        mean, v = self.moments(f)
        target = np.where(self.lognormal, self.logy, self.y)
        sd = np.sqrt(v)
        z = (target - mean) / sd
        ll = np.where(self.uncensored, -.5 * (LOG2PI + np.log(v) + z * z), 0.)
        zl = np.where(self.has_limit, (self.limit_t - mean) / sd, 0.)
        if np.any(self.left):
            one = log_ndtr(z)
            two = _log_diff(z, zl)
            ll = ll + np.where(self.left & ~self.has_limit, one, 0.) + np.where(self.left & self.has_limit, two, 0.)
        if np.any(self.right):
            one = log_ndtr(-z)
            two = _log_diff(zl, z)
            ll = ll + np.where(self.right & ~self.has_limit, one, 0.) + np.where(self.right & self.has_limit, two, 0.)
        total = np.sum(ll, axis=1)
        total = np.where(np.isfinite(total), total, -np.inf)
        if not scores:
            return total, None, None
        dv = np.where(self.uncensored, -.5 * (1. / v - (target - mean) ** 2 / v ** 2), 0.)
        if np.any(~self.uncensored):
            dv = dv + self._censored_dv(z, zl, v)
        dvds = np.zeros(f.shape + (3,))
        dvds[..., 0] = np.where(self.lognormal, 0., 2. * (self.sp * f) ** 2)
        dvds[..., 1] = np.where(self.lognormal, 0., 2. * self.sa ** 2)
        dvds[..., 2] = np.where(self.lognormal, 2. * self.sl ** 2, 0.)
        dv = np.where(np.isfinite(dv), dv, 0.)
        return total, dv, dvds

    def _censored_dv(self, z, zl, v):
        with np.errstate(over='ignore', invalid='ignore', divide='ignore'):
            out = np.zeros_like(z)
            # left, single: d/dv log Phi(z) = lambda(z) * (-z/(2v))
            lam = np.exp(_log_phi(z) - log_ndtr(z))
            out = np.where(self.left & ~self.has_limit, lam * (-.5 * z / v), out)
            lam_r = np.exp(_log_phi(z) - log_ndtr(-z))
            out = np.where(self.right & ~self.has_limit, -lam_r * (-.5 * z / v), out)
            # intervals
            left_den = _log_diff(z, zl)
            a = np.exp(_log_phi(z) - left_den) * (-.5 * z / v)
            b = np.exp(_log_phi(zl) - left_den) * (-.5 * zl / v)
            out = np.where(self.left & self.has_limit, a - b, out)
            right_den = _log_diff(zl, z)
            a = np.exp(_log_phi(zl) - right_den) * (-.5 * zl / v)
            b = np.exp(_log_phi(z) - right_den) * (-.5 * z / v)
            out = np.where(self.right & self.has_limit, a - b, out)
        return out

    def gradient_weights(self, pred):
        """For the conditional-mode search: dll/df (K, m) and a positive curvature weight w (K, m)."""
        f = pred[:, self.index]
        mean, v = self.moments(f)
        target = np.where(self.lognormal, self.logy, self.y)
        sd = np.sqrt(v)
        r = target - mean
        z = r / sd
        with np.errstate(divide='ignore', invalid='ignore', over='ignore'):
            dmean = np.where(self.lognormal, 1. / f, 1.)              # d mean / d f
            dvdf = np.where(self.lognormal, 0., 2. * self.sp ** 2 * f)
            dll_dv = -.5 * (1. / v - r * r / v ** 2)
            grad = np.where(self.uncensored, r / v * dmean + dll_dv * dvdf, 0.)
            weight = np.where(self.uncensored, dmean ** 2 / v + .5 * (dvdf / v) ** 2, 0.)
            if np.any(~self.uncensored):
                zl = np.where(self.has_limit, (self.limit_t - mean) / sd, 0.)
                dv_c = self._censored_dv(z, zl, v)
                # d ll / d mean at fixed v
                lam = np.exp(_log_phi(z) - log_ndtr(z))
                lam_r = np.exp(_log_phi(z) - log_ndtr(-z))
                left_den = _log_diff(z, zl)
                right_den = _log_diff(zl, z)
                d_left_int = -(np.exp(_log_phi(z) - left_den) - np.exp(_log_phi(zl) - left_den)) / sd
                d_right_int = -(np.exp(_log_phi(zl) - right_den) - np.exp(_log_phi(z) - right_den)) / sd
                dmu = np.where(self.left & ~self.has_limit, -lam / sd,
                      np.where(self.right & ~self.has_limit, lam_r / sd,
                      np.where(self.left, d_left_int, d_right_int)))
                curv = np.where(self.left & ~self.has_limit, lam * (z + lam),
                       np.where(self.right & ~self.has_limit, lam_r * (lam_r - z), 1.)) / v
                grad = np.where(self.uncensored, grad, dmu * dmean + dv_c * dvdf)
                weight = np.where(self.uncensored, weight, np.maximum(curv, 1e-12) * dmean ** 2)
        grad = np.where(np.isfinite(grad), grad, 0.)
        weight = np.where(np.isfinite(weight), weight, 0.)
        return grad, weight
