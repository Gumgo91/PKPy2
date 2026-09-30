"""Importance-sampled marginal likelihood of a general model with fixed particles.

For each subject, particles b = (phi, kappa) are drawn once per bank from an
equal mixture of the population prior at the bank's population point and a
subject-specific Gaussian (scrambled Sobol points, equal counts per component;
weights use the full mixture density). The subject component starts at the
Laplace mode with covariance 2 H^-1 + 0.01 prior and is adapted twice to the
weighted mean and 1.5 times the weighted covariance of a pilot sample of 2^10
particles from the current mixture. The
particles stay fixed while the population parameters change: density
coordinates only reweight them, and prediction coordinates recompute the
predictions of the same particles. The objective is
OFV = -2 sum_i log( mean_k p(y_i | b_k) p(b_k) / q_i(b_k) ).
"""
import math
import numpy as np
from scipy.linalg import solve_triangular
from scipy.special import logsumexp, ndtri
from scipy.stats import qmc
from .laplace import SubjectContext, LOG2PI


def _mvn_logpdf(b, mean, cov):
    chol = np.linalg.cholesky(cov)
    z = solve_triangular(chol, (b - mean).T, lower=True)
    return -.5 * (len(mean) * LOG2PI + 2. * np.sum(np.log(np.diag(chol))) + np.sum(z * z, axis=0))


def _sobol_normal(d, power, seed):
    u = qmc.Sobol(d, scramble=True, seed=np.random.default_rng(seed)).random_base2(power)
    return ndtri(np.clip(u, np.finfo(float).eps, 1. - np.finfo(float).eps))


class Bank:
    def __init__(self, problem, x, *, power=12, seed=0, states=None, powers=None, adapt_rounds=2, pilot_power=10):
        from .laplace import LaplaceObjective
        self.problem = problem
        self.x_ref = np.array(x, dtype=float)
        pop = problem.population(x)
        if states is None:
            _, states = LaplaceObjective(problem).states(x)
            if states is None:
                raise ValueError('population point outside the model domain')
        self.states = states
        self.particles = []
        self.logq = []
        self.powers = np.full(problem.N, power, dtype=int) if powers is None else np.asarray(powers, dtype=int)
        self.seed = seed
        for i in range(problem.N):
            ctx = SubjectContext(problem, i, pop)
            d = ctx.d
            if d == 0:
                self.particles.append(np.zeros((1, 0)))
                self.logq.append(np.zeros(1))
                continue
            prior_cov = np.zeros((d, d))
            prior_cov[:ctx.q, :ctx.q] = pop.omega
            if d > ctx.q:
                prior_cov[ctx.q:, ctx.q:] = np.diag(ctx.iov_var)
            st = states[i]
            try:
                post_cov = 2. * np.linalg.inv(st.H) + .01 * prior_cov
                np.linalg.cholesky(post_cov)
                post_mean = st.b
            except np.linalg.LinAlgError:
                post_cov, post_mean = prior_cov, ctx.m0
            p = int(self.powers[i])
            streams = np.random.SeedSequence([int(seed) % (2 ** 63), i]).generate_state(2 + 2 * adapt_rounds)
            # Adaptive proposal: the subject component is moved to the weighted mean and 1.5 times the
            # weighted covariance of a pilot sample drawn from the current mixture.
            for r in range(adapt_rounds):
                pp = min(p, pilot_power)
                u1 = _sobol_normal(d, pp - 1, int(streams[2 + 2 * r]))
                u2 = _sobol_normal(d, pp - 1, int(streams[3 + 2 * r]))
                bp = np.vstack([ctx.m0 + u1 @ np.linalg.cholesky(prior_cov).T,
                                post_mean + u2 @ np.linalg.cholesky(post_cov).T])
                lq = np.logaddexp(_mvn_logpdf(bp, ctx.m0, prior_cov), _mvn_logpdf(bp, post_mean, post_cov)) - math.log(2.)
                ll, _, _ = ctx.obs.loglik(ctx.predict(bp))
                lp, _ = ctx.log_prior(bp)
                lw = np.where(np.isfinite(ll + lp), ll + lp - lq, -np.inf)
                if not np.any(np.isfinite(lw)):
                    break
                w = np.exp(lw - logsumexp(lw))
                if 1. / np.dot(w, w) < 5.:
                    break
                mean = w @ bp
                cov = (bp - mean).T @ ((bp - mean) * w[:, None])
                cov = 1.5 * .5 * (cov + cov.T) + .01 * prior_cov
                try:
                    np.linalg.cholesky(cov)
                except np.linalg.LinAlgError:
                    break
                post_mean, post_cov = mean, cov
            z1 = _sobol_normal(d, p - 1, int(streams[0]))
            z2 = _sobol_normal(d, p - 1, int(streams[1]))
            b = np.vstack([ctx.m0 + z1 @ np.linalg.cholesky(prior_cov).T,
                           post_mean + z2 @ np.linalg.cholesky(post_cov).T])
            lq = np.logaddexp(_mvn_logpdf(b, ctx.m0, prior_cov), _mvn_logpdf(b, post_mean, post_cov)) - math.log(2.)
            b.setflags(write=False)
            self.particles.append(b)
            self.logq.append(lq)
        self.pred = [None] * problem.N
        self.pred_key = None
        self.last = None

    # ------------------------------------------------------------ predictions
    def _key(self, x, pop):
        pc = self.problem.prediction_coords
        return tuple(np.asarray(x)[pc].tolist())

    def _predictions(self, x, pop):
        key = self._key(x, pop)
        if key == self.pred_key and self.pred[0] is not None:
            return key, self.pred
        out = []
        for i in range(self.problem.N):
            ctx = SubjectContext(self.problem, i, pop)
            out.append(ctx.predict(self.particles[i]))
        return key, out

    def set_predictions(self, x):
        pop = self.problem.population(x)
        key, pred = self._predictions(x, pop)
        self.pred_key, self.pred = key, pred
        return pop

    # ------------------------------------------------------------ objective
    def subject_terms(self, x, pred=None, scores=False):
        """Per-subject OFV, ESS, normalized weights and density scores at x (None outside the valid domain)."""
        try:
            with np.errstate(over='ignore', invalid='ignore', divide='ignore'):
                return self._subject_terms(x, pred, scores)
        except (ArithmeticError, ValueError, np.linalg.LinAlgError):
            return None

    def _subject_terms(self, x, pred=None, scores=False):
        problem = self.problem
        pop = problem.population(x)
        if not problem.valid(pop):
            return None
        if pred is None:
            key = self._key(x, pop)
            if key != self.pred_key or self.pred[0] is None:
                key, fresh = self._predictions(x, pop)
                self.pred_key, self.pred = key, fresh
            pred = self.pred
        rows = []
        for i in range(problem.N):
            ctx = SubjectContext(problem, i, pop)
            b = self.particles[i]
            ll, dv, dvds = ctx.obs.loglik(pred[i], scores=scores)
            lp, u = ctx.log_prior(b)
            lw = ll + lp - self.logq[i]
            total = logsumexp(lw)
            if not np.isfinite(total):
                return None
            w = np.exp(lw - total)
            ofv = -2. * (total - math.log(len(lw)))
            row = dict(ofv=float(ofv), ess=float(1. / np.dot(w, w)), weights=w)
            if scores:
                row['score'] = self._density_score(ctx, b, w, u, dv, dvds)
            rows.append(row)
        return rows

    def _density_score(self, ctx, b, w, u, dv, dvds):
        """d OFV_i / dx for density coordinates (zeros elsewhere)."""
        problem = self.problem
        g = np.zeros(len(problem.labels))
        pop = ctx.pop
        q = ctx.q
        if q:
            L = pop.chol
            # Omega^-1 (phi - mu) = L^-T u
            oinv = solve_triangular(L.T, u, lower=False)          # (q, K)
            mean_oinv = oinv @ w                                  # (q,)
            for a, j in enumerate(problem.iiv):
                c = problem.theta_coord[j]
                if c >= 0:
                    g[c] = -2. * mean_oinv[a]
            for k in problem.tic_effects:
                c = problem.beta_coord[k]
                e = problem.effects[k]
                if c >= 0 and e['parameter'] in problem.iiv:
                    a = problem.iiv.index(e['parameter'])
                    deriv = problem.effect_derivative(k, problem.z_tic[ctx.i, k], pop.beta[k])
                    g[c] = -2. * mean_oinv[a] * deriv
            S = (u * w) @ u.T
            G = -np.diag(1. / np.diag(L)) + solve_triangular(L.T, S, lower=False)
            for row, col, kind, coord, _ in problem.chol_plan:
                if coord < 0:
                    continue
                g[coord] = -2. * (G[row, col] * L[row, col] if kind == 'log' else G[row, col])
        if problem.Q and ctx.O:
            kap = b[:, q:].reshape(len(w), ctx.O, problem.Q)
            s2 = pop.iov_sd ** 2
            t = np.sum(kap ** 2 / s2 - 1., axis=1)               # (K, Q)
            mean_t = w @ t
            for j in range(problem.Q):
                c = problem.iov_coord[j]
                if c >= 0:
                    g[c] = -2. * mean_t[j]
        if dv is not None and dv.shape[1]:
            out = ctx.obs.out
            for o in range(problem.n_out):
                mask = out == o
                if not np.any(mask):
                    continue
                for comp in range(3):
                    c = problem.sigma_coord[o, comp]
                    if c < 0:
                        continue
                    per = np.sum(dv[:, mask] * dvds[:, mask, comp], axis=1)
                    g[c] = -2. * float(w @ per)
        return g

    def evaluate(self, x, *, gradient=True, per_subject=False, fd_step=1e-4, differences='central',
                 finite_differences=True):
        """OFV and its gradient at x; per_subject also returns subject OFVs and score rows.

        finite_differences=False returns only the analytic (density-coordinate) part of the
        gradient; prediction-coordinate components are then zero.
        """
        x = np.asarray(x, dtype=float)
        rows = self.subject_terms(x, scores=gradient)
        if rows is None:
            return (np.inf, np.zeros_like(x)) if not per_subject else (np.inf, np.zeros_like(x), None)
        value = float(sum(r['ofv'] for r in rows))
        self.last = dict(x=x.copy(), rows=rows, min_ess=min(r['ess'] for r in rows))
        if not gradient:
            return (value, None) if not per_subject else (value, None, rows)
        per = np.array([r['score'] for r in rows])
        grad = per.sum(axis=0)
        problem = self.problem
        saved_key, saved_pred = self.pred_key, self.pred
        for j in (problem.prediction_coords if finite_differences else ()):
            h = fd_step * max(1., abs(x[j]))
            lo, hi = problem.bounds[j]
            plus, minus = x.copy(), x.copy()
            plus[j] = x[j] + h if hi is None else min(x[j] + h, hi)
            minus[j] = x[j] - h if lo is None else max(x[j] - h, lo)
            if differences == 'forward':
                minus = x.copy()
            span = plus[j] - minus[j]
            vals = []
            for point in (plus, minus):
                if differences == 'forward' and point is minus:
                    vals.append(np.array([q['ofv'] for q in rows]))
                    continue
                pop = problem.population(point)
                if not problem.valid(pop):
                    vals.append(None)
                    continue
                _, pred = self._predictions(point, pop)
                r = self.subject_terms(point, pred=pred)
                vals.append(None if r is None else np.array([q['ofv'] for q in r]))
            self.pred_key, self.pred = saved_key, saved_pred
            if vals[0] is None or vals[1] is None or span <= 0:
                continue
            col = (vals[0] - vals[1]) / span
            per[:, j] = col
            grad[j] = col.sum()
        self.last['per_subject_score'] = per
        return (value, grad) if not per_subject else (value, grad, rows)
