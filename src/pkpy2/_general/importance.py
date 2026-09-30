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

With problem.integration['proposal'] == 'mixture', the subject half of the
particles comes from a mixture of multivariate t components (3 degrees of
freedom) instead of one Gaussian: the pilot samples are searched for heavy or
high-density particles that no component explains; a Newton solve from such a
particle adds a component at a new conditional mode (bimodal posteriors) or, when
it returns to a known mode, a component centred on the particle (long tails).
Components whose joint density is more than 10 below the best mode are dropped,
and the components are adapted by mixture population Monte Carlo (each moves to
its responsibility-weighted mean and 1.5 times its covariance; shares follow the
weighted mass, at least 0.1 each).
"""
import math
import numpy as np
from scipy.linalg import solve_triangular
from scipy.special import gammaln, logsumexp, ndtri
from scipy.stats import chi2, qmc
from .laplace import SubjectContext, LOG2PI


def _mvn_logpdf(b, mean, cov):
    chol = np.linalg.cholesky(cov)
    z = solve_triangular(chol, (b - mean).T, lower=True)
    return -.5 * (len(mean) * LOG2PI + 2. * np.sum(np.log(np.diag(chol))) + np.sum(z * z, axis=0))


def _sobol_normal(d, power, seed):
    u = qmc.Sobol(d, scramble=True, seed=np.random.default_rng(seed)).random_base2(power)
    return ndtri(np.clip(u, np.finfo(float).eps, 1. - np.finfo(float).eps))


T_DF = 3.


def _t_logpdf(b, mean, cov, df=T_DF):
    d = len(mean)
    chol = np.linalg.cholesky(cov)
    z = solve_triangular(chol, (np.atleast_2d(b) - mean).T, lower=True)
    m2 = np.sum(z * z, axis=0)
    return (gammaln(.5 * (df + d)) - gammaln(.5 * df) - .5 * d * math.log(df * math.pi)
            - np.sum(np.log(np.diag(chol))) - .5 * (df + d) * np.log1p(m2 / df))


def _mahalanobis(b, mean, cov):
    diff = np.atleast_2d(b) - mean
    return np.sqrt(np.sum(diff * np.linalg.solve(cov, diff.T).T, axis=1))


def _mixture_draw(d, power, m0, prior_cov, components, shares, seed_prior, seed_mix):
    """2^power particles: half from the prior, half split over the t components in proportion to their shares;
    returns the particles and the log density of the full mixture."""
    n = 2 ** (power - 1)
    z1 = _sobol_normal(d, power - 1, seed_prior)
    u = qmc.Sobol(d + 1, scramble=True, seed=np.random.default_rng(seed_mix)).random_base2(power - 1)
    u = np.clip(u, np.finfo(float).eps, 1. - np.finfo(float).eps)
    z2 = ndtri(u[:, :d]) / np.sqrt(chi2.ppf(u[:, d], T_DF) / T_DF)[:, None]
    counts = np.floor(np.asarray(shares) * n).astype(int)
    counts[int(np.argmax(shares))] += n - int(counts.sum())
    blocks, used, start = [m0 + z1 @ np.linalg.cholesky(prior_cov).T], [], 0
    for c, k in zip(components, counts):
        if k > 0:
            blocks.append(c['mean'] + z2[start:start + k] @ np.linalg.cholesky(c['cov']).T)
            used.append((c, k / n))
        start += k
    b = np.vstack(blocks)
    parts = [math.log(.5) + _mvn_logpdf(b, m0, prior_cov)]
    parts += [math.log(.5 * share) + _t_logpdf(b, c['mean'], c['cov']) for c, share in used]
    return b, logsumexp(np.vstack(parts), axis=0)


def _mixture_components(ctx, prior_cov, state, streams, power, pilot_power, rounds, max_components=4,
                        delta_j=10., top=8, tail_weight=.01):
    """Components and shares of the t-mixture proposal of one subject (see the module docstring)."""
    d = ctx.d
    try:
        cov = 2. * np.linalg.inv(state.H) + .01 * prior_cov
        np.linalg.cholesky(cov)
        components = [dict(mean=np.array(state.b, dtype=float), cov=cov, J=float(state.J))]
    except np.linalg.LinAlgError:
        components = [dict(mean=ctx.m0.copy(), cov=prior_cov.copy(), J=np.inf)]
    shares = [1.]
    tau = math.sqrt(chi2.ppf(.999, d))
    for r in range(rounds):
        bp, lq = _mixture_draw(d, min(power, pilot_power), ctx.m0, prior_cov, components, shares,
                               int(streams[2 + 2 * r]), int(streams[3 + 2 * r]))
        ll, _, _ = ctx.obs.loglik(ctx.predict(bp))
        lp, _ = ctx.log_prior(bp)
        joint = np.where(np.isfinite(ll + lp), ll + lp, -np.inf)
        lw = joint - lq
        if not np.any(np.isfinite(lw)):
            break
        w = np.exp(lw - logsumexp(lw))
        changed = False
        far = np.min(np.vstack([_mahalanobis(bp, c['mean'], c['cov']) for c in components]), axis=0) > tau
        heavy = [k for k in np.argsort(-w) if far[k] and w[k] >= 1e-3][:top]
        dense = [k for k in np.argsort(-joint) if far[k] and np.isfinite(joint[k])][:top]
        for k in dict.fromkeys(heavy + dense):
            if len(components) >= max_components:
                break
            if min(_mahalanobis(bp[k], c['mean'], c['cov'])[0] for c in components) <= tau:
                continue
            best = min(c['J'] for c in components)
            st = ctx.solve(start=bp[k])
            if not np.isfinite(st.ofv) or st.J > best + delta_j:
                continue
            if min(_mahalanobis(st.b, c['mean'], c['cov'])[0] for c in components) > tau:
                try:
                    cov = 2. * np.linalg.inv(st.H) + .01 * prior_cov
                    np.linalg.cholesky(cov)
                except np.linalg.LinAlgError:
                    continue
                components.append(dict(mean=np.array(st.b, dtype=float), cov=cov, J=float(st.J)))
                changed = True
            elif w[k] >= tail_weight and -joint[k] <= best + delta_j:
                near = components[int(np.argmin([_mahalanobis(bp[k], c['mean'], c['cov'])[0] for c in components]))]
                components.append(dict(mean=bp[k].copy(), cov=near['cov'].copy(), J=float(-joint[k])))
                changed = True
        best = min(c['J'] for c in components)
        kept = [c for c in components if c['J'] <= best + delta_j]
        changed = changed or len(kept) != len(components)
        components = kept
        if changed:
            shares = [1. / len(components)] * len(components)
            continue
        if 1. / np.dot(w, w) < 5.:
            break
        lr = np.vstack([math.log(sh) + _t_logpdf(bp, c['mean'], c['cov']) for c, sh in zip(components, shares)])
        rho = np.exp(lr - logsumexp(lr, axis=0))
        mass = np.zeros(len(components))
        for j, c in enumerate(components):
            wc = w * rho[j]
            mass[j] = wc.sum()
            if mass[j] <= 0:
                continue
            wn = wc / mass[j]
            if 1. / np.dot(wn, wn) < 5.:
                continue
            mean = wn @ bp
            cov = (bp - mean).T @ ((bp - mean) * wn[:, None])
            cov = 1.5 * .5 * (cov + cov.T) + .01 * prior_cov
            try:
                np.linalg.cholesky(cov)
            except np.linalg.LinAlgError:
                continue
            c['mean'], c['cov'] = mean, cov
        mass = np.maximum(mass / max(mass.sum(), 1e-300), .1)
        shares = list(mass / mass.sum())
    return components, shares


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
        self.proposal = getattr(problem, 'integration', {}).get('proposal', 'gaussian')
        self.components = []
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
            if self.proposal == 'mixture':
                streams = np.random.SeedSequence([int(seed) % (2 ** 63), i]).generate_state(4 + 2 * adapt_rounds)
                components, shares = _mixture_components(ctx, prior_cov, st, streams, p, pilot_power, adapt_rounds + 1)
                b, lq = _mixture_draw(d, p, ctx.m0, prior_cov, components, shares, int(streams[0]), int(streams[1]))
                b.setflags(write=False)
                self.particles.append(b)
                self.logq.append(lq)
                self.components.append(len(components))
                continue
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
