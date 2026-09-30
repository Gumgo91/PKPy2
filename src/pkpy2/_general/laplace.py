"""Conditional modes, curvature and the Laplace objective of a general model.

Random effects of subject i are b = (phi, kappa): phi are the physical
transformed values of the parameters with IIV (phi = mu_i + eta) and kappa the
interoccasion effects (occasions x IOV parameters). The joint negative log
density is J(b) = -log p(y | b) - log N(phi; mu_i, Omega) - log N(kappa; 0, S).
Modes are found by Newton steps with the expected-information (Fisher)
curvature H = Jf' W Jf + prior precision, derivatives of the predictions by
central differences evaluated in one batched prediction call. The Laplace
contribution is 2 J(b*) + log det H - d log(2 pi) (FOCE-I-type curvature).
"""
from dataclasses import dataclass
import math
import time
import numpy as np
from scipy.linalg import solve_triangular, cho_factor, cho_solve
from .likelihood import ObservationModel
from .packing import batch_predict

LOG2PI = math.log(2. * math.pi)


@dataclass
class ModeState:
    b: np.ndarray
    J: float
    H: np.ndarray
    jac: np.ndarray        # (m, d) derivatives of the used predictions
    pred: np.ndarray       # (n_obs,) predictions at the mode
    ofv: float
    iterations: int
    converged: bool


class SubjectContext:
    """Population-dependent quantities of one subject at a population point."""

    def __init__(self, problem, i, pop):
        self.problem = problem
        self.i = i
        self.pop = pop
        s = problem.subjects[i]
        self.subject = s
        self.mu = problem.subject_mu(i, pop)
        self.tv = problem.subject_tv(i, pop)
        self.O = len(s.occasions) if problem.Q else 0
        self.q = problem.q
        self.d = self.q + self.O * problem.Q
        self.obs = ObservationModel(s, pop)
        self.m0 = np.concatenate([self.mu[problem.iiv], np.zeros(self.O * problem.Q)])
        self.chol = pop.chol
        self.iov_var = np.tile(pop.iov_sd ** 2, self.O)
        self.logdet_prior = (2. * np.sum(np.log(np.diag(self.chol))) if self.q else 0.) + float(np.sum(np.log(self.iov_var)))

    def inputs(self, b):
        """base (K, P) transformed values and kappa (K, O, Q) for particles b (K, d)."""
        K = b.shape[0]
        base = np.tile(self.mu, (K, 1))
        if self.q:
            base[:, self.problem.iiv] = b[:, :self.q]
        if self.problem.Q:
            kappa = b[:, self.q:].reshape(K, self.O, self.problem.Q)
        else:
            kappa = np.zeros((K, 1, 0))
        return base, kappa

    def predict(self, b):
        base, kappa = self.inputs(np.atleast_2d(b))
        pred, errors = batch_predict(self.problem.kernel, self.subject, base, self.tv, kappa,
                                     self.problem.iov_param, self.problem.transform)
        pred[errors != 0] = np.nan
        return pred

    def log_prior(self, b):
        """log N(phi; mu, Omega) + log N(kappa; 0, S) for particles b (K, d); also returns u = L^-1 (phi - mu)."""
        b = np.atleast_2d(b)
        total = np.full(b.shape[0], -.5 * (self.d * LOG2PI + self.logdet_prior))
        u = None
        if self.q:
            delta = (b[:, :self.q] - self.m0[:self.q]).T
            u = solve_triangular(self.chol, delta, lower=True)
            total -= .5 * np.sum(u * u, axis=0)
        if self.d > self.q:
            total -= .5 * np.sum(b[:, self.q:] ** 2 / self.iov_var, axis=1)
        return total, u

    def precision(self):
        p = np.zeros((self.d, self.d))
        if self.q:
            inv = cho_solve((self.chol, True), np.eye(self.q))
            p[:self.q, :self.q] = inv
        if self.d > self.q:
            p[self.q:, self.q:] = np.diag(1. / self.iov_var)
        return p

    def joint(self, b, pred=None):
        """J(b) = -loglik - log prior for particles b (K, d)."""
        b = np.atleast_2d(b)
        if pred is None:
            pred = self.predict(b)
        ll, _, _ = self.obs.loglik(pred)
        lp, _ = self.log_prior(b)
        value = -(ll + lp)
        return np.where(np.isfinite(value), value, np.inf), pred

    def solve(self, start=None, max_iter=60, tol=1e-7):
        d = self.d
        if d == 0:
            value, pred = self.joint(np.zeros((1, 0)))
            ofv = 2. * float(value[0])
            return ModeState(np.zeros(0), float(value[0]), np.zeros((0, 0)), np.zeros((self.obs.m, 0)),
                             pred[0], ofv, 0, bool(np.isfinite(ofv)))
        b = np.array(self.m0 if start is None else start, dtype=float)
        precision = self.precision()
        scale = np.sqrt(np.concatenate([np.diag(self.pop.omega), self.iov_var]))
        converged = False
        J0 = np.inf
        for it in range(max_iter):
            h = 1e-4 * np.maximum(1., np.abs(b))
            pts = np.vstack([b, b + np.diag(h), b - np.diag(h)])
            pred = self.predict(pts)
            J0 = float(self.joint(pts[:1], pred[:1])[0][0])
            if not np.isfinite(J0):
                if start is not None and it == 0:
                    b = self.m0.copy()
                    continue
                break
            used = self.obs.index
            jac = (pred[1:d + 1, used] - pred[d + 1:, used]).T / (2. * h)
            if not np.all(np.isfinite(jac)):
                jac = np.where(np.isfinite(jac), jac, 0.)
            grad_f, weight = self.obs.gradient_weights(pred[:1])
            grad = -jac.T @ grad_f[0] + precision @ (b - self.m0)
            H = jac.T @ (jac * weight[0][:, None]) + precision
            try:
                step = -np.linalg.solve(H, grad)
            except np.linalg.LinAlgError:
                step = -grad / np.diag(H)
            limit = 3. * scale
            ratio = np.max(np.abs(step) / limit)
            if ratio > 1.:
                step /= ratio
            alphas = np.array([1., .5, .25, .125, .0625, .015625])
            trial = b + alphas[:, None] * step
            values, _ = self.joint(trial)
            decrease = float(grad @ step)
            ok = np.nonzero(values <= J0 + 1e-4 * alphas * decrease)[0]
            pick = ok[0] if len(ok) else int(np.argmin(values))
            if values[pick] > J0 and not len(ok):
                converged = np.max(np.abs(grad) * scale) < 1e-3
                break
            move = alphas[pick] * step
            b = b + move
            if np.max(np.abs(move) / scale) < tol or J0 - values[pick] < 1e-10:
                converged = True
                break
        # final curvature at the returned point
        h = 1e-4 * np.maximum(1., np.abs(b))
        pts = np.vstack([b, b + np.diag(h), b - np.diag(h)])
        pred = self.predict(pts)
        J = float(self.joint(pts[:1], pred[:1])[0][0])
        used = self.obs.index
        jac = (pred[1:d + 1, used] - pred[d + 1:, used]).T / (2. * h)
        jac = np.where(np.isfinite(jac), jac, 0.)
        _, weight = self.obs.gradient_weights(pred[:1])
        H = jac.T @ (jac * weight[0][:, None]) + precision
        H = .5 * (H + H.T)
        sign, logdet = np.linalg.slogdet(H)
        ofv = 2. * J + logdet - d * LOG2PI if sign > 0 else np.inf
        return ModeState(b, J, H, jac, pred[0], float(ofv), it + 1, converged and np.isfinite(ofv))

    def fixed_mode_ofv(self, state):
        """Laplace contribution at this population point with the mode and prediction derivatives held fixed."""
        if self.d == 0:
            value, _ = self.joint(np.zeros((1, 0)), state.pred[None, :])
            return 2. * float(value[0])
        J = float(self.joint(state.b[None, :], state.pred[None, :])[0][0])
        _, weight = self.obs.gradient_weights(state.pred[None, :])
        H = state.jac.T @ (state.jac * weight[0][:, None]) + self.precision()
        sign, logdet = np.linalg.slogdet(.5 * (H + H.T))
        return 2. * J + logdet - self.d * LOG2PI if sign > 0 else np.inf


def search_mode(ctx, state, power=8):
    """Conditional mode after a second Newton solve from the best of 2^power prior draws (scrambled Sobol,
    seeded by the subject index); the lower joint density J wins. Guards against a first solve that ends
    in a secondary basin, e.g. with the peripheral compartments of a three-compartment model swapped."""
    d = ctx.d
    if d == 0:
        return state
    prior_cov = np.zeros((d, d))
    if ctx.q:
        prior_cov[:ctx.q, :ctx.q] = ctx.pop.omega
    if d > ctx.q:
        prior_cov[ctx.q:, ctx.q:] = np.diag(ctx.iov_var)
    from scipy.special import ndtri
    from scipy.stats import qmc
    u = qmc.Sobol(d, scramble=True, seed=np.random.default_rng([int(ctx.i), 7919])).random_base2(power)
    b = ctx.m0 + ndtri(np.clip(u, np.finfo(float).eps, 1. - np.finfo(float).eps)) @ np.linalg.cholesky(prior_cov).T
    J, _ = ctx.joint(b)
    k = int(np.argmin(J))
    if not np.isfinite(J[k]) or (np.isfinite(state.J) and J[k] >= state.J):
        return state
    alt = ctx.solve(start=b[k])
    return alt if np.isfinite(alt.ofv) and (not np.isfinite(state.ofv) or alt.J < state.J) else state


class ModeStart:
    """A warm start for LaplaceObjective.states that carries only the random-effect vector b (e.g. saved modes)."""
    __slots__ = ('b',)

    def __init__(self, b):
        self.b = np.asarray(b, dtype=float)


def best_modes(problem, x, warm):
    """Per subject, the conditional mode with the lower joint density J among a re-solve from the warm start and a
    cold start (with search_mode when problem.integration['mode_search'] is set); None outside the domain."""
    objective = LaplaceObjective(problem)
    _, cold = objective.states(x, starts=[None] * problem.N)
    _, again = objective.states(x, starts=warm)
    if cold is None or again is None:
        return again if cold is None else cold
    return [a if a.J <= c.J else c for a, c in zip(again, cold)]


class LaplaceObjective:
    """Sum of subject Laplace contributions; modes are warm-started between calls.

    With problem.integration['mode_search'], a subject without a warm start also gets search_mode."""

    def __init__(self, problem):
        self.problem = problem
        self.modes = [None] * problem.N
        self.evaluations = 0
        self.mode_search = bool(getattr(problem, 'integration', {}).get('mode_search'))

    def states(self, x, starts=None):
        """Laplace OFV and subject modes at x; (inf, None) outside the numerically valid domain."""
        try:
            with np.errstate(over='ignore', invalid='ignore', divide='ignore'):
                return self._states(x, starts)
        except (ArithmeticError, ValueError, np.linalg.LinAlgError):
            return np.inf, None

    def _states(self, x, starts=None):
        pop = self.problem.population(x)
        if not self.problem.valid(pop):
            return np.inf, None
        starts = starts if starts is not None else self.modes
        out = []
        total = 0.
        for i in range(self.problem.N):
            ctx = SubjectContext(self.problem, i, pop)
            start = starts[i].b if starts[i] is not None and len(starts[i].b) == ctx.d else None
            st = ctx.solve(start)
            if start is None and self.mode_search:
                st = search_mode(ctx, st)
            out.append(st)
            total += st.ofv
        self.evaluations += 1
        return total, out

    def value(self, x):
        total, states = self.states(x)
        if states is not None and np.isfinite(total):
            self.modes = states
        return total

    def value_grad(self, x, bounds=None, exact=False):
        """Objective and central-difference gradient. For density coordinates the modes and prediction derivatives
        are held fixed (fast; neglects the movement of the modes through the curvature term) unless exact=True,
        which re-solves the modes at every difference point for all coordinates (the total derivative). The exact
        gradient uses steps of 1e-4 instead of 1e-5 (relative): the re-solved modes carry a small solver noise
        (about 1e-5 in the OFV for weakly identified random effects), which a smaller step would amplify."""
        x = np.asarray(x, dtype=float)
        total, states = self.states(x)
        if states is None or not np.isfinite(total):
            return np.inf, np.zeros_like(x)
        self.modes = states
        grad = np.zeros_like(x)
        problem = self.problem
        for j in range(len(x)):
            h = (1e-4 if exact else 1e-5) * max(1., abs(x[j]))
            plus, minus = x.copy(), x.copy()
            plus[j] += h
            minus[j] -= h
            if bounds is not None:
                lo, hi = bounds[j]
                if lo is not None:
                    minus[j] = max(minus[j], lo)
                if hi is not None:
                    plus[j] = min(plus[j], hi)
            span = plus[j] - minus[j]
            if span <= 0:
                continue
            if problem.density[j] and not exact:
                fp = self._fixed(plus, states)
                fm = self._fixed(minus, states)
            else:
                fp, _ = self.states(plus, states)
                fm, _ = self.states(minus, states)
            grad[j] = (fp - fm) / span if np.isfinite(fp) and np.isfinite(fm) else 0.
        return total, grad

    def _fixed(self, x, states):
        try:
            with np.errstate(over='ignore', invalid='ignore', divide='ignore'):
                return self._fixed_ofv(x, states)
        except (ArithmeticError, ValueError, np.linalg.LinAlgError):
            return np.inf

    def _fixed_ofv(self, x, states):
        pop = self.problem.population(x)
        if not self.problem.valid(pop):
            return np.inf
        return sum(SubjectContext(self.problem, i, pop).fixed_mode_ofv(states[i]) for i in range(self.problem.N))


def linear_domain_bounds(problem, margin=.98):
    """problem.bounds tightened so that every linear covariate effect 1 + beta (z - center) stays positive for all
    subjects and records (beta within margin of the domain edge); a line search then cannot leave the domain."""
    bounds = list(problem.bounds)
    for k in problem.linear_effects:
        c = int(problem.beta_coord[k])
        if c < 0:
            continue
        z = (np.concatenate([zt[:, k] for zt in problem.z_tv]) if problem.effects[k]['time_varying']
             else problem.z_tic[:, k])
        zmax, zmin = float(np.max(z)), float(np.min(z))
        lo = -margin / zmax if zmax > 0 else None
        hi = -margin / zmin if zmin < 0 else None
        old_lo, old_hi = bounds[c]
        lo = old_lo if lo is None else (lo if old_lo is None else max(lo, old_lo))
        hi = old_hi if hi is None else (hi if old_hi is None else min(hi, old_hi))
        bounds[c] = (lo, hi)
    return bounds


class AnchoredObjective:
    """value_grad whose warm starts are the modes of the lowest objective value seen so far, so that a far
    line-search trial (whose modes may sit in another basin of a multimodal subject) does not carry its modes
    into the next evaluation. Tracks the best point (x, value, modes)."""

    def __init__(self, objective, exact):
        self.objective = objective
        self.exact = exact
        self.x, self.value, self.modes = None, np.inf, None

    def __call__(self, x):
        if self.modes is not None:
            self.objective.modes = list(self.modes)
        value, grad = self.objective.value_grad(x, self.objective.problem.bounds, exact=self.exact)
        if np.isfinite(value) and value < self.value:
            self.x, self.value, self.modes = np.array(x, dtype=float), float(value), list(self.objective.modes)
        return value, grad


def polish(problem, x, modes=None, *, scaled=False, max_iterations=20):
    """Continue the minimization from x with the exact gradient (modes re-solved at every difference point, warm
    started from `modes`, anchored at the best point). Returns (x, modes, record) of the best point seen."""
    from scipy.optimize import minimize
    start = time.process_time()
    x = np.asarray(x, dtype=float)
    if scaled:
        from .refine import coordinate_scale
        scale = coordinate_scale(problem)
    else:
        scale = np.ones(len(x))
    objective = LaplaceObjective(problem)
    if modes is not None:
        objective.modes = list(modes)
    anchor = AnchoredObjective(objective, exact=True)

    def f(u):
        value, grad = anchor(u * scale)
        return value, grad * scale
    value0, _ = f(x / scale)
    fit = minimize(f, x / scale, jac=True, method='L-BFGS-B',
                   bounds=[(None if lo is None else lo / s, None if hi is None else hi / s)
                           for (lo, hi), s in zip(linear_domain_bounds(problem), scale)],
                   options=dict(maxiter=max_iterations, ftol=1e-12, gtol=1e-5, maxls=30))
    record = dict(start_ofv=float(value0), ofv=float(anchor.value), iterations=int(fit.nit), status=str(fit.message),
                  cpu_seconds=time.process_time() - start)
    if anchor.x is None:
        return x, modes, record
    return anchor.x, anchor.modes, record


def explore(problem, x0, *, starts=4, spreads=(math.log(3.), math.log(10.)), max_iterations=150,
            cpu_budget_seconds=600., seed=0, agreement_starts=3, agreement_tolerance=.1, scaled=False,
            callback=None, return_modes=False, gradient='fast', wall_seconds=None):
    """Multi-start minimization of the Laplace objective over the typical values and covariate coefficients.

    scaled=True measures each covariate coefficient in units of one standard deviation of its covariate term
    during the minimization (the refinement's coordinate scale), keeps linear covariate coefficients inside
    their valid domain (linear_domain_bounds) and perturbs only the typical values at the extra starts. It
    conditions models whose linear or exponential covariate effects use natural units (age in years, for
    example), where unscaled quasi-Newton steps stall and perturbed coefficients leave the valid domain.
    gradient='exact' re-solves the conditional modes at every difference point (LaplaceObjective.value_grad):
    slower, but the fast gradient, which holds the modes fixed, can be biased when random effects are weakly
    identified (rich-data multi-compartment models). wall_seconds stops the minimization (at the current iterate)
    and the remaining starts after that many seconds. The defaults keep the unscaled, fast minimization.
    """
    if gradient not in ('fast', 'exact'):
        raise ValueError("gradient must be 'fast' or 'exact'")
    exact = gradient == 'exact'
    deadline = None if wall_seconds is None else time.perf_counter() + float(wall_seconds)

    def stop(intermediate_result):
        if time.perf_counter() > deadline:
            raise StopIteration
    timed = dict(callback=stop) if deadline is not None else {}
    from scipy.optimize import minimize
    from scipy.stats import qmc
    start_time = time.process_time()
    structural = [j for j, lab in enumerate(problem.labels) if lab.startswith(('theta:', 'beta:'))]
    if scaled:
        from .refine import coordinate_scale
        scale = coordinate_scale(problem)
        perturbed = [j for j in structural if problem.labels[j].startswith('theta:')]
    else:
        scale = np.ones(len(x0))
        perturbed = structural
    points = [np.array(x0, dtype=float)]
    if starts > 1 and perturbed:
        m = int(math.ceil(math.log2(max(starts - 1, 1))))
        draws = qmc.Sobol(len(perturbed), scramble=True, seed=seed).random_base2(m)[:starts - 1]
        for k, u in enumerate(draws):
            p = points[0].copy()
            p[perturbed] += (2. * u - 1.) * spreads[k % len(spreads)]
            for j, (lo, hi) in enumerate(problem.bounds):
                if lo is not None:
                    p[j] = max(p[j], lo + 1e-6)
                if hi is not None:
                    p[j] = min(p[j], hi - 1e-6)
            points.append(p)
    results = []
    for index, p in enumerate(points):
        if results and (time.process_time() - start_time > cpu_budget_seconds
                        or (deadline is not None and time.perf_counter() > deadline)):
            break
        objective = LaplaceObjective(problem)
        value0 = objective.value(p)
        if not np.isfinite(value0):
            results.append(dict(start=index, ofv=np.inf, x=p, status='nonfinite start', evaluations=1))
            continue
        def run(point):
            if scaled or exact or timed:
                # opt-in path: warm starts anchored at the best point (AnchoredObjective); scaled coordinates
                # and linear-covariate domain bounds when scaled
                anchor = AnchoredObjective(objective, exact)
                if objective.modes and objective.modes[0] is not None:
                    anchor.modes = list(objective.modes)
                s = scale if scaled else np.ones(len(point))
                limits = linear_domain_bounds(problem) if scaled else problem.bounds

                def f(u):
                    value, grad = anchor(u * s)
                    return value, grad * s
                fit = minimize(f, point / s, jac=True, method='L-BFGS-B',
                               bounds=[(None if lo is None else lo / w, None if hi is None else hi / w)
                                       for (lo, hi), w in zip(limits, s)],
                               options=dict(maxiter=max_iterations, ftol=1e-10, gtol=1e-4, maxls=30), **timed)
                if anchor.x is None:
                    return fit, fit.x * s
                objective.modes = list(anchor.modes)
                return fit, anchor.x
            fit = minimize(lambda z: objective.value_grad(z, problem.bounds), point, jac=True, method='L-BFGS-B',
                           bounds=[(lo, hi) for lo, hi in problem.bounds],
                           options=dict(maxiter=max_iterations, ftol=1e-10, gtol=1e-4, maxls=30))
            return fit, fit.x
        fit, x_end = run(p)
        value = objective.value(x_end)
        iterations, continued = int(fit.nit), 0
        if objective.mode_search:
            # warm-started modes follow the optimization path; when a cold search finds a subject mode with a
            # lower joint density at the end point, adopt it and continue the minimization (at most twice)
            for _ in range(2):
                if deadline is not None and time.perf_counter() > deadline:
                    break
                modes = best_modes(problem, x_end, objective.modes)
                if modes is None or not sum(m.ofv for m in modes) < value - 1e-6:
                    break
                objective.modes = modes
                fit, x_end = run(x_end)
                value = objective.value(x_end)
                iterations += int(fit.nit)
                continued += 1
        results.append(dict(start=index, ofv=float(value), start_ofv=float(value0), x=x_end.copy(),
                            status=str(fit.message), iterations=iterations, evaluations=objective.evaluations,
                            **(dict(mode_restarts=continued, modes=list(objective.modes)) if objective.mode_search
                               else {})))
        if callback is not None:
            callback(dict(start=index, ofv=float(value), iterations=iterations))
        done = [r for r in results if np.isfinite(r['ofv'])]
        if len(done) >= agreement_starts and index + 1 >= agreement_starts:
            best = min(r['ofv'] for r in done[:agreement_starts])
            if all(abs(r['ofv'] - best) <= agreement_tolerance for r in done[:agreement_starts]):
                break
    finite = [r for r in results if np.isfinite(r['ofv'])]
    if not finite:
        raise RuntimeError('the Laplace objective is not finite at any start')
    best = min(finite, key=lambda r: r['ofv'])
    optima = []
    for r in sorted(finite, key=lambda r: r['ofv']):
        if all(abs(r['ofv'] - o['ofv']) > agreement_tolerance for o in optima):
            optima.append(dict(ofv=r['ofv'], start=r['start']))
    record = dict(status='completed', starts=len(results), best_ofv=best['ofv'], distinct_optima=optima,
                  cpu_seconds=time.process_time() - start_time,
                  runs=[{k: (v.tolist() if isinstance(v, np.ndarray) else v) for k, v in r.items() if k != 'modes'}
                        for r in results])
    if return_modes:
        return best['x'], record, best.get('modes')
    return best['x'], record
