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


class LaplaceObjective:
    """Sum of subject Laplace contributions; modes are warm-started between calls."""

    def __init__(self, problem):
        self.problem = problem
        self.modes = [None] * problem.N
        self.evaluations = 0

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
            out.append(st)
            total += st.ofv
        self.evaluations += 1
        return total, out

    def value(self, x):
        total, states = self.states(x)
        if states is not None and np.isfinite(total):
            self.modes = states
        return total

    def value_grad(self, x, bounds=None):
        x = np.asarray(x, dtype=float)
        total, states = self.states(x)
        if states is None or not np.isfinite(total):
            return np.inf, np.zeros_like(x)
        self.modes = states
        grad = np.zeros_like(x)
        problem = self.problem
        for j in range(len(x)):
            h = 1e-5 * max(1., abs(x[j]))
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
            if problem.density[j]:
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


def explore(problem, x0, *, starts=4, spreads=(math.log(3.), math.log(10.)), max_iterations=150,
            cpu_budget_seconds=600., seed=0, agreement_starts=3, agreement_tolerance=.1, callback=None):
    """Multi-start minimization of the Laplace objective over the typical values and covariate coefficients."""
    from scipy.optimize import minimize
    from scipy.stats import qmc
    start_time = time.process_time()
    structural = [j for j, lab in enumerate(problem.labels) if lab.startswith(('theta:', 'beta:'))]
    points = [np.array(x0, dtype=float)]
    if starts > 1 and structural:
        m = int(math.ceil(math.log2(max(starts - 1, 1))))
        draws = qmc.Sobol(len(structural), scramble=True, seed=seed).random_base2(m)[:starts - 1]
        for k, u in enumerate(draws):
            p = points[0].copy()
            p[structural] += (2. * u - 1.) * spreads[k % len(spreads)]
            for j, (lo, hi) in enumerate(problem.bounds):
                if lo is not None:
                    p[j] = max(p[j], lo + 1e-6)
                if hi is not None:
                    p[j] = min(p[j], hi - 1e-6)
            points.append(p)
    results = []
    for index, p in enumerate(points):
        if time.process_time() - start_time > cpu_budget_seconds and results:
            break
        objective = LaplaceObjective(problem)
        value0 = objective.value(p)
        if not np.isfinite(value0):
            results.append(dict(start=index, ofv=np.inf, x=p, status='nonfinite start', evaluations=1))
            continue
        fit = minimize(lambda z: objective.value_grad(z, problem.bounds), p, jac=True, method='L-BFGS-B',
                       bounds=[(lo, hi) for lo, hi in problem.bounds],
                       options=dict(maxiter=max_iterations, ftol=1e-10, gtol=1e-4, maxls=30))
        value = objective.value(fit.x)
        results.append(dict(start=index, ofv=float(value), start_ofv=float(value0), x=fit.x.copy(),
                            status=str(fit.message), iterations=int(fit.nit), evaluations=objective.evaluations))
        if callback is not None:
            callback(dict(start=index, ofv=float(value), iterations=int(fit.nit)))
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
                  runs=[{k: (v.tolist() if isinstance(v, np.ndarray) else v) for k, v in r.items()} for r in results])
    return best['x'], record
