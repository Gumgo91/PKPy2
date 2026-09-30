"""Fitting entry point of the general engine and its result object."""
from dataclasses import dataclass, field
import json
import math
import time
import numpy as np
from .problem import GeneralProblem
from . import laplace as L
from .refine import refine
from .uncertainty import estimate_uncertainty, laplace_uncertainty


@dataclass
class GeneralFitResult:
    problem: GeneralProblem = field(repr=False)
    x: np.ndarray
    ofv: float
    status: str
    audit: dict
    estimation: dict
    seconds: float
    uncertainty_report: dict | None = None

    @property
    def converged(self):
        return self.status == 'converged' and bool(self.audit.get('passed'))

    @property
    def model(self):
        return self.problem.structure.name

    @property
    def modes(self):
        """Conditional modes of a Laplace fit (warm starts that reproduce its OFV), else None."""
        saved = (self.estimation or {}).get('conditional_modes')
        return [L.ModeStart(b) for b in saved] if saved is not None else None

    def _describe(self):
        return self.problem.describe(self.x)

    @property
    def theta(self):
        return self._describe()['theta']

    @property
    def omega(self):
        return self._describe()['omega']

    @property
    def omega_covariance(self):
        return self._describe()['omega_covariance']

    @property
    def iov(self):
        return self._describe()['iov']

    @property
    def coefficients(self):
        return self._describe()['coefficients']

    @property
    def sigma(self):
        return self._describe()['sigma']

    @property
    def labels(self):
        return list(self.problem.labels)

    @property
    def n_param(self):
        return len(self.x)

    @property
    def aic(self):
        return self.ofv + 2 * len(self.x)

    @property
    def bic(self):
        return self.ofv + len(self.x) * math.log(self.problem.n_obs)

    def uncertainty(self, **options):
        if not self.converged:
            raise ValueError('uncertainty requires a converged fit')
        if (self.estimation or {}).get('method') == 'laplace':
            self.uncertainty_report = laplace_uncertainty(self.problem, self.x, self.ofv, modes=self.modes, **options)
        else:
            self.uncertainty_report = estimate_uncertainty(self.problem, self.x, self.ofv, **options)
        return self.uncertainty_report

    def to_dict(self):
        d = self._describe()
        return dict(format='pkpy2-general-fit-v1', model=self.model, engine='general', theta=d['theta'],
                    omega=d['omega'], omega_covariance=d['omega_covariance'], iov=d['iov'],
                    coefficients=d['coefficients'], sigma=d['sigma'], ofv=self.ofv, status=self.status,
                    converged=self.converged, audit=self.audit, seconds=self.seconds, x=self.x.tolist(),
                    coordinates=self.labels, n_param=self.n_param, aic=self.aic, bic=self.bic,
                    data_sha256=self.problem.data_sha256, estimation=self.estimation,
                    uncertainty=self.uncertainty_report)

    def save(self, path):
        from pathlib import Path

        def convert(v):
            if isinstance(v, np.ndarray):
                return v.tolist()
            if isinstance(v, np.generic):
                return v.item()
            raise TypeError(type(v).__name__)
        Path(path).write_text(json.dumps(self.to_dict(), indent=1, default=convert, allow_nan=True), encoding='utf-8')


def laplace_audit(problem, x, modes=None, *, ofv_tolerance=.05, step=1e-3):
    """Stationarity of the Laplace objective at x as a Newton decrement: sum_j g_j^2 / (2 H_jj), the OFV decrease
    still available from the exact projected gradient g (modes re-solved) with the diagonal curvature H_jj (one
    forward difference of the fast gradient per coordinate, step * max(coordinate scale, |x_j|)). Passed when it
    is at most ofv_tolerance. modes: conditional modes to warm-start from (those of the minimization); also
    returned, for reproducing the OFV later."""
    from .refine import coordinate_scale, projected_score
    objective = L.LaplaceObjective(problem)
    if modes is not None:
        objective.modes = list(modes)
    value, exact = objective.value_grad(x, problem.bounds, exact=True)
    if not np.isfinite(value):
        return dict(method='laplace', ofv=float(value), passed=False, message='objective not finite'), None
    base = list(objective.modes)
    _, grad = objective.value_grad(x, problem.bounds)
    scale = coordinate_scale(problem)
    pg = projected_score(x, exact, problem.bounds)
    rows = {}
    for j in range(len(x)):
        if abs(pg[j]) * scale[j] < .01:
            continue
        h = -np.sign(pg[j]) * step * max(scale[j], abs(x[j]))           # a step downhill stays inside the bounds
        lo, hi = problem.bounds[j]
        point = x.copy()
        point[j] = min(max(x[j] + h, lo if lo is not None else -np.inf), hi if hi is not None else np.inf)
        if point[j] == x[j]:
            continue
        objective.modes = list(base)
        v2, g2 = objective.value_grad(point, problem.bounds)
        curvature = (g2[j] - grad[j]) / (point[j] - x[j]) if np.isfinite(v2) else np.nan
        rows[problem.labels[j]] = float(pg[j] ** 2 / (2. * curvature)) if curvature > 0 else np.inf
    decrement = float(sum(rows.values()))
    worst = max(rows, key=rows.get) if rows else None
    return (dict(method='laplace', ofv=float(value), newton_decrement=decrement, ofv_tolerance=ofv_tolerance,
                 worst_coordinate=worst, passed=bool(decrement <= ofv_tolerance)), base)


def fit_general(data, model, *, seed=0, workers=None, laplace_options=None, refinement_options=None, callback=None,
                start=None, integration=None, method='importance', laplace_tolerance=.05, polish_iterations=20):
    """Laplace exploration followed by importance refinement and an independent audit.

    integration: dict(proposal='gaussian'|'mixture', mode_search=bool) for every bank of the fit and of
    later computations on the result (GeneralProblem.integration); the defaults keep one adapted Gaussian.
    method='laplace' stops after the Laplace exploration and a short polish with the exact gradient (at most
    polish_iterations): the OFV is the Laplace (FOCE-I-type) objective, the fit counts as converged when the
    Newton decrement is at most laplace_tolerance OFV units (laplace_audit), the conditional modes are saved in
    estimation['conditional_modes'], and uncertainty() uses the curvature of the Laplace objective."""
    import numba
    if method not in ('importance', 'laplace'):
        raise ValueError("method must be 'importance' or 'laplace'")
    if workers is not None:
        numba.set_num_threads(max(1, min(int(workers), numba.config.NUMBA_NUM_THREADS)))
    t0 = time.perf_counter()
    problem = GeneralProblem(data, model, integration)
    x0 = problem.x0 if start is None else np.asarray(start, dtype=float)
    opts = dict(starts=4, max_iterations=150, cpu_budget_seconds=1200.)
    opts.update(laplace_options or {})
    emit = (lambda phase: (lambda row: callback(dict(phase=phase, **row)))) if callback else (lambda phase: None)
    if method == 'laplace':
        x_lap, exploration, modes = L.explore(problem, x0, seed=seed + 30000049, callback=emit('laplace_exploration'),
                                              return_modes=True, **opts)
        x_lap, modes, exploration['polish'] = L.polish(problem, x_lap, modes, scaled=bool(opts.get('scaled')),
                                                       max_iterations=polish_iterations)
        if callback is not None:
            callback(dict(phase='laplace_polish', **exploration['polish']))
        audit, modes = laplace_audit(problem, x_lap, modes, ofv_tolerance=laplace_tolerance)
        status = 'converged' if audit['passed'] else 'partial'
        estimation = dict(method='laplace', laplace_exploration=exploration, seed=seed,
                          message='stationary Laplace objective' if audit['passed'] else
                          audit.get('message') or f"Newton decrement {audit['newton_decrement']:.3g} "
                          f"(mostly {audit['worst_coordinate']}) above {laplace_tolerance}")
        if modes is not None:
            estimation['conditional_modes'] = [np.asarray(m.b, dtype=float).tolist() for m in modes]
        if integration:
            estimation['integration'] = dict(problem.integration)
        return GeneralFitResult(problem, np.asarray(x_lap, dtype=float), float(audit['ofv']), status, audit, estimation,
                                time.perf_counter() - t0)
    x_lap, exploration = L.explore(problem, x0, seed=seed + 30000049, callback=emit('laplace_exploration'), **opts)
    ropts = dict(power=10, audit_power=14)
    ropts.update(refinement_options or {})
    result = refine(problem, x_lap, seed=seed + 10000019, callback=emit('importance_refinement'), **ropts)
    estimation = dict(laplace_exploration=exploration, refinement=dict(
        status=result['status'], message=result['message'], stages=result['stages'], seconds=result['seconds']),
        seed=seed)
    if integration:
        estimation['integration'] = dict(problem.integration)
    return GeneralFitResult(problem, result['x'], float(result['ofv']), result['status'], result['audit'], estimation,
                            time.perf_counter() - t0)


def evaluate_general(data, model, *, seed=0, power=14, ofv_tolerance=.05, minimum_ess=100, integration=None):
    """Marginal OFV and a result object at the declared values, without estimation.

    The analogue of a NONMEM run with MAXEVAL=0: the OFV is the mean of two
    independent importance banks; audit['passed'] reports whether they agree
    within ofv_tolerance with at least minimum_ess effective particles per subject.
    Diagnostics, simulation and VPC accept the returned result.
    """
    from .importance import Bank
    t0 = time.perf_counter()
    problem = GeneralProblem(data, model, integration)
    x = problem.x0.copy()
    _, states = L.LaplaceObjective(problem).states(x)
    replicas = []
    for rep in range(2):
        b = Bank(problem, x, power=power, seed=seed + 9000001 + rep, states=states)
        v, _ = b.evaluate(x, gradient=False)
        replicas.append(dict(seed=seed + 9000001 + rep, ofv=float(v), minimum_ess=float(b.last['min_ess'])))
    ofv_range = abs(replicas[0]['ofv'] - replicas[1]['ofv'])
    random = problem.q + problem.Q > 0
    audit = dict(x=x.tolist(), replicas=replicas, ofv_range=ofv_range, power=power,
                 passed=bool(ofv_range <= ofv_tolerance and
                             (not random or min(r['minimum_ess'] for r in replicas) >= minimum_ess)))
    return GeneralFitResult(problem, x, float(np.mean([r['ofv'] for r in replicas])), 'evaluated', audit,
                            dict(evaluated=True, seed=seed), time.perf_counter() - t0)
