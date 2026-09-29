"""Deterministic multi-start Laplace exploration before importance-sampled refinement.

The Laplace objective (2 x joint NLL at the individual modes + log det of the observed
individual Hessians - d log 2 pi, normal constants included) is minimized over the
population coordinates with L-BFGS-B and central-difference gradients, from the supplied
start and from deterministic scrambled-Sobol perturbations of the structural coordinates.
The best Laplace optimum is used only as the starting point: the final estimate, objective
value, and convergence status still come from importance-sampled refinement and its
independent two-bank audit. Distinct Laplace optima are recorded as a multimodality
diagnostic.
"""
from dataclasses import dataclass, field
import math
import time

import numpy as np
from scipy.optimize import minimize
from scipy.stats import qmc

from .packed_solver import PackedEvaluator, InnerSettings


class _Budget(Exception):
    pass


@dataclass
class LaplaceResult:
    x: np.ndarray
    ofv: float
    status: str
    message: str
    iterations: int
    evaluations: int
    cpu_seconds: float
    start_ofv: float
    history: list = field(default_factory=list)
    starts: list = field(default_factory=list)

    def record(self):
        return dict(status=self.status, message=self.message, iterations=self.iterations, evaluations=self.evaluations,
                    cpu_seconds=self.cpu_seconds, start_ofv=self.start_ofv, final_ofv=self.ofv,
                    improvement=(self.start_ofv - self.ofv) if np.isfinite(self.start_ofv) and np.isfinite(self.ofv) else None,
                    x=self.x.tolist(), history=self.history, starts=self.starts,
                    distinct_optima=distinct_optima(self.starts))


def distinct_optima(starts, ofv_gap=1.0):
    """Laplace optima more than ofv_gap apart, best first (a multimodality diagnostic)."""
    found = []
    for row in sorted((r for r in starts if r.get('final_ofv') is not None), key=lambda r: r['final_ofv']):
        if all(abs(row['final_ofv'] - f['ofv']) > ofv_gap for f in found):
            found.append(dict(ofv=row['final_ofv'], start_index=row['start_index']))
    return found


def explore_laplace(study, eta_indices, x0, decode, *, penalty=None, bounds=None, workers=1, cpu_budget_seconds=120.,
                    max_iterations=200, gradient_step=1e-3, gtol=1e-3, callback=None, starts=8,
                    spreads=(math.log(3.), math.log(10.)), structural=None, seed=0, agreement_starts=4,
                    agreement_tolerance=.1):
    """Multi-start Laplace minimization within a CPU budget; the result is never worse than x0.

    `starts` counts the supplied start plus deterministic scrambled-Sobol perturbations of the
    coordinates listed in `structural`, alternating between the half-widths in `spreads`
    (log scale for positive parameters), so that nearby and distant regions are both visited.
    Exploration stops early when the first `agreement_starts` completed starts, which include
    distant ones, reach the same Laplace optimum within `agreement_tolerance` OFV units.
    A finite set of starts cannot guarantee the global optimum; distinct optima are reported.
    """
    spreads = tuple(float(s) for s in spreads)
    if type(starts) is not int or starts < 1 or not spreads or not np.isfinite(spreads).all() or min(spreads) < 0:
        raise ValueError('positive start count and finite nonnegative spreads required')
    x0 = np.asarray(x0, dtype=float).copy()
    points = [x0]
    structural = list(range(len(x0))) if structural is None else list(structural)
    if starts > 1 and structural:
        sobol = qmc.Sobol(d=len(structural), scramble=True, seed=np.random.default_rng(seed))
        draws = sobol.random_base2(m=max(0, math.ceil(math.log2(starts - 1))))[:starts - 1]
        for k, unit in enumerate(2 * draws - 1):
            point = x0.copy()
            point[structural] += unit * spreads[k % len(spreads)]
            points.append(point)
    start = time.process_time()
    results, rows, evaluations = [], [], 0
    for index, point in enumerate(points):
        remaining = cpu_budget_seconds - (time.process_time() - start)
        if remaining <= 0:
            break
        r = _explore_single(study, eta_indices, point, decode, penalty=penalty, bounds=bounds, workers=workers,
                            cpu_budget_seconds=remaining, max_iterations=max_iterations, gradient_step=gradient_step,
                            gtol=gtol, callback=callback)
        evaluations += r.evaluations
        results.append(r)
        rows.append(dict(start_index=index, status=r.status, message=r.message,
                         start_ofv=r.start_ofv if np.isfinite(r.start_ofv) else None,
                         final_ofv=r.ofv if np.isfinite(r.ofv) else None, iterations=r.iterations,
                         cpu_seconds=r.cpu_seconds))
        if len(results) == agreement_starts:
            values = [x.ofv for x in results]
            if all(np.isfinite(values)) and max(values) - min(values) <= agreement_tolerance:
                rows[-1]['stopped'] = 'first starts agree on one Laplace optimum'
                break
    finite = [r for r in results if np.isfinite(r.ofv)]
    if not finite:
        return LaplaceResult(x0, float('nan'), 'not_applicable', 'Laplace objective not finite at any start', 0,
                             evaluations, time.process_time() - start, float('nan'), [], rows)
    best = min(finite, key=lambda r: r.ofv)
    status = 'converged' if best.status == 'converged' else 'partial'
    return LaplaceResult(best.x, best.ofv, status, best.message, sum(r.iterations for r in results), evaluations,
                         time.process_time() - start, results[0].start_ofv, best.history, rows)


def _explore_single(study, eta_indices, x0, decode, *, penalty=None, bounds=None, workers=1, cpu_budget_seconds=60.,
                    max_iterations=200, gradient_step=1e-3, gtol=1e-3, callback=None):
    """Minimize the Laplace objective from x0 within a CPU budget; never worse than x0."""
    if not np.isfinite([cpu_budget_seconds, gradient_step, gtol]).all() or cpu_budget_seconds <= 0 \
            or gradient_step <= 0 or gtol <= 0 or type(max_iterations) is not int or max_iterations < 1:
        raise ValueError('positive finite Laplace exploration settings required')
    start = time.process_time()
    x0 = np.asarray(x0, dtype=float).copy()
    penalty = penalty or (lambda point: 0.)
    settings = InnerSettings(method='compiled-BFGS', refine_modes=True, curvature='observed')
    box = bounds or [(None, None)] * len(x0)
    lower = np.array([-np.inf if b[0] is None else b[0] for b in box])
    upper = np.array([np.inf if b[1] is None else b[1] for b in box])
    x0 = np.clip(x0, lower, upper)
    best = dict(x=x0.copy(), ofv=np.inf)
    counts = dict(evaluations=0)
    history = []

    with PackedEvaluator(study, eta_indices, workers=workers, settings=settings) as evaluator:
        def value(point):
            if time.process_time() - start >= cpu_budget_seconds:
                raise _Budget()
            counts['evaluations'] += 1
            try:
                total, _ = evaluator.evaluate(*decode(point))
                total += penalty(point)
            except (ValueError, RuntimeError, ArithmeticError, OverflowError, FloatingPointError):
                return np.inf
            if not math.isfinite(total):
                return np.inf
            if total < best['ofv']:
                best.update(x=point.copy(), ofv=total)
            return total

        def value_gradient(point):
            f0 = value(point)
            if not np.isfinite(f0):
                return f0, np.zeros_like(point)
            grad = np.zeros_like(point)
            for j in range(len(point)):
                h = gradient_step * max(1., abs(point[j]))
                a, b = point.copy(), point.copy()
                a[j] = min(point[j] + h, upper[j]); b[j] = max(point[j] - h, lower[j])
                fa, fb = value(a), value(b)
                if not (np.isfinite(fa) and np.isfinite(fb)) or a[j] == b[j]:
                    return np.inf, np.zeros_like(point)
                grad[j] = (fa - fb) / (a[j] - b[j])
            return f0, grad

        try:
            start_ofv = value(x0)
        except _Budget:
            start_ofv = float('nan')
        if not np.isfinite(start_ofv):
            return LaplaceResult(x0, float('nan'), 'not_applicable', 'Laplace objective not finite at the start',
                                 0, counts['evaluations'], time.process_time() - start, float('nan'), history)

        def monitor(point):
            history.append(dict(iteration=len(history) + 1, ofv=float(best['ofv']), cpu_seconds=time.process_time() - start))
            if callback is not None:
                callback(dict(phase='laplace_exploration', **history[-1]))

        status, message, iterations = 'partial', '', 0
        try:
            fit = minimize(value_gradient, x0, jac=True, method='L-BFGS-B',
                           bounds=[(None if not np.isfinite(lo) else lo, None if not np.isfinite(hi) else hi)
                                   for lo, hi in zip(lower, upper)],
                           callback=monitor, options=dict(maxiter=max_iterations, gtol=gtol, ftol=1e-10, maxls=30))
            iterations = int(fit.nit)
            status = 'converged' if fit.success else 'partial'
            message = str(fit.message)
        except _Budget:
            iterations = len(history)
            message = 'CPU budget exhausted'
    x = best['x'] if best['ofv'] <= start_ofv else x0
    return LaplaceResult(x, float(min(best['ofv'], start_ofv)), status, message, iterations, counts['evaluations'],
                         time.process_time() - start, float(start_ofv), history)
