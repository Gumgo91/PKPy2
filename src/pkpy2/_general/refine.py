"""Staged importance refinement with independent validation and a two-bank audit.

Each stage minimizes the fixed-particle objective inside a trust box (by default one
quasi-Newton step with damped BFGS curvature updates; 'lbfgs' runs L-BFGS-B), validates
the candidate on an independently drawn bank at both endpoints, and accepts it
only when the two banks agree and the objective does not worsen. A point is
reported as converged when two further independent banks agree on the OFV
within ofv_tolerance, every subject's effective sample size is at least
minimum_ess, and each projected score is within its tolerance: score_tolerance,
or score_tolerance / SE for a coordinate whose local standard error SE (from
the quasi-Newton curvature) is below one, i.e. the estimate lies within
score_tolerance / 2 standard errors of the stationary point in that coordinate.
"""
import time
import numpy as np
from scipy.optimize import minimize
from .importance import Bank


def projected_score(x, g, bounds):
    out = np.array(g, dtype=float, copy=True)
    for j, (lower, upper) in enumerate(bounds):
        if lower is not None and x[j] <= lower + 1e-7 and out[j] > 0:
            out[j] = 0.
        if upper is not None and x[j] >= upper - 1e-7 and out[j] < 0:
            out[j] = 0.
    return out


def score_tolerances(hessian, n, score_tolerance):
    """Per-coordinate score tolerance: score_tolerance * max(1, 1 / SE) with SE = sqrt(2 [H^-1]_jj)."""
    if hessian is None:
        return np.full(n, score_tolerance)
    try:
        se = np.sqrt(np.maximum(2. * np.diag(np.linalg.inv(hessian)), 1e-300))
    except np.linalg.LinAlgError:
        return np.full(n, score_tolerance)
    return score_tolerance * np.maximum(1., 1. / se)


def coordinate_scale(problem):
    scale = np.ones(len(problem.labels))
    for k, e in enumerate(problem.effects):
        c = problem.beta_coord[k]
        if c < 0:
            continue
        if e['time_varying']:
            z = np.concatenate([zt[:, k] for zt in problem.z_tv])
        else:
            z = problem.z_tic[:, k]
        sd = float(np.std(z))
        scale[c] = 1. / max(sd, 1e-3) if sd > 0 else 1.
    return scale


def curvature(bank, x, bounds, step=1e-3):
    """Positive-definite curvature of the fixed-bank OFV from central differences of its gradient."""
    n = len(x)
    H = np.zeros((n, n))
    for j in range(n):
        h = step * max(1., abs(x[j]))
        lo, hi = bounds[j]
        plus, minus = x.copy(), x.copy()
        plus[j] = x[j] + h if hi is None else min(x[j] + h, hi)
        minus[j] = x[j] - h if lo is None else max(x[j] - h, lo)
        span = plus[j] - minus[j]
        if span <= 0:
            H[j, j] = 1.
            continue
        _, gp = bank.evaluate(plus)
        _, gm = bank.evaluate(minus)
        H[:, j] = (gp - gm) / span
    H = .5 * (H + H.T)
    w, vectors = np.linalg.eigh(H)
    w = np.maximum(w, 1e-6 * max(float(np.max(np.abs(w))), 1.))
    return (vectors * w) @ vectors.T


def damped_bfgs(H, s, y):
    """BFGS update with Powell damping, which keeps the curvature positive definite."""
    Hs = H @ s
    sHs = float(s @ Hs)
    if not sHs > 0:
        return H
    sy = float(s @ y)
    theta = 1. if sy >= .2 * sHs else .8 * sHs / (sHs - sy)
    r = theta * y + (1. - theta) * Hs
    return H - np.outer(Hs, Hs) / sHs + np.outer(r, r) / float(s @ r)


def quadratic_step(g, H, x, box):
    """Minimizer of g's + s'Hs/2 within the box (no objective evaluations)."""
    limits = [(lo - x[j], hi - x[j]) for j, (lo, hi) in enumerate(box)]
    fit = minimize(lambda d: (float(g @ d + .5 * d @ H @ d), g + H @ d), np.zeros(len(x)), jac=True,
                   method='L-BFGS-B', bounds=limits, options=dict(maxiter=500, ftol=1e-15, gtol=1e-12))
    return np.array(fit.x, dtype=float)


def refine(problem, x0, *, seed=0, power=10, audit_power=14, max_power=16, score_tolerance=.2, ofv_tolerance=.05,
           minimum_ess=100, max_stages=40, stage_iterations=40, radius=.5, max_radius=4., wall_budget_seconds=3600.,
           step_method='newton', callback=None):
    if step_method not in ('newton', 'lbfgs'):
        raise ValueError("step_method must be 'newton' or 'lbfgs'")
    start = time.perf_counter()
    hessian = None
    x = np.array(x0, dtype=float)
    bounds = problem.bounds
    random = problem.q + problem.Q > 0
    scale = coordinate_scale(problem)
    stages = []
    audit = {}
    status, message = 'partial', 'stage budget exhausted'
    value = np.inf
    bank = Bank(problem, x, power=power, seed=seed)

    def budget():
        if time.perf_counter() - start > wall_budget_seconds:
            raise TimeoutError('wall-clock budget exhausted')

    def run_audit(point, stage, level):
        tol = score_tolerances(hessian, len(point), score_tolerance)
        replicas, banks = [], []
        for rep in range(2):
            budget()
            b = Bank(problem, point, power=level, seed=seed + 9000001 + 2 * stage + rep)
            v, g = b.evaluate(point)
            pg = projected_score(point, g, bounds)
            replicas.append(dict(seed=seed + 9000001 + 2 * stage + rep, ofv=float(v), score=g.tolist(),
                                 projected_score_max=float(np.max(np.abs(pg))),
                                 scaled_score_max=float(np.max(np.abs(pg) / tol)), minimum_ess=float(b.last['min_ess'])))
            banks.append(b)
        result = dict(x=point.tolist(), replicas=replicas, ofv_range=abs(replicas[0]['ofv'] - replicas[1]['ofv']),
                      power=level, score_tolerances=tol.tolist())
        result['passed'] = bool(result['ofv_range'] <= ofv_tolerance and all(
            r['scaled_score_max'] <= 1. and (not random or r['minimum_ess'] >= minimum_ess) for r in replicas))
        return result, banks

    try:
        for stage in range(max_stages):
            budget()
            f0, g0 = bank.evaluate(x)
            ess0 = bank.last['min_ess'] if bank.last else 0.
            if not np.isfinite(f0):
                message = 'objective not finite at the retained point'
                break
            box = []
            for j in range(len(x)):
                lo, hi = bounds[j]
                a, b = x[j] - radius * scale[j], x[j] + radius * scale[j]
                box.append((a if lo is None else max(a, lo), b if hi is None else min(b, hi)))
            if step_method == 'newton':
                # Quasi-Newton step on the working bank: curvature from gradient differences once,
                # then damped BFGS updates; the step is shortened until the bank objective decreases.
                if hessian is None:
                    hessian = curvature(bank, x, bounds)
                    f0, g0 = bank.evaluate(x)
                step = quadratic_step(g0, hessian, x, box)
                iterations = 0
                for iterations in range(8):
                    fc_bank, gc_bank = bank.evaluate(x + step)
                    if (np.isfinite(fc_bank) and fc_bank <= f0 + 1e-4 * float(g0 @ step)) or iterations == 7:
                        break
                    step = .5 * step
                if np.isfinite(fc_bank) and np.any(step != 0):
                    hessian = damped_bfgs(hessian, step, gc_bank - g0)
                candidate = x + step
                optimizer_message = 'quasi-Newton step'
            else:
                fit = minimize(lambda z: bank.evaluate(z), x, jac=True, method='L-BFGS-B', bounds=box,
                               options=dict(maxiter=stage_iterations, ftol=1e-12, gtol=1e-5, maxls=20))
                candidate = np.array(fit.x, dtype=float)
                iterations, optimizer_message = int(fit.nit), str(fit.message)
                fc_bank, _ = bank.evaluate(candidate)
            essc_bank = bank.last['min_ess'] if bank.last else 0.
            budget()
            fresh = Bank(problem, candidate, power=power, seed=seed + 1000003 * (stage + 1))
            fx_fresh, _ = fresh.evaluate(x, gradient=False)
            essx_fresh = fresh.last['min_ess'] if fresh.last else 0.
            fc_fresh, gc_fresh = fresh.evaluate(candidate)
            essc_fresh = fresh.last['min_ess'] if fresh.last else 0.
            predicted = f0 - fc_bank
            actual = fx_fresh - fc_fresh
            gap = fc_fresh - fc_bank
            tol = max(ofv_tolerance, .1 * max(predicted, 0.))
            adequate = (not random) or min(ess0, essc_bank, essx_fresh, essc_fresh) >= minimum_ess
            agreement = abs(gap) <= tol and abs(fx_fresh - f0) <= tol
            accepted = bool(np.isfinite(fc_fresh) and adequate and agreement and actual >= -ofv_tolerance)
            pg = projected_score(candidate, gc_fresh, bounds) if accepted else np.full(len(x), np.nan)
            row = dict(stage=stage, power=int(power), radius=float(radius), accepted=accepted,
                       before_ofv=float(f0), fixed_sample_ofv=float(fc_bank), independent_ofv=float(fc_fresh),
                       paired_improvement=float(actual), independent_gap=float(gap),
                       minimum_ess=float(min(ess0, essc_bank, essx_fresh, essc_fresh)),
                       projected_score_max=float(np.nanmax(np.abs(pg))) if accepted else None,
                       scaled_score_max=float(np.nanmax(np.abs(pg) / score_tolerances(hessian, len(x), score_tolerance)))
                       if accepted else None,
                       optimizer_message=optimizer_message, iterations=int(iterations), x=candidate.tolist(),
                       seconds=time.perf_counter() - start)
            stages.append(row)
            if callback is not None:
                callback(row)
            if accepted:
                ratio = actual / predicted if predicted > 1e-8 else 0.
                x = candidate
                value = float(fc_fresh)
                bank = fresh
                if actual > ofv_tolerance and ratio > .75:
                    radius = min(max_radius, radius * 1.5)
                elif ratio < .25 and predicted > ofv_tolerance:
                    radius = max(.025, radius * .5)
                if np.max(np.abs(pg) / score_tolerances(hessian, len(x), score_tolerance)) < 2.5:
                    level = max(audit_power, power)
                    audit, banks = run_audit(x, stage, level)
                    row['audit'] = audit
                    if audit['passed']:
                        value = float(np.mean([r['ofv'] for r in audit['replicas']]))
                        status = 'converged'
                        message = 'two independent integrations agree and projected marginal scores pass'
                        break
                    bank = banks[0]
                    power = max(power, level)
            else:
                if not adequate:
                    # too few effective particles at one endpoint: shorter steps and more particles
                    radius = max(.025, radius * .5)
                if not adequate or not agreement:
                    if power >= max_power:
                        message = 'integration precision exhausted before a step decision'
                        break
                    power += 1
                    bank = Bank(problem, x, power=power, seed=seed + 700001 + stage)
                else:
                    radius = max(.025, radius * .5)
    except (TimeoutError, ValueError, RuntimeError, ArithmeticError, np.linalg.LinAlgError) as error:
        message = f'{type(error).__name__}: {error}'
    return dict(x=x, ofv=value, status=status, message=message, stages=stages, audit=audit,
                seconds=time.perf_counter() - start)
