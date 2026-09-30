"""Observed marginal information, robust (sandwich) covariance and reporting-scale intervals.

The Hessian of the OFV is formed from central differences of the marginal
gradient on two independent banks and two step sizes, as for the classic engine.
Covariance of the coordinates is 2 H^-1. The sandwich covariance is
H^-1 (sum_i s_i s_i') H^-1 * 4 / ... expressed for the log-likelihood:
I = H/2, S = sum_i u_i u_i' with u_i = -1/2 dOFV_i/dx, cov_robust = I^-1 S I^-1.
Reporting-scale quantities (typical values, variances, covariances,
correlations, residual SDs) use the delta method with numerical Jacobians.
"""
import time
import numpy as np
from scipy.stats import norm
from .importance import Bank
from .problem import from_transformed


def _hessian(bank, x, step, bounds, prediction_coords=()):
    """Central differences of the bank gradient. Columns of density coordinates use only the
    analytic gradient (their prediction-coordinate rows follow from symmetry); columns of
    prediction coordinates use the full gradient."""
    n = len(x)
    pred = sorted(prediction_coords)
    dens = [k for k in range(n) if k not in set(pred)]
    matrix = np.zeros((n, n))
    for j in range(n):
        h = step * max(1., abs(x[j]))
        lo, hi = bounds[j]
        if (lo is not None and x[j] - h <= lo) or (hi is not None and x[j] + h >= hi):
            raise ValueError(f'curvature neighborhood touches a bound: {j}')
        plus, minus = x.copy(), x.copy()
        plus[j] += h
        minus[j] -= h
        full = j in pred
        column = (bank.evaluate(plus, finite_differences=full)[1]
                  - bank.evaluate(minus, finite_differences=full)[1]) / (2 * h)
        if full:
            matrix[:, j] = column
        else:
            matrix[dens, j] = column[dens]
    for j in dens:
        for k in pred:
            matrix[k, j] = matrix[j, k]
    asymmetry = float(np.linalg.norm(matrix - matrix.T) / max(np.linalg.norm(matrix), 1e-30))
    return .5 * (matrix + matrix.T), asymmetry


def reporting_quantities(problem, x):
    """Named reporting-scale quantities as a function of the coordinates."""
    pop = problem.population(x)
    out = {}
    for j, n in enumerate(problem.names):
        if problem.theta_coord[j] >= 0:
            out[f'theta:{n}'] = from_transformed(pop.theta_t[j], problem.transform_names[j])
    for k, e in enumerate(problem.effects):
        if problem.beta_coord[k] >= 0:
            out[f"beta:{e['label']}"] = float(pop.beta[k])
    for g in problem.omega_groups:
        for a in g:
            out[f'omega:{problem.iiv_names[a]}'] = float(pop.omega[a, a])
        for a in g:
            for b in g:
                if b < a:
                    cov = float(pop.omega[a, b])
                    out[f'omega_cov:{problem.iiv_names[a]},{problem.iiv_names[b]}'] = cov
                    out[f'omega_corr:{problem.iiv_names[a]},{problem.iiv_names[b]}'] = cov / np.sqrt(
                        pop.omega[a, a] * pop.omega[b, b])
    for j, n in enumerate(problem.iov_names):
        out[f'iov:{n}'] = float(pop.iov_sd[j] ** 2)
    for o, name in enumerate(problem.structure.output_names):
        for c, comp in enumerate(('proportional', 'additive', 'lognormal')):
            if problem.sigma_coord[o, c] >= 0:
                out[f'sigma:{name}:{comp}'] = float([pop.sigma_prop, pop.sigma_add, pop.sigma_ln][c][o])
    return out


def delta_intervals(problem, x, cov, confidence):
    z = norm.ppf((1 + confidence) / 2)
    base = reporting_quantities(problem, x)
    names = list(base)
    jac = np.zeros((len(names), len(x)))
    for j in range(len(x)):
        h = 1e-6 * max(1., abs(x[j]))
        p, m = x.copy(), x.copy()
        p[j] += h
        m[j] -= h
        qp, qm = reporting_quantities(problem, p), reporting_quantities(problem, m)
        jac[:, j] = [(qp[n] - qm[n]) / (2 * h) for n in names]
    rows = []
    var = np.einsum('ij,jk,ik->i', jac, cov, jac)
    for k, n in enumerate(names):
        est = base[n]
        se = float(np.sqrt(max(var[k], 0.)))
        coord = n if n in problem.labels else None
        if coord is not None and (n.startswith('theta:') and problem.transform_names[problem.names.index(n[6:])] == 'log'):
            j = problem.labels.index(coord)
            sd = np.sqrt(cov[j, j])
            interval = [float(np.exp(x[j] - z * sd)), float(np.exp(x[j] + z * sd))]
        elif n.startswith(('omega:', 'iov:', 'sigma:')) and est > 0:
            sd_log = se / est
            interval = [float(est * np.exp(-z * sd_log)), float(est * np.exp(z * sd_log))]
        else:
            interval = [float(est - z * se), float(est + z * se)]
        rows.append(dict(quantity=n, estimate=float(est), se=se,
                         rse_pct=float(100 * se / abs(est)) if est != 0 else None, interval=interval))
    return rows


def _laplace_hessian(problem, x, step, scale, modes=None, exact=True):
    """Central differences of the Laplace gradient (exact: modes re-solved at every difference point) with the
    conditional modes warm-started at x; step j is step * max(scale_j, |x_j|) with scale_j the refinement's
    coordinate scale."""
    from .laplace import LaplaceObjective
    objective = LaplaceObjective(problem)
    _, base = objective.states(x, starts=modes)
    if base is None:
        raise ValueError('the Laplace objective is not finite at the estimate')
    n = len(x)
    matrix = np.zeros((n, n))
    for j in range(n):
        h = step * max(scale[j], abs(x[j]))
        lo, hi = problem.bounds[j]
        if (lo is not None and x[j] - h <= lo) or (hi is not None and x[j] + h >= hi):
            raise ValueError(f'curvature neighborhood touches a bound: {problem.labels[j]}')
        grads = []
        for sign in (1., -1.):
            point = x.copy()
            point[j] += sign * h
            objective.modes = list(base)
            value, g = objective.value_grad(point, problem.bounds, exact=exact)
            if not np.isfinite(value):
                raise ValueError(f'the Laplace objective is not finite near the estimate ({problem.labels[j]})')
            grads.append(g)
        matrix[:, j] = (grads[0] - grads[1]) / (2 * h)
    asymmetry = float(np.linalg.norm(matrix - matrix.T) / max(np.linalg.norm(matrix), 1e-30))
    return .5 * (matrix + matrix.T), asymmetry


def laplace_uncertainty(problem, x, fit_ofv, *, modes=None, confidence=.95, step=.002, exact=True, callback=None):
    """Covariance 2 H^-1 from the curvature H of the Laplace objective (the analogue of a NONMEM $COV step after
    FOCE-I), from differences of the exact gradient (exact=False: modes held fixed, faster but possibly biased),
    checked at two step sizes; reporting-scale intervals by the delta method."""
    from .refine import coordinate_scale
    start = time.perf_counter()
    x = np.asarray(x, dtype=float)
    scale = coordinate_scale(problem)
    try:
        coarse, a1 = _laplace_hessian(problem, x, step, scale, modes, exact)
        if callback:
            callback(dict(stage='laplace_curvature', step=step, seconds=time.perf_counter() - start))
        fine, a2 = _laplace_hessian(problem, x, step / 2, scale, modes, exact)
    except ValueError as error:
        status = 'boundary_estimate' if 'bound' in str(error) else 'unresolved_curvature'
        return dict(status=status, method='laplace_observed_information', message=str(error), intervals=[],
                    seconds=time.perf_counter() - start)
    h = fine
    norm_h = max(np.linalg.norm(h), 1e-30)
    step_change = float(np.linalg.norm(coarse - fine) / norm_h)
    values = np.linalg.eigvalsh(h)
    report = dict(method='laplace_observed_information', confidence=confidence, coordinates=problem.labels,
                  point=x.tolist(), hessian=h.tolist(), eigenvalues=values.tolist(), relative_step_change=step_change,
                  asymmetry=[a1, a2], ofv=float(fit_ofv))
    if values[0] <= 0:
        report.update(status='unresolved_curvature', intervals=[], seconds=time.perf_counter() - start)
        return report
    cov = 2. * np.linalg.inv(h)
    sd = np.sqrt(np.diag(cov))
    se_change = float(np.max(np.abs(np.sqrt(np.diag(2. * np.linalg.inv(coarse))) / sd - 1))) \
        if np.linalg.eigvalsh(coarse)[0] > 0 else np.inf
    stable = bool(step_change <= .1 and se_change <= .2 and max(a1, a2) <= .1)
    report.update(status='computed' if stable else 'unresolved_curvature', numerically_stable=stable,
                  maximum_relative_se_change=se_change, covariance=cov.tolist(),
                  correlation=(cov / np.outer(sd, sd)).tolist(),
                  intervals=delta_intervals(problem, x, cov, confidence) if stable else [],
                  seconds=time.perf_counter() - start)
    return report


def estimate_uncertainty(problem, x, fit_ofv, *, confidence=.95, power=14, step=.002, seed=41000001, callback=None):
    start = time.perf_counter()
    x = np.asarray(x, dtype=float)
    boundary = [problem.labels[j] for j, (lo, hi) in enumerate(problem.bounds)
                if (lo is not None and x[j] - step * max(1., abs(x[j])) <= lo)
                or (hi is not None and x[j] + step * max(1., abs(x[j])) >= hi)]
    if boundary:
        return dict(status='boundary_estimate', boundary_coordinates=boundary, intervals=[],
                    seconds=time.perf_counter() - start)
    matrices, records, scores = [], [], []
    for rep in range(2):
        bank = Bank(problem, x, power=power, seed=seed + rep)
        value, grad = bank.evaluate(x)
        per = bank.last['per_subject_score']
        scores.append(per)
        min_ess = bank.last['min_ess']
        coarse, a1 = _hessian(bank, x, step, problem.bounds, problem.prediction_coords)
        fine, a2 = _hessian(bank, x, step / 2, problem.bounds, problem.prediction_coords)
        matrices.append((coarse, fine))
        records.append(dict(seed=seed + rep, ofv=float(value), minimum_ess=float(min_ess), asymmetry=[a1, a2],
                            score=grad.tolist()))
        if callback:
            callback(dict(stage='curvature_replica', replica=rep, seconds=time.perf_counter() - start))
    h = .5 * (matrices[0][1] + matrices[1][1])
    norm_h = max(np.linalg.norm(h), 1e-30)
    step_change = max(np.linalg.norm(a - b) / norm_h for a, b in matrices)
    bank_change = np.linalg.norm(matrices[0][1] - matrices[1][1]) / norm_h
    values = np.linalg.eigvalsh(h)
    report = dict(method='importance_sampled_marginal_information', confidence=confidence, coordinates=problem.labels,
                  point=x.tolist(), hessian=h.tolist(), eigenvalues=values.tolist(),
                  relative_step_change=float(step_change), relative_bank_change=float(bank_change),
                  source_ofv_gap=float(max(abs(r['ofv'] - fit_ofv) for r in records)), replicas=records, power=power)
    if values[0] <= 0:
        report.update(status='unresolved_curvature', intervals=[], seconds=time.perf_counter() - start)
        return report
    cov = 2. * np.linalg.inv(h)
    covs = [2. * np.linalg.inv(m) for pair in matrices for m in pair if np.linalg.eigvalsh(m)[0] > 0]
    sd = np.sqrt(np.diag(cov))
    se_change = max(float(np.max(np.abs(np.sqrt(np.diag(c)) / sd - 1))) for c in covs) if len(covs) == 4 else np.inf
    stable = bool(step_change <= .1 and bank_change <= .1 and se_change <= .2 and report['source_ofv_gap'] <= .05
                  and min(r['minimum_ess'] for r in records) >= 100 and max(max(r['asymmetry']) for r in records) <= .1)
    # sandwich: u_i = -dOFV_i/dx / 2 (log-likelihood scores); information I = H/2
    u = -.5 * .5 * (scores[0] + scores[1])
    meat = u.T @ u
    info_inv = np.linalg.inv(.5 * h)
    robust = info_inv @ meat @ info_inv
    report.update(status='computed' if stable else 'unresolved_curvature', numerically_stable=stable,
                  maximum_relative_se_change=se_change, covariance=cov.tolist(), robust_covariance=robust.tolist(),
                  correlation=(cov / np.outer(sd, sd)).tolist(),
                  intervals=delta_intervals(problem, x, cov, confidence) if stable else [],
                  robust_intervals=delta_intervals(problem, x, robust, confidence) if stable else [],
                  seconds=time.perf_counter() - start)
    return report
