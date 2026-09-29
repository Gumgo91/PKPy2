"""Post-fit marginal information, local intervals and practical identifiability.

No refit, model selection, diagonal inverse, jitter or eigenvalue clipping.
Independent QMC banks and two difference steps audit the full observed Hessian.
Likelihood curvature at a penalized estimate is explicitly provisional; penalized
curvature intervals are Laplace intervals, not frequentist confidence intervals.
"""
import copy
import time
import numpy as np
from scipy.stats import norm


def _gradient_hessian(evaluate, x, step, bounds):
    """Central gradient differences; boundary neighborhoods are not regularized."""
    matrix = np.empty((len(x), len(x)))
    for j in range(len(x)):
        h = step*max(1., abs(x[j]))
        lower, upper = bounds[j]
        if ((lower is not None and x[j]-h <= lower)
                or (upper is not None and x[j]+h >= upper)):
            raise ValueError('curvature neighborhood touches boundary: '+str(j))
        plus = x.copy(); minus = x.copy(); plus[j] += h; minus[j] -= h
        matrix[:, j] = (evaluate(plus)[1]-evaluate(minus)[1])/(2*h)
    asymmetry = float(np.linalg.norm(matrix-matrix.T)/max(np.linalg.norm(matrix), 1e-30))
    return .5*(matrix+matrix.T), asymmetry


def _information(matrix, labels):
    values, vectors = np.linalg.eigh(matrix*.5)
    scale = max(float(np.max(np.abs(values))), 1e-30)
    weak = []
    for j in range(min(3, len(values))):
        ix = np.argsort(np.abs(vectors[:, j]))[::-1][:4]
        weak.append(dict(eigenvalue=float(values[j]),
                         loadings={labels[i]:float(vectors[i, j]) for i in ix}))
    report = dict(eigenvalues=values.tolist(), positive_definite=bool(values[0]>0),
                  numerical_rank=int(np.sum(values>scale*1e-10)),
                  coordinate_system='log positive parameters, raw covariate coefficients',
                  condition_number=float(values[-1]/values[0]) if values[0]>0 else None,
                  weakest_directions=weak, covariance=None, correlation=None)
    if values[0] > 0:
        cov = 2*np.linalg.solve(matrix, np.eye(len(matrix)))
        sd = np.sqrt(np.diag(cov))
        report.update(covariance=cov.tolist(), correlation=(cov/np.outer(sd, sd)).tolist())
    return report


def _intervals(x, covariance, labels, confidence):
    z = norm.ppf((1+confidence)/2); sd = np.sqrt(np.diag(covariance)); rows = []
    for j, label in enumerate(labels):
        factor = 2. if label.startswith('log_omega_sd:') else 1.
        positive = label.startswith('log_')
        estimate = float(np.exp(factor*x[j])) if positive else float(x[j])
        lo, hi = factor*(x[j]-z*sd[j]), factor*(x[j]+z*sd[j])
        finite = not positive or (-700 < lo < hi < 700)
        interval = ([float(np.exp(lo)), float(np.exp(hi))] if positive else [float(lo), float(hi)]) if finite else None
        rows.append(dict(coordinate=label, estimate=estimate,
                         se=float(estimate*factor*sd[j]) if positive else float(sd[j]),
                         rse_pct=float(100*factor*sd[j]) if positive else None,
                         interval=interval, interval_numerically_representable=bool(finite),
                         large_local_uncertainty=bool(positive and factor*sd[j]>.5)))
    return rows


def estimate_uncertainty(problem, fit, *, confidence=.95, power=16, step=.002,
                         workers=4, seed=41000001, callback=None):
    """Return a separate report; never mutate or refit the selected FitResult.

    Full data and penalized Hessians are distinguished. Returned Wald intervals
    are local, conditional on the selected model and evaluated at the saved
    estimate. They do not account for selection, boundary distributions, prior
    bias, structural nonidentifiability or small-sample coverage.
    """
    from .importance_objective import PhysicalImportance
    from .subject_work import SubjectWork
    if not 0 < confidence < 1 or not np.isfinite(step) or step <= 0:
        raise ValueError('invalid confidence or finite-difference step')
    start = time.perf_counter()
    study, indices, x, decode, bounds, labels = problem.study, problem.indices, fit.x, problem.decode, problem.bounds, problem.labels
    # A Wald interval is undefined when the estimate lies on (or within one
    # finite-difference step of) a declared bound; report this instead.
    boundary = [labels[j] for j, (lower, upper) in enumerate(bounds)
                if (lower is not None and x[j]-step*max(1., abs(x[j])) <= lower)
                or (upper is not None and x[j]+step*max(1., abs(x[j])) >= upper)]
    if boundary:
        return dict(method='independent_QMC_full_marginal_information', confidence=float(confidence),
                    status='boundary_estimate', boundary_coordinates=boundary, coordinates=labels, point=x.tolist(),
                    model=problem.model, source_data_objective=fit.ofv, data_fingerprint=problem.data_sha256,
                    information=dict(data=dict(intervals=[], numerically_stable=False)),
                    interpretation='Estimate lies on a declared bound; local Wald intervals are not reported',
                    seconds=time.perf_counter()-start)
    penalty = lambda point: 0.
    hp = np.zeros((len(x), len(x)))
    work = SubjectWork(workers, len(study.subject_ids)); records = []; matrices = []
    try:
        for replica in range(2):
            bank = PhysicalImportance(study, indices, *decode(x), power=power,
                                      seed=seed+replica, pilot_power=14, work=work)
            min_ess = np.inf
            def evaluate(point):
                nonlocal min_ess
                value, gradient = bank.value_gradient(point, decode, bounds=bounds, analytic_non_eta=True)
                min_ess = min(min_ess, bank.last_min_ess)
                return value, gradient
            value, gradient = evaluate(x)
            coarse, asym_coarse = _gradient_hessian(evaluate, x, step, bounds)
            fine, asym_fine = _gradient_hessian(evaluate, x, step/2, bounds)
            matrices.append((coarse, fine))
            records.append(dict(seed=seed+replica, data_objective=value, data_score=gradient.tolist(),
                                minimum_ess=float(min_ess), asymmetry=[asym_coarse, asym_fine]))
            if callback: callback(dict(stage='curvature_replica', replica=replica, seconds=time.perf_counter()-start))
    finally:
        work.close()
    h = .5*(matrices[0][1]+matrices[1][1]); norm_h = max(np.linalg.norm(h), 1e-30)
    source_gap = max(abs(r['data_objective']-fit.ofv) for r in records)
    step_change = max(np.linalg.norm(a-b)/norm_h for a,b in matrices)
    bank_change = np.linalg.norm(matrices[0][1]-matrices[1][1])/norm_h
    reports = {}
    for name, extra in [('data', np.zeros_like(h)), ('penalized', hp)]:
        info = _information(h+extra, labels)
        candidates = [_information(a+extra, labels) for pair in matrices for a in pair]
        se_change = None
        if info['covariance'] is not None and all(c['covariance'] is not None for c in candidates):
            sd = np.sqrt(np.diag(info['covariance']))
            se_change = float(max(np.max(np.abs(np.sqrt(np.diag(c['covariance']))/sd-1)) for c in candidates))
        stable = bool(step_change<=.1 and bank_change<=.1 and se_change is not None and se_change<=.2
                      and source_gap<=.05
                      and min(r['minimum_ess'] for r in records)>=100
                      and max(max(r['asymmetry']) for r in records)<=.1)
        info.update(numerically_stable=stable, maximum_relative_se_change=se_change,
                    intervals=_intervals(x, info['covariance'], labels, confidence) if stable else [],
                    interpretation=('local likelihood Wald approximation at saved estimate' if name=='data'
                                    else 'local penalized/Laplace interval; not a frequentist confidence interval'))
        reports[name] = info
    data_intervals = reports['data']['intervals']
    issues = [r['coordinate'] for r in data_intervals if r['large_local_uncertainty']
              or not r['interval_numerically_representable']]
    return dict(method='independent_QMC_full_marginal_information', confidence=float(confidence),
                status='computed' if reports['data']['numerically_stable'] else 'unresolved_curvature',
                coordinates=labels, point=x.tolist(), model=problem.model,
                covariate_effect_specs=copy.deepcopy(problem.spec["effects"]),
                source_data_objective=fit.ofv,
                selected_model_conditional=True, selection_uncertainty_accounted=False,
                coverage_calibrated=False, structural_identifiability_proven=False,
                estimate_is_penalized=bool(np.any(hp)), data_fingerprint=problem.data_sha256,
                prior_penalty=penalty(x), prior_hessian=hp.tolist(), data_hessian=h.tolist(),
                replicas=records, step=step, sample_power=power,
                relative_step_change=float(step_change), relative_bank_change=float(bank_change),
                maximum_source_objective_gap=float(source_gap),
                practical_identifiability=dict(
                    status=('local_precision_concerns' if issues else
                            'local_curvature_resolved' if reports['data']['numerically_stable'] else 'unresolved'),
                    flagged_coordinates=issues, large_rse_advisory_threshold_pct=50.,
                    advisory_threshold_calibrated=False,
                    interpretation='Local precision diagnostic; not a proof of structural or global identifiability'),
                information=reports, seconds=time.perf_counter()-start)

