"""Uncertainty and model-building tools shared by the classic and general engines.

profile():  profile likelihood of one parameter; the 95% interval is where the
            refitted OFV exceeds its minimum by the chi-square(1) 95% quantile (3.84).
bootstrap(): nonparametric case bootstrap (subjects resampled with replacement),
            each replicate refitted from the original estimates.
sir():      sampling importance resampling (Dosne et al. 2016, 2017) with a multivariate
            normal proposal from the local covariance, updated over iterations from the
            resampled vectors; marginal OFVs come from one fixed importance bank.
robust_covariance(): sandwich covariance from per-subject scores.
likelihood_ratio_test(), compare(): nested-model tests and information criteria.
stepwise_covariates(): forward inclusion (p < forward_alpha) then backward
            elimination (p < backward_alpha) by likelihood-ratio tests.
"""
import copy
import dataclasses
import math
import numpy as np
from scipy.stats import chi2


# ------------------------------------------------------------------ specifications

def _is_general(result):
    return result.problem.__class__.__name__ == 'GeneralProblem'


def data_of(result):
    if _is_general(result):
        return result.problem.dataset
    return list(result.problem.subjects)


def specification_of(result, *, at_estimate=True):
    """The fitted specification, with starting values at the estimates when at_estimate."""
    from .api import ModelSpec, Parameter, Covariate
    if _is_general(result):
        model = copy.deepcopy(result.problem.model)
        if not at_estimate:
            return model
        d = result.problem.describe(result.x)
        model.theta = {n: dataclasses.replace(p, value=float(d['theta'][n])) for n, p in model.theta.items()}
        model.omega = {n: dataclasses.replace(p, value=float(d['omega'][n])) for n, p in model.omega.items()}
        cov = {}
        for key, p in model.omega_covariance.items():
            a, b = key
            value = d['omega_covariance'].get(f'{a},{b}', d['omega_covariance'].get(f'{b},{a}', p.value))
            cov[key] = dataclasses.replace(p, value=float(value))
        for blk in model.omega_blocks:
            for x_, a in enumerate(blk):
                for b in blk[:x_]:
                    if (a, b) not in cov and (b, a) not in cov:
                        value = d['omega_covariance'].get(f'{a},{b}', d['omega_covariance'].get(f'{b},{a}', 0.))
                        cov[(a, b)] = Parameter(float(value))
        model.omega_covariance = cov
        model.iov = {n: dataclasses.replace(p, value=float(d['iov'][n])) for n, p in model.iov.items()}
        effects = list(model.covariates)
        labels = [e['label'] for e in result.problem.effects]
        model.covariates = tuple(dataclasses.replace(c, coefficient=dataclasses.replace(
            c.coefficient, value=float(d['coefficients'][labels[k]]))) for k, c in enumerate(effects))
        from .model import Residual
        residual = {}
        for name, r in model.residual.items():
            kw = {}
            for comp, p in r.components():
                key = f'{name}:{comp}'
                if comp == 'lognormal' or getattr(r, comp) is not None:
                    kw[comp] = dataclasses.replace(p, value=float(d['sigma'].get(key, p.value)))
            residual[name] = Residual(**kw)
        model.residual = residual
        return model
    cm = result.problem
    spec = cm.spec
    bounds = spec.get('bounds', {})

    def par(group, name, value, fixed):
        lo, hi = bounds.get(group, {}).get(name, (None, None))
        return Parameter(float(value), fixed, lo, hi)
    theta = {n: par('theta', n, v if at_estimate else cm.theta[n], n in spec['fixed_theta']) for n, v in result.theta.items()}
    omega = {n: par('omega', n, v if at_estimate else spec['omega'][n], n in spec['fixed_omega'])
             for n, v in result.omega.items()}
    sig = result.sigma if at_estimate else spec['sigma']
    covs = tuple(Covariate(e['parameter'], e['covariate'], e['center'],
                           Parameter(float(b if at_estimate else e['beta']), e['fixed'], e.get('lower'), e.get('upper')),
                           e.get('form', 'power'), e.get('level'))
                 for e, b in zip(spec['effects'], result.coefficients))
    return ModelSpec(result.model, theta, omega, par('sigma', 'sigma_prop', sig['sigma_prop'], 'sigma_prop' in spec['fixed_sigma']),
                     par('sigma', 'sigma_add', sig['sigma_add'], 'sigma_add' in spec['fixed_sigma']), covs,
                     spec['omega_floor'])


def _refit(data, spec, fit_options):
    from . import fit
    return fit(data, spec, **(fit_options or {}))


def _estimates(result):
    """Flat dict of reporting-scale estimates."""
    out = {f'theta:{k}': float(v) for k, v in result.theta.items()}
    out.update({f'omega:{k}': float(v) for k, v in result.omega.items()})
    if _is_general(result):
        out.update({f'omega_cov:{k}': float(v) for k, v in result.omega_covariance.items()})
        out.update({f'iov:{k}': float(v) for k, v in result.iov.items()})
        out.update({f'beta:{k}': float(v) for k, v in result.coefficients.items()})
        out.update({f'sigma:{k}': float(v) for k, v in result.sigma.items()})
    else:
        out.update({f'beta:{k}': float(v) for k, v in enumerate(result.coefficients)})
        out.update({f'sigma:{k}': float(v) for k, v in result.sigma.items()})
    return out


def _fix(spec, parameter, value):
    """Copy of spec with one quantity fixed at value (theta name, 'omega:NAME', 'sigma:...' or 'beta:k')."""
    from .api import Parameter
    if parameter in spec.theta or parameter.startswith('theta:'):
        name = parameter.split(':', 1)[-1]
        theta = dict(spec.theta)
        theta[name] = Parameter(float(value), True)
        return dataclasses.replace(spec, theta=theta)
    if parameter.startswith('omega:'):
        name = parameter.split(':', 1)[1]
        omega = dict(spec.omega)
        omega[name] = Parameter(float(value), True)
        return dataclasses.replace(spec, omega=omega)
    if parameter.startswith('beta:'):
        key = parameter.split(':', 1)[1]
        labels = [f'{c.parameter}~{c.covariate}' + (f'={c.level:g}' if c.level is not None else '')
                  for c in spec.covariates]
        k = labels.index(key) if key in labels else int(key)
        covs = list(spec.covariates)
        covs[k] = dataclasses.replace(covs[k], coefficient=Parameter(float(value), True))
        return dataclasses.replace(spec, covariates=tuple(covs))
    raise ValueError(f'profiles are supported for typical values, variances and covariate coefficients: {parameter}')


# ------------------------------------------------------------------ profile likelihood

def profile(result, parameter, values=None, *, points=9, span=.5, fit_options=None, level=.95):
    """Profile likelihood of `parameter` over `values` (default: estimate * exp(+-span) on a log grid)."""
    estimate = _estimates(result)
    key = parameter if ':' in parameter else f'theta:{parameter}'
    if key.startswith('beta:') and not _is_general(result):
        center = float(result.coefficients[int(key.split(':')[1])])
    else:
        center = estimate[key]
    if values is None:
        if key.startswith('beta:'):
            values = center + np.linspace(-span, span, points)
        else:
            values = center * np.exp(np.linspace(-span, span, points))
    values = np.sort(np.asarray(values, dtype=float))
    base = specification_of(result)
    data = data_of(result)
    options = dict(fit_options or {})
    if _is_general(result):
        options.setdefault('laplace_options', dict(starts=1))
    rows = []
    for v in values:
        fitted = _refit(data, _fix(base, key if not key.startswith('theta:') else key[6:], v), options)
        rows.append(dict(value=float(v), ofv=float(fitted.ofv), status=fitted.status, converged=bool(fitted.converged)))
    ofv = np.array([r['ofv'] for r in rows])
    reference = min(float(result.ofv), float(np.nanmin(ofv)))
    delta = ofv - reference
    threshold = chi2.ppf(level, 1)
    positive = not key.startswith('beta:') and np.all(values > 0)
    grid = np.log(values) if positive else values
    j0 = int(np.nanargmin(delta))

    def crossing(direction):
        j = j0
        while 0 <= j + direction < len(grid):
            a, b = j, j + direction
            if delta[a] <= threshold < delta[b]:
                point = grid[a] + (threshold - delta[a]) / (delta[b] - delta[a]) * (grid[b] - grid[a])
                return float(np.exp(point) if positive else point)
            j = b
        return None
    interval = [crossing(-1), crossing(1)]
    return dict(parameter=key, estimate=center, reference_ofv=reference, points=rows, delta_ofv=delta.tolist(),
                threshold=float(threshold), interval=interval, level=level)


# ------------------------------------------------------------------ bootstrap

def bootstrap(result, n=200, *, seed=20260930, fit_options=None, callback=None):
    from .data import Dataset
    rng = np.random.default_rng(seed)
    base = specification_of(result)
    data = data_of(result)
    options = dict(fit_options or {})
    if _is_general(result):
        options.setdefault('laplace_options', dict(starts=1))
    rows = []
    for r in range(n):
        if isinstance(data, Dataset):
            sample = data.resample(rng)
        else:
            picks = rng.integers(0, len(data), len(data))
            sample = [dataclasses.replace(data[j], sid=k + 1) for k, j in enumerate(picks)]
        try:
            fitted = _refit(sample, base, dict(options, seed=int(rng.integers(1, 2 ** 31))))
            row = dict(replicate=r, ofv=float(fitted.ofv), converged=bool(fitted.converged), status=fitted.status,
                       estimates=_estimates(fitted))
        except (ValueError, RuntimeError, ArithmeticError) as error:
            row = dict(replicate=r, converged=False, status=f'error: {error}', estimates={})
        rows.append(row)
        if callback is not None:
            callback(row)
    ok = [r for r in rows if r['converged']]
    names = list(_estimates(result))
    summary = {}
    for name in names:
        vals = np.array([r['estimates'][name] for r in ok if name in r['estimates']])
        if len(vals):
            summary[name] = dict(estimate=_estimates(result)[name], median=float(np.median(vals)),
                                 interval=[float(np.quantile(vals, .025)), float(np.quantile(vals, .975))],
                                 se=float(np.std(vals, ddof=1)) if len(vals) > 1 else None)
    return dict(replicates=rows, converged=len(ok), requested=n, summary=summary)


# ------------------------------------------------------------------ SIR

def sir(result, *, samples=1000, resamples=200, iterations=3, inflation=1., covariance=None, power=12,
        seed=20260930):
    """Sampling importance resampling with iterative proposal updates (Dosne et al. 2016, 2017).

    Iteration 1 draws `samples` vectors from a multivariate normal proposal centred at the
    estimate with the local covariance (times `inflation`); every later iteration uses the
    covariance of the vectors resampled in the previous one. Importance weights are
    exp(-dOFV/2)/q, with marginal OFVs from one fixed importance bank at the estimate.
    Returns percentile intervals of the final resample and per-iteration diagnostics.
    """
    from ._diagnostics import general_view
    from ._general.importance import Bank
    from ._general.uncertainty import reporting_quantities, estimate_uncertainty
    problem, x = general_view(result)
    if covariance is None:
        report = estimate_uncertainty(problem, x, float(result.ofv), power=power)
        if report.get('covariance') is None:
            raise ValueError('a positive-definite local covariance is required for the SIR proposal')
        covariance = np.array(report['covariance'])
    cov = np.asarray(covariance, dtype=float) * inflation
    rng = np.random.default_rng(seed)
    bank = Bank(problem, x, power=power, seed=seed + 1)
    ref, _ = bank.evaluate(x, gradient=False)
    history = []
    for it in range(iterations):
        chol = np.linalg.cholesky(cov)
        z = rng.standard_normal((samples, len(x)))
        draws = x + z @ chol.T
        ofv = np.full(samples, np.inf)
        ess = np.zeros(samples)
        for s_ in range(samples):
            if not all((lo is None or draws[s_, j] >= lo) and (hi is None or draws[s_, j] <= hi)
                       for j, (lo, hi) in enumerate(problem.bounds)):
                continue
            v, _ = bank.evaluate(draws[s_], gradient=False)
            ofv[s_] = v
            ess[s_] = bank.last['min_ess'] if bank.last else 0.
        log_w = -.5 * (ofv - ref) + .5 * np.sum(z * z, axis=1)
        log_w[~np.isfinite(log_w)] = -np.inf
        w = np.exp(log_w - np.max(log_w))
        w /= w.sum()
        m = min(resamples, int(np.sum(w > 0)))
        picks = rng.choice(samples, size=m, replace=False, p=w)
        history.append(dict(iteration=it + 1, samples=samples, resamples=m, effective_samples=float(1. / np.sum(w ** 2)),
                            max_weight=float(w.max()), mean_resampled_delta_ofv=float(np.mean(ofv[picks] - ref)),
                            minimum_bank_ess=float(np.min(ess[np.isfinite(ofv)]))))
        chosen = draws[picks]
        if it + 1 < iterations:
            nxt = np.cov(chosen, rowvar=False)
            try:
                np.linalg.cholesky(nxt)
                cov = nxt
            except np.linalg.LinAlgError:
                pass
    names = list(reporting_quantities(problem, x))
    values = np.array([[reporting_quantities(problem, c)[n] for n in names] for c in chosen])
    est = reporting_quantities(problem, x)
    summary = {n: dict(estimate=float(est[n]), interval=[float(np.quantile(values[:, k], .025)),
                                                         float(np.quantile(values[:, k], .975))])
               for k, n in enumerate(names)}
    last = history[-1]
    return dict(summary=summary, iterations=history, samples=samples, resamples=last['resamples'],
                effective_samples=last['effective_samples'], max_weight=last['max_weight'],
                resampled_delta_ofv=(ofv[picks] - ref).tolist(), chi2_expected_mean=len(x),
                minimum_bank_ess=last['minimum_bank_ess'])


def robust_covariance(result, **options):
    """Sandwich covariance and intervals at the estimate (general-engine evaluation for either engine)."""
    from ._diagnostics import general_view
    from ._general.uncertainty import estimate_uncertainty
    problem, x = general_view(result)
    report = estimate_uncertainty(problem, x, float(result.ofv), **options)
    return dict(status=report['status'], coordinates=problem.labels, covariance=report.get('covariance'),
                robust_covariance=report.get('robust_covariance'), intervals=report.get('intervals'),
                robust_intervals=report.get('robust_intervals'))


# ------------------------------------------------------------------ model comparison

def likelihood_ratio_test(reduced, full, df=None):
    df = df if df is not None else full.n_param - reduced.n_param
    delta = float(reduced.ofv - full.ofv)
    return dict(delta_ofv=delta, df=int(df), p_value=float(chi2.sf(delta, df)) if df > 0 else None)


def compare(results, names=None):
    names = names or [f'model {k + 1}' for k in range(len(results))]
    return [dict(model=n, ofv=float(r.ofv), n_param=int(r.n_param), aic=float(r.aic), bic=float(r.bic),
                 converged=bool(r.converged)) for n, r in zip(names, results)]


def stepwise_covariates(data, specification, candidates, *, forward_alpha=.05, backward_alpha=.01,
                        fit_options=None, callback=None):
    """Stepwise covariate modeling by likelihood-ratio tests (df = 1 per coefficient)."""
    from . import fit
    options = dict(fit_options or {})

    def with_covs(spec, covs):
        return dataclasses.replace(spec, covariates=tuple(covs))
    current = list(specification.covariates)
    base = fit(data, specification, **options)
    history = [dict(step='base', covariates=[_label(c) for c in current], ofv=float(base.ofv))]
    remaining = [c for c in candidates if _label(c) not in {_label(k) for k in current}]
    while remaining:
        trials = []
        for c in remaining:
            try:
                res = fit(data, with_covs(specification_of_start(base, specification), current + [c]), **options)
            except (ValueError, RuntimeError, ArithmeticError, np.linalg.LinAlgError) as error:
                history.append(dict(step='forward', candidate=_label(c), error=f'{type(error).__name__}: {error}'))
                continue
            trials.append((base.ofv - res.ofv, c, res))
            if callback:
                callback(dict(step='forward', candidate=_label(c), delta_ofv=float(base.ofv - res.ofv)))
        if not trials:
            break
        delta, best, res = max(trials, key=lambda t: t[0])
        if chi2.sf(delta, 1) >= forward_alpha:
            break
        current.append(best)
        remaining.remove(best)
        base = res
        history.append(dict(step='forward', added=_label(best), delta_ofv=float(delta), ofv=float(res.ofv)))
    changed = True
    while changed and current:
        changed = False
        trials = []
        for c in current:
            reduced = [k for k in current if k is not c]
            try:
                res = fit(data, with_covs(specification_of_start(base, specification), reduced), **options)
            except (ValueError, RuntimeError, ArithmeticError, np.linalg.LinAlgError) as error:
                history.append(dict(step='backward', candidate=_label(c), error=f'{type(error).__name__}: {error}'))
                continue
            trials.append((res.ofv - base.ofv, c, res))
            if callback:
                callback(dict(step='backward', candidate=_label(c), delta_ofv=float(res.ofv - base.ofv)))
        if not trials:
            break
        delta, weakest, res = min(trials, key=lambda t: t[0])
        if chi2.sf(delta, 1) >= backward_alpha:
            current.remove(weakest)
            base = res
            changed = True
            history.append(dict(step='backward', removed=_label(weakest), delta_ofv=float(delta), ofv=float(res.ofv)))
    return dict(covariates=[_label(c) for c in current], final=base, history=history,
                specification=with_covs(specification, current))


def specification_of_start(result, specification):
    """Specification whose starting values are the estimates of `result` where parameters match."""
    try:
        fitted = specification_of(result)
    except Exception:
        return specification
    if _is_general(result):
        return dataclasses.replace(specification, theta=fitted.theta, omega=fitted.omega,
                                   residual=fitted.residual, iov=fitted.iov)
    return dataclasses.replace(specification, theta=fitted.theta, omega=fitted.omega,
                               sigma_prop=fitted.sigma_prop, sigma_add=fitted.sigma_add)


def _label(c):
    return f'{c.parameter}~{c.covariate}' + (f'={c.level:g}' if c.level is not None else '') + f'[{c.form}]'
