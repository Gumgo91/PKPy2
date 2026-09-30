"""Model diagnostics: individual estimates, predictions, residuals, NPDE and shrinkage.

Individual estimates are the conditional modes (empirical Bayes estimates, EBE)
and the importance-sampled conditional means and SDs at the final estimate.
PRED uses the typical values (random effects zero); IPRED the conditional mode.
WRES uses the first-order linearization at the typical values, CWRES the
FOCE linearization at the conditional mode (Hooker et al. 2007):
E = f(b*) - G (b* - b0), Cov = G Omega_b G' + R(b*), residual = chol(Cov)^-1 (y - E).
IWRES = (y - IPRED) / SD(IPRED). Residuals of lognormal outputs are on the log
scale. Decorrelation conventions: cwres='cholesky' (default; NONMEM/PsN, with the
lower Cholesky factor L of Cov: L^-1 (y - E)) or 'diagonal' (each residual divided
by the square root of its own variance, as nlmixr2 reports CWRES); npde='cholesky'
(default; the npde package, L^-1) or 'cholesky_upper' (U^-1 with Cov = U'U, as
nlmixr2 computes NPDE). NPDE follow Brendel et al. (2006) and Comets et al. (2008): simulated
replicates of each subject's design, Cholesky decorrelation, prediction
discrepancies and the inverse normal transform. Censored observations have no
residuals (reported as NaN). eta-shrinkage is 1 - SD(EBE)/omega; epsilon-shrinkage
is 1 - SD(IWRES).
"""
import math
import numpy as np
from scipy.linalg import solve_triangular
from scipy.special import ndtri
from ._general.laplace import SubjectContext, LaplaceObjective
from ._general.importance import Bank


def general_view(result):
    """(GeneralProblem, x) for a classic FitResult or a GeneralFitResult."""
    if hasattr(result, 'problem') and result.problem.__class__.__name__ == 'GeneralProblem':
        return result.problem, np.asarray(result.x, dtype=float)
    from .api import Parameter
    from .model import Model, Residual
    from .data import Dataset
    from . import structures as S
    from .api import Covariate
    from ._general.problem import GeneralProblem
    cm = result.problem
    spec = cm.spec
    theta = {n: Parameter(float(v), n in spec['fixed_theta']) for n, v in result.theta.items()}
    omega = {n: Parameter(float(v), n in spec['fixed_omega']) for n, v in result.omega.items()}
    sp, sa = result.sigma['sigma_prop'], result.sigma['sigma_add']
    residual = Residual(proportional=Parameter(sp, 'sigma_prop' in spec['fixed_sigma'] or sp == 0),
                        additive=Parameter(sa, 'sigma_add' in spec['fixed_sigma'] or sa == 0))
    covs = tuple(Covariate(e['parameter'], e['covariate'], e['center'], Parameter(float(b), e['fixed']),
                           e.get('form', 'power'), e.get('level'))
                 for e, b in zip(spec['effects'], result.coefficients))
    model = Model(S.from_classic(result.model), theta, omega, covariates=covs, residual=residual,
                  omega_floor=spec['omega_floor'])
    problem = GeneralProblem(Dataset.from_subjects(cm.subjects), model)
    return problem, problem.x0.copy()


def individual_estimates(result, *, power=12, seed=7):
    """Per subject: EBE (mode), conditional mean and SD of the random effects, and individual parameters.

    Individual parameters are evaluated at the mode, at the subject's mean value of any
    time-varying covariate and without interoccasion effects.
    """
    problem, x = general_view(result)
    pop = problem.population(x)
    _, states = LaplaceObjective(problem).states(x, starts=getattr(result, 'modes', None))
    bank = Bank(problem, x, power=power, seed=seed, states=states)
    rows = bank.subject_terms(x)
    out = []
    for i, st in enumerate(states):
        ctx = SubjectContext(problem, i, pop)
        w = rows[i]['weights']
        b = bank.particles[i]
        mean = w @ b if ctx.d else np.zeros(0)
        sd = np.sqrt(np.maximum(w @ (b - mean) ** 2, 0.)) if ctx.d else np.zeros(0)
        eta_mode = st.b[:ctx.q] - ctx.m0[:ctx.q]
        eta_mean = mean[:ctx.q] - ctx.m0[:ctx.q]
        base, _ = ctx.inputs(st.b[None, :])
        individual = {}
        for j, n in enumerate(problem.names):
            psi = base[0, j]
            kind = problem.transform_names[j]
            individual[n] = float(math.exp(psi) if kind == 'log' else 1 / (1 + math.exp(-psi)) if kind == 'logit' else psi)
        out.append(dict(id=problem.subjects[i].id,
                        eta_mode={n: float(v) for n, v in zip(problem.iiv_names, eta_mode)},
                        eta_mean={n: float(v) for n, v in zip(problem.iiv_names, eta_mean)},
                        eta_sd={n: float(v) for n, v in zip(problem.iiv_names, sd[:ctx.q])},
                        kappa_mode=st.b[ctx.q:].reshape(ctx.O, problem.Q).tolist() if problem.Q else [],
                        parameters=individual, ess=float(rows[i]['ess'])))
    return out


def _factor(cov):
    try:
        return np.linalg.cholesky(cov)
    except np.linalg.LinAlgError:
        vals, vecs = np.linalg.eigh(cov)
        return np.linalg.cholesky(vecs @ np.diag(np.maximum(vals, 1e-12)) @ vecs.T)


def _decorrelate(r, cov, method):
    """r (m,) or (n, m) residual vectors decorrelated by the chosen convention."""
    if method == 'diagonal':
        return r / np.sqrt(np.diag(cov))
    L = _factor(cov)
    if method == 'cholesky':
        return solve_triangular(L, r.T, lower=True).T
    if method == 'cholesky_upper':
        return solve_triangular(L.T, r.T, lower=False).T
    raise ValueError('decorrelation must be cholesky, cholesky_upper or diagonal')


def _chol_residual(y, e, cov, method='cholesky'):
    return _decorrelate(y - e, cov, method)


def _residual_scale(obs, f):
    """Target, mean and variance on the modeling scale (log for lognormal outputs)."""
    mean, v = obs.moments(f[None, :])
    target = np.where(obs.lognormal, obs.logy, obs.y)
    return target, mean[0], v[0]


def _simulate_subject(ctx, obs_all, n, rng):
    """Simulated observations (n, m_used) on the modeling scale for one subject (uncensored values)."""
    d = ctx.d
    pop = ctx.pop
    if d:
        cov = np.zeros((d, d))
        cov[:ctx.q, :ctx.q] = pop.omega
        if d > ctx.q:
            cov[ctx.q:, ctx.q:] = np.diag(ctx.iov_var)
        b = ctx.m0 + rng.standard_normal((n, d)) @ np.linalg.cholesky(cov).T
    else:
        b = np.zeros((n, 0))
    pred = ctx.predict(b)
    f = pred[:, obs_all.index]
    mean, v = obs_all.moments(f)
    return mean + np.sqrt(v) * rng.standard_normal(mean.shape), pred


def diagnostics(result, *, npde_samples=1000, seed=20260930, cwres='cholesky', npde='cholesky', simulations=None):
    """Observation-level table (dict of arrays) and summary (shrinkage).

    simulations: optionally the output of simulate() for the same result, whose
    replicates are then used for NPDE (natural scale; log scale for lognormal outputs).
    """
    problem, x = general_view(result)
    pop = problem.population(x)
    lap = LaplaceObjective(problem)
    _, states = lap.states(x, starts=getattr(result, 'modes', None))        # a Laplace fit's own modes
    rng = np.random.default_rng(seed)
    cols = {k: [] for k in ('ID', 'TIME', 'OUTPUT', 'DV', 'CENS', 'PRED', 'IPRED', 'RES', 'IRES', 'WRES', 'CWRES',
                            'IWRES', 'NPDE', 'PD', 'EPRED')}
    eta_modes = []
    for i, st in enumerate(states):
        ctx = SubjectContext(problem, i, pop)
        obs = ctx.obs
        s = problem.subjects[i]
        m = obs.m
        if m == 0:
            continue
        idx = obs.index
        zero = ctx.m0[None, :]
        pts = [zero]
        d = ctx.d
        h = 1e-4 * np.maximum(1., np.abs(ctx.m0)) if d else np.zeros(0)
        if d:
            pts += [ctx.m0 + np.diag(h), ctx.m0 - np.diag(h)]
        pred0 = ctx.predict(np.vstack(pts))
        f0 = pred0[0, idx]
        G0 = (pred0[1:d + 1, idx] - pred0[d + 1:, idx]).T / (2 * h) if d else np.zeros((m, 0))
        fstar = st.pred[idx]
        Gs = st.jac
        prior = np.zeros((d, d))
        if d:
            prior[:ctx.q, :ctx.q] = pop.omega
            if d > ctx.q:
                prior[ctx.q:, ctx.q:] = np.diag(ctx.iov_var)
        y, mean0, v0 = _residual_scale(obs, f0)
        _, means, vs = _residual_scale(obs, fstar)
        # chain rule for the log scale: d log f = df / f
        J0 = G0 / np.where(obs.lognormal, f0, 1.)[:, None]
        Js = Gs / np.where(obs.lognormal, fstar, 1.)[:, None]
        unc = obs.uncensored
        wres = np.full(m, np.nan)
        cwres_ = np.full(m, np.nan)
        if np.any(unc):
            k = np.nonzero(unc)[0]
            cov0 = J0[k] @ prior @ J0[k].T + np.diag(v0[k])
            wres[k] = _chol_residual(y[k], mean0[k], cov0, cwres)
            e = means[k] - Js[k] @ (st.b - ctx.m0)
            covs = Js[k] @ prior @ Js[k].T + np.diag(vs[k])
            cwres_[k] = _chol_residual(y[k], e, covs, cwres)
        iwres = np.where(unc, (y - means) / np.sqrt(vs), np.nan)
        # NPDE by simulation
        if simulations is not None:
            sel = simulations['ID'] == s.id
            sims = simulations['SIM'][:, sel]
            sims = np.where(obs.lognormal, np.log(np.where(sims > 0, sims, np.nan)), sims)
        else:
            sims, _ = _simulate_subject(ctx, obs, npde_samples, rng)
        nsim = sims.shape[0]
        npde_ = np.full(m, np.nan)
        pd_ = np.full(m, np.nan)
        epred = np.nanmean(sims, axis=0)
        if np.any(unc):
            k = np.nonzero(unc)[0]
            sk = sims[:, k]
            ok = np.all(np.isfinite(sk), axis=1)
            sk = sk[ok]
            e = sk.mean(axis=0)
            pd_[k] = (np.sum(sk < y[k], axis=0) + .5 * np.sum(sk == y[k], axis=0)) / len(sk)
            cov = np.cov(sk, rowvar=False).reshape(len(k), len(k))
            try:
                ys = _decorrelate(y[k] - e, cov, npde)
                ss = _decorrelate(sk - e, cov, npde)
                pde = np.mean(ss < ys, axis=0)
                pde = np.clip(pde, 1. / (2 * len(sk)), 1. - 1. / (2 * len(sk)))
                npde_[k] = ndtri(pde)
            except np.linalg.LinAlgError:
                pass
        pdc = np.clip(pd_, 1. / (2 * nsim), 1. - 1. / (2 * nsim))
        back = lambda v: np.where(obs.lognormal, np.exp(np.where(obs.lognormal, v, 0.)), v)
        cols['ID'] += [s.id] * m
        cols['TIME'] += s.obs_time[idx].tolist()
        cols['OUTPUT'] += [problem.structure.output_names[o] for o in obs.out]
        cols['DV'] += obs.y.tolist()
        cols['CENS'] += obs.cens.tolist()
        cols['PRED'] += f0.tolist()
        cols['IPRED'] += fstar.tolist()
        cols['RES'] += (obs.y - f0).tolist()
        cols['IRES'] += (obs.y - fstar).tolist()
        cols['WRES'] += wres.tolist()
        cols['CWRES'] += cwres_.tolist()
        cols['IWRES'] += iwres.tolist()
        cols['NPDE'] += npde_.tolist()
        cols['PD'] += ndtri(pdc).tolist()
        cols['EPRED'] += back(epred).tolist()
        eta_modes.append(st.b[:ctx.q] - ctx.m0[:ctx.q])
    table = {k: np.array(v) for k, v in cols.items()}
    etas = np.array(eta_modes) if eta_modes else np.zeros((0, problem.q))
    omega_sd = np.sqrt(np.diag(pop.omega))
    eta_shrinkage = {n: float(1. - np.std(etas[:, a], ddof=1) / omega_sd[a]) if len(etas) > 1 else None
                     for a, n in enumerate(problem.iiv_names)}
    iw = table['IWRES'][np.isfinite(table['IWRES'])]
    summary = dict(eta_shrinkage=eta_shrinkage,
                   epsilon_shrinkage=float(1. - np.std(iw, ddof=1)) if len(iw) > 1 else None,
                   npde_mean=float(np.nanmean(table['NPDE'])), npde_variance=float(np.nanvar(table['NPDE'], ddof=1)),
                   cwres_mean=float(np.nanmean(table['CWRES'])), cwres_variance=float(np.nanvar(table['CWRES'], ddof=1)))
    return table, summary
