"""Simulation from a fitted (or declared) model and visual predictive checks.

simulate() draws random effects from their population distribution and residual
errors from the residual model, on the design (dosing, sampling times,
covariates) of the analysis data. vpc() compares percentiles of the observed
data with the distribution of the same percentiles over simulated replicates,
per time bin and output (Holford 2005); prediction_corrected=True applies the
prediction correction of Bergstrand et al. (2011).
"""
import numpy as np
from ._general.laplace import SubjectContext
from ._diagnostics import general_view


def simulate(result, n=100, *, seed=20260930, x=None):
    """Simulated DV (n, observations) in data order, with ID, TIME, OUTPUT and PRED columns."""
    problem, x_hat = general_view(result)
    x = x_hat if x is None else np.asarray(x, dtype=float)
    pop = problem.population(x)
    rng = np.random.default_rng(seed)
    blocks, ids, times, outputs, preds, dvs, cens = [], [], [], [], [], [], []
    for i in range(problem.N):
        ctx = SubjectContext(problem, i, pop)
        obs = ctx.obs
        if obs.m == 0:
            continue
        d = ctx.d
        if d:
            cov = np.zeros((d, d))
            cov[:ctx.q, :ctx.q] = pop.omega
            if d > ctx.q:
                cov[ctx.q:, ctx.q:] = np.diag(ctx.iov_var)
            b = ctx.m0 + rng.standard_normal((n, d)) @ np.linalg.cholesky(cov).T
        else:
            b = np.zeros((n, 0))
        pred = ctx.predict(np.vstack([ctx.m0[None, :], b]))
        f = pred[1:, obs.index]
        mean, v = obs.moments(f)
        y = mean + np.sqrt(v) * rng.standard_normal(mean.shape)
        y = np.where(obs.lognormal, np.exp(y), y)
        blocks.append(y)
        s = problem.subjects[i]
        ids += [s.id] * obs.m
        times += s.obs_time[obs.index].tolist()
        outputs += [problem.structure.output_names[o] for o in obs.out]
        preds += pred[0, obs.index].tolist()
        dvs += obs.y.tolist()
        cens += obs.cens.tolist()
    return dict(ID=np.array(ids), TIME=np.array(times), OUTPUT=np.array(outputs), PRED=np.array(preds),
                DV=np.array(dvs), CENS=np.array(cens), SIM=np.hstack(blocks))


def _bins(times, bins):
    """Bin edges and bin index of each time: one bin per nominal time when there are at most
    2 * bins distinct times, otherwise `bins` quantile bins (or the given edges)."""
    unique = np.unique(times)
    if isinstance(bins, int) and len(unique) <= 2 * bins:
        mids = .5 * (unique[1:] + unique[:-1])
        edges = np.concatenate([[unique[0]], mids, [unique[-1]]])
    elif isinstance(bins, int):
        edges = np.unique(np.quantile(times, np.linspace(0, 1, bins + 1)))
    else:
        edges = np.asarray(bins, dtype=float)
    edges[0] = min(edges[0], times.min()) - 1e-9
    edges[-1] = max(edges[-1], times.max()) + 1e-9
    return edges, np.clip(np.searchsorted(edges, times, side='right') - 1, 0, len(edges) - 2)


def vpc(result, *, n=500, bins=8, quantiles=(.05, .5, .95), ci=.95, prediction_corrected=False,
        lloq=None, seed=20260930):
    """Visual predictive check per output.

    Returns {output: dict(edges, bin_time, observed (n_bins, n_q), lower/median/upper
    (n_bins, n_q) of the simulated percentiles, observed_blq and simulated BLQ
    fraction interval when lloq is given)}. lloq: a limit or {output: limit}. With a
    limit, observed censored values and simulated values below it are both set to
    the limit before percentiles are taken (the censored VPC of Bergstrand et al.
    2009), and the observed and simulated BLQ fractions are reported per bin.
    """
    sim = simulate(result, n, seed=seed)
    out = {}
    alpha = (1 - ci) / 2
    for name in np.unique(sim['OUTPUT']):
        k = sim['OUTPUT'] == name
        t = sim['TIME'][k]
        y = sim['DV'][k].copy()
        s = sim['SIM'][:, k].copy()
        pred = sim['PRED'][k]
        cens = sim['CENS'][k]
        limit = lloq.get(str(name)) if isinstance(lloq, dict) else lloq
        blq_sim_values = None
        if limit is not None:
            blq_sim_values = s < limit
            y = np.where((cens == 1) | (y < limit), limit, y)
            s = np.maximum(s, limit)
        edges, which = _bins(t, bins)
        nb = len(edges) - 1
        if prediction_corrected:
            for j in range(nb):
                m = which == j
                if not np.any(m):
                    continue
                ref = np.median(pred[m])
                factor = ref / np.where(pred[m] != 0, pred[m], np.nan)
                y[m] = y[m] * factor
                s[:, m] = s[:, m] * factor
        obs_q = np.full((nb, len(quantiles)), np.nan)
        sim_q = np.full((n, nb, len(quantiles)), np.nan)
        centers = np.full(nb, np.nan)
        blq_obs = np.full(nb, np.nan)
        blq_sim = np.full((n, nb), np.nan)
        for j in range(nb):
            m = which == j
            if not np.any(m):
                continue
            centers[j] = np.median(t[m])
            obs_q[j] = np.quantile(y[m], quantiles)
            sim_q[:, j, :] = np.quantile(s[:, m], quantiles, axis=1).T
            if limit is not None or np.any(cens != 0):
                blq_obs[j] = np.mean((cens[m] == 1) | (sim['DV'][k][m] < limit if limit is not None else False))
                if limit is not None:
                    blq_sim[:, j] = np.mean(blq_sim_values[:, m], axis=1)
        entry = dict(edges=edges, bin_time=centers, quantiles=list(quantiles), observed=obs_q,
                     lower=np.nanquantile(sim_q, alpha, axis=0), median=np.nanquantile(sim_q, .5, axis=0),
                     upper=np.nanquantile(sim_q, 1 - alpha, axis=0), prediction_corrected=prediction_corrected,
                     n_simulations=n, observations=dict(time=t, dv=y))
        if limit is not None:
            entry.update(lloq=limit, observed_blq=blq_obs, simulated_blq=np.nanquantile(blq_sim, [alpha, .5, 1 - alpha], axis=0))
        inside = (obs_q >= entry['lower']) & (obs_q <= entry['upper'])
        entry['observed_within_interval'] = float(np.nanmean(inside))
        out[str(name)] = entry
    return out


def simulate_data(model, design, *, seed=20260930, lloq=None):
    """A copy of `design` (a Dataset) with DV simulated from `model` at its declared values.

    lloq: {output name: limit}; simulated values below the limit are censored
    (CENS = 1, DV = limit), as in data analysed with the M3 method.
    """
    import copy
    from ._general.problem import GeneralProblem
    from ._general.laplace import SubjectContext
    from .data import Dataset
    design = copy.deepcopy(design)
    for ind in design:
        obs = ind.evid == 0
        ind.dv[obs] = 1.
        ind.mdv[obs] = 0
        ind.cens[obs] = 0
    problem = GeneralProblem(design, model)
    pop = problem.population(problem.x0)
    rng = np.random.default_rng(seed)
    for i, ind in enumerate(design):
        ctx = SubjectContext(problem, i, pop)
        d = ctx.d
        if d:
            cov = np.zeros((d, d))
            cov[:ctx.q, :ctx.q] = pop.omega
            if d > ctx.q:
                cov[ctx.q:, ctx.q:] = np.diag(ctx.iov_var)
            b = ctx.m0 + np.linalg.cholesky(cov) @ rng.standard_normal(d)
        else:
            b = np.zeros(0)
        pred = ctx.predict(b[None, :])[0]
        obs = ctx.obs
        f = pred[obs.index][None, :]
        mean, v = obs.moments(f)
        y = (mean + np.sqrt(v) * rng.standard_normal(mean.shape))[0]
        y = np.where(obs.lognormal, np.exp(y), y)
        rows = np.nonzero(ind.evid == 0)[0][obs.index]
        ind.dv[rows] = y
        if lloq:
            names = problem.structure.output_names
            for k, r in enumerate(rows):
                limit = lloq.get(names[obs.out[k]])
                if limit is not None and y[k] < limit:
                    ind.dv[r] = limit
                    ind.cens[r] = 1
    return Dataset(list(design), covariates=design.covariate_names, occasion=design.occasion_column)
