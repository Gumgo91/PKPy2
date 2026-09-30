"""Estimation coordinates and per-subject designs of a general Model.

Coordinates (all unconstrained unless bounded):
  theta:NAME        transformed typical value (log, logit or identity scale)
  beta:k            coefficient of covariate effect k
  omega:NAME        log SD of an uncorrelated random effect, or log of the Cholesky
                    diagonal inside an omega block; omega:A,B is a Cholesky off-diagonal
  iov:NAME          log SD of the interoccasion effect
  sigma:OUT:kind    log residual SD (kind: proportional, additive, lognormal)

A coordinate is a 'density' coordinate when, for fixed physical individual
parameters, it changes only the random-effect or residual densities (typical
values and time-invariant covariate effects of parameters with IIV, variances,
residual SDs). The others ('prediction' coordinates) change predictions.
"""
from dataclasses import dataclass
import hashlib
import json
import math
import numpy as np
from ..model import Model
from .packing import pack_individual, kernel_arguments

TRANSFORM = {'log': 0, 'logit': 1, 'identity': 2}
FORM = {'combined': 0, 'lognormal': 1}


def to_transformed(value, kind):
    if kind == 'log':
        if value <= 0:
            raise ValueError('log-transformed parameters must be positive')
        return math.log(value)
    if kind == 'logit':
        if not 0 < value < 1:
            raise ValueError('logit-transformed parameters must lie in (0, 1)')
        return math.log(value / (1 - value))
    return float(value)


def from_transformed(psi, kind):
    if kind == 'log':
        return math.exp(psi)
    if kind == 'logit':
        return 1. / (1. + math.exp(-psi))
    return float(psi)


def _bound(value, kind):
    if value is None:
        return None
    if kind == 'log':
        return math.log(value) if value > 0 else None
    if kind == 'logit':
        if value <= 0:
            return None
        if value >= 1:
            return None
        return math.log(value / (1 - value))
    return float(value)


@dataclass
class Population:
    theta_t: np.ndarray          # (P,) transformed typical values
    beta: np.ndarray             # (C,) covariate coefficients
    chol: np.ndarray             # (q, q) lower Cholesky factor of Omega
    omega: np.ndarray            # (q, q)
    iov_sd: np.ndarray           # (Q,)
    sigma_prop: np.ndarray       # (n_out,)
    sigma_add: np.ndarray        # (n_out,)
    sigma_ln: np.ndarray         # (n_out,)
    form: np.ndarray             # (n_out,) 0 combined, 1 lognormal, -1 not observed


INTEGRATION_DEFAULTS = dict(proposal='gaussian', mode_search=False)


class GeneralProblem:
    def __init__(self, dataset, model, integration=None):
        if not isinstance(model, Model):
            raise ValueError('a pkpy2.Model is required')
        self.model = model
        self.dataset = dataset
        # Importance proposals and conditional-mode starts used by every bank built on this problem
        # (estimation, audit, uncertainty, diagnostics): see importance.Bank and laplace.LaplaceObjective.
        self.integration = dict(INTEGRATION_DEFAULTS)
        if integration:
            unknown = set(integration) - set(INTEGRATION_DEFAULTS)
            if unknown:
                raise ValueError(f'unknown integration options {sorted(unknown)}')
            if integration.get('proposal', 'gaussian') not in ('gaussian', 'mixture'):
                raise ValueError("integration proposal must be 'gaussian' or 'mixture'")
            self.integration.update(integration)
        s = model.structure
        self.structure = s
        self.names = list(s.parameters)
        P = len(self.names)
        self.P = P
        self.transform_names = [model.transform(n) for n in self.names]
        self.transform = np.array([TRANSFORM[t] for t in self.transform_names], dtype=np.int64)
        self.iiv = [j for j, n in enumerate(self.names) if n in model.omega]
        self.iiv_names = [self.names[j] for j in self.iiv]
        self.q = len(self.iiv)
        self.iov = [j for j, n in enumerate(self.names) if n in model.iov]
        self.iov_names = [self.names[j] for j in self.iov]
        self.Q = len(self.iov)
        self.iov_param = np.array(self.iov, dtype=np.int64)
        if self.Q and dataset.occasion_column is None:
            raise ValueError('interoccasion variability needs an occasion column in the data')
        # covariate effects
        tv = dataset.time_varying
        self.effects = []
        for k, c in enumerate(model.covariates):
            if c.covariate not in dataset.covariate_names:
                raise ValueError(f'covariate {c.covariate} was not read from the data')
            if c.form == 'power' and c.center <= 0:
                raise ValueError('power covariate centers must be positive')
            self.effects.append(dict(index=k, parameter=self.names.index(c.parameter), covariate=c.covariate,
                                     form=c.form, center=float(c.center), level=c.level,
                                     time_varying=bool(tv[c.covariate]), coefficient=c.coefficient,
                                     label=f'{c.parameter}~{c.covariate}' + (f'={c.level:g}' if c.level is not None else '')))
        self.C = len(self.effects)
        # outputs and residuals
        outs = s.output_names
        self.n_out = len(outs)
        observed = sorted({int(d) for ind in dataset for d in ind.dvid[(ind.evid == 0) & (ind.mdv == 0)]})
        for d in observed:
            if d > self.n_out:
                raise ValueError(f'DVID {d} exceeds the outputs {outs}')
            if outs[d - 1] not in model.residual:
                raise ValueError(f'output {outs[d - 1]} is observed but has no residual model')
        self.observed_outputs = [outs[d - 1] for d in observed]
        # coordinates
        labels, x0, bounds, roles, self._decode_plan = [], [], [], [], []
        self.theta_const = np.zeros(P)
        self.theta_coord = np.full(P, -1, dtype=np.int64)
        for j, n in enumerate(self.names):
            par = model.theta[n]
            kind = self.transform_names[j]
            value = to_transformed(par.value, kind) if not (par.fixed and kind == 'log' and par.value == 0) else -np.inf
            if par.fixed:
                self.theta_const[j] = value
            else:
                self.theta_coord[j] = len(x0)
                labels.append(f'theta:{n}'); x0.append(value)
                bounds.append((_bound(par.lower, kind), _bound(par.upper, kind)))
                roles.append('density' if j in self.iiv else 'prediction')
        self.beta_const = np.zeros(self.C)
        self.beta_coord = np.full(self.C, -1, dtype=np.int64)
        for k, e in enumerate(self.effects):
            par = e['coefficient']
            if par.fixed:
                self.beta_const[k] = par.value
            else:
                self.beta_coord[k] = len(x0)
                labels.append(f"beta:{e['label']}"); x0.append(float(par.value))
                bounds.append((par.lower, par.upper))
                roles.append('prediction' if e['time_varying'] or e['parameter'] not in self.iiv else 'density')
        # omega: groups of IIV positions (blocks and singletons), Cholesky parameterization
        position = {n: j for j, n in enumerate(self.iiv_names)}
        groups = [[position[n] for n in blk] for blk in model.omega_blocks]
        in_block = {j for g in groups for j in g}
        groups += [[j] for j in range(self.q) if j not in in_block]
        self.omega_groups = groups
        self.chol_plan = []    # (row, col, kind 'log'|'raw', coordinate or -1, constant)
        floor = .5 * math.log(model.omega_floor)
        for g in groups:
            names_g = [self.iiv_names[j] for j in g]
            cov = np.zeros((len(g), len(g)))
            for a, na in enumerate(names_g):
                cov[a, a] = model.omega[na].value
                for b, nb in enumerate(names_g[:a]):
                    value = model.omega_covariance.get((na, nb)) or model.omega_covariance.get((nb, na))
                    cov[a, b] = cov[b, a] = value.value if value is not None else 0.
            try:
                chol = np.linalg.cholesky(cov)
            except np.linalg.LinAlgError:
                raise ValueError(f'initial omega block {names_g} is not positive definite')
            fixed = model.omega[names_g[0]].fixed
            for a in range(len(g)):
                for b in range(a + 1):
                    row, col = g[a], g[b]
                    kind = 'log' if a == b else 'raw'
                    value = math.log(chol[a, b]) if a == b else float(chol[a, b])
                    if fixed:
                        self.chol_plan.append((row, col, kind, -1, value))
                        continue
                    self.chol_plan.append((row, col, kind, len(x0), value))
                    if a == b:
                        labels.append(f'omega:{names_g[a]}')
                        par = model.omega[names_g[a]]
                        lo = max(floor, .5 * math.log(par.lower)) if par.lower else floor
                        hi = .5 * math.log(par.upper) if par.upper else None
                        bounds.append((lo if len(g) == 1 else floor, hi if len(g) == 1 else None))
                    else:
                        labels.append(f'omega:{names_g[a]},{names_g[b]}')
                        bounds.append((None, None))
                    x0.append(value)
                    roles.append('density')
        self.iov_const = np.zeros(self.Q)
        self.iov_coord = np.full(self.Q, -1, dtype=np.int64)
        for j, n in enumerate(self.iov_names):
            par = model.iov[n]
            value = .5 * math.log(par.value)
            if par.fixed:
                self.iov_const[j] = value
            else:
                self.iov_coord[j] = len(x0)
                labels.append(f'iov:{n}'); x0.append(value)
                lo = max(floor, .5 * math.log(par.lower)) if par.lower else floor
                bounds.append((lo, .5 * math.log(par.upper) if par.upper else None))
                roles.append('density')
        self.form = np.full(self.n_out, -1, dtype=np.int64)
        self.sigma_const = np.zeros((self.n_out, 3))       # proportional, additive, lognormal (natural SD)
        self.sigma_coord = np.full((self.n_out, 3), -1, dtype=np.int64)
        for o, name in enumerate(outs):
            if name not in model.residual:
                continue
            r = model.residual[name]
            self.form[o] = FORM[r.form]
            for comp, par in r.components():
                col = {'proportional': 0, 'additive': 1, 'lognormal': 2}[comp]
                if par.fixed or par.value == 0:
                    if not par.fixed:
                        raise ValueError('estimated residual SDs must start above zero')
                    self.sigma_const[o, col] = par.value
                else:
                    self.sigma_coord[o, col] = len(x0)
                    labels.append(f'sigma:{name}:{comp}'); x0.append(math.log(par.value))
                    bounds.append((_bound(par.lower, 'log'), _bound(par.upper, 'log')))
                    roles.append('density')
            if r.form == 'combined' and max(p.value for _, p in r.components()) <= 0:
                raise ValueError(f'residual of {name} needs a positive component')
        if not x0:
            raise ValueError('at least one parameter must be estimated')
        self.labels = labels
        self.x0 = np.array(x0, dtype=float)
        self.bounds = bounds
        self.roles = roles
        self.density = np.array([r == 'density' for r in roles])
        self.prediction_coords = [j for j, r in enumerate(roles) if r == 'prediction']
        # subjects
        self.subjects = [pack_individual(ind, s) for ind in dataset]
        self.N = len(self.subjects)
        self.kernel = kernel_arguments(s)
        self.z_tic = np.zeros((self.N, self.C))
        self.z_tv = []
        for i, ind in enumerate(dataset):
            rows = len(ind.time)
            ztv = np.zeros((rows, self.C))
            for k, e in enumerate(self.effects):
                values = ind.covariates[e['covariate']]
                z = self._design(e, values)
                if e['time_varying']:
                    ztv[:, k] = z
                else:
                    self.z_tic[i, k] = z[0]
            self.z_tv.append(ztv)
        self.tv_effects = [k for k, e in enumerate(self.effects) if e['time_varying']]
        self.tic_effects = [k for k, e in enumerate(self.effects) if not e['time_varying']]
        # A time-varying effect is split into its value at the subject's mean covariate
        # (part of the typical value, so it moves with the random-effect density) and the
        # within-subject deviation (part of the record-level predictions). The model is
        # unchanged; the importance weights react less to the coefficient.
        self.z_base = np.zeros((self.N, self.C))
        for i in range(self.N):
            for k in self.tv_effects:
                self.z_base[i, k] = float(np.mean(self.z_tv[i][:, k]))
        self.linear_effects = [k for k, e in enumerate(self.effects) if e['form'] == 'linear']
        self.n_obs = sum(int(np.sum(sa.obs_used)) for sa in self.subjects)
        self.data_sha256 = self._fingerprint()

    @staticmethod
    def _design(effect, values):
        values = np.asarray(values, dtype=float)
        form = effect['form']
        if form == 'power':
            if np.any(values <= 0):
                raise ValueError(f"power covariate {effect['covariate']} must be positive")
            return np.log(values / effect['center'])
        if form == 'categorical':
            return (values == effect['level']).astype(float)
        return values - effect['center']

    def _fingerprint(self):
        payload = dict(structure=self.structure.name, parameters=self.names, subjects=[
            dict(id=str(s.id), time=s.rec_time.tolist(), evid=s.rec_evid.tolist(), amt=s.rec_amt.tolist(),
                 cmt=s.rec_cmt.tolist(), rate=s.rec_rate.tolist(), ii=s.rec_ii.tolist(), ss=s.rec_ss.tolist(),
                 dv=s.obs_dv.tolist(), used=s.obs_used.tolist(), out=s.obs_out.tolist(), cens=s.obs_cens.tolist(),
                 occ=s.rec_occ.tolist()) for s in self.subjects], z=self.z_tic.tolist(),
            ztv=[z.tolist() for z in self.z_tv])
        return hashlib.sha256(json.dumps(payload, sort_keys=True, default=str).encode()).hexdigest()

    # ------------------------------------------------------------ decoding
    def population(self, x):
        x = np.asarray(x, dtype=float)
        theta_t = self.theta_const.copy()
        free = self.theta_coord >= 0
        theta_t[free] = x[self.theta_coord[free]]
        beta = self.beta_const.copy()
        free = self.beta_coord >= 0
        beta[free] = x[self.beta_coord[free]]
        chol = np.zeros((self.q, self.q))
        for row, col, kind, coord, const in self.chol_plan:
            value = x[coord] if coord >= 0 else const
            chol[row, col] = np.exp(value) if kind == 'log' else value      # inf beyond the float range
        iov = self.iov_const.copy()
        free = self.iov_coord >= 0
        iov[free] = x[self.iov_coord[free]]
        sig = self.sigma_const.copy()
        for o in range(self.n_out):
            for c in range(3):
                if self.sigma_coord[o, c] >= 0:
                    sig[o, c] = np.exp(x[self.sigma_coord[o, c]])
        return Population(theta_t, beta, chol, chol @ chol.T, np.exp(iov), sig[:, 0].copy(), sig[:, 1].copy(),
                          sig[:, 2].copy(), self.form)

    def effect_value(self, k, z, beta):
        """Additive contribution on the transformed scale; nan when a linear effect is invalid."""
        if self.effects[k]['form'] == 'linear':
            arg = 1. + beta * z
            return np.where(arg > 0, np.log(np.where(arg > 0, arg, 1.)), np.nan)
        return beta * z

    def effect_derivative(self, k, z, beta):
        if self.effects[k]['form'] == 'linear':
            return z / (1. + beta * z)
        return z

    def subject_mu(self, i, pop):
        mu = pop.theta_t.copy()
        for k in self.tic_effects:
            mu[self.effects[k]['parameter']] += self.effect_value(k, self.z_tic[i, k], pop.beta[k])
        for k in self.tv_effects:
            mu[self.effects[k]['parameter']] += self.effect_value(k, self.z_base[i, k], pop.beta[k])
        return mu

    def subject_tv(self, i, pop):
        rows = self.subjects[i].n_records
        tv = np.zeros((rows, self.P))
        for k in self.tv_effects:
            tv[:, self.effects[k]['parameter']] += (self.effect_value(k, self.z_tv[i][:, k], pop.beta[k])
                                                    - self.effect_value(k, self.z_base[i, k], pop.beta[k]))
        return tv

    def valid(self, pop):
        """False when a linear covariate effect leaves its domain (fixed zeros are allowed: log 0 = -inf)."""
        if not self.linear_effects:
            return True
        mus = [self.subject_mu(i, pop) for i in range(self.N)]
        if any(np.any(np.isnan(m)) for m in mus):
            return False
        return all(not np.any(np.isnan(self.subject_tv(i, pop))) for i in range(self.N))

    def natural_theta(self, pop):
        return {n: from_transformed(pop.theta_t[j], self.transform_names[j]) for j, n in enumerate(self.names)}

    def describe(self, x):
        pop = self.population(x)
        theta = self.natural_theta(pop)
        omega = {n: float(pop.omega[a, a]) for a, n in enumerate(self.iiv_names)}
        covariance = {}
        for g in self.omega_groups:
            for a in g:
                for b in g:
                    if b < a:
                        covariance[f'{self.iiv_names[a]},{self.iiv_names[b]}'] = float(pop.omega[a, b])
        sigma = {}
        for o, name in enumerate(self.structure.output_names):
            if self.form[o] < 0:
                continue
            if self.form[o] == 1:
                sigma[f'{name}:lognormal'] = float(pop.sigma_ln[o])
            else:
                sigma[f'{name}:proportional'] = float(pop.sigma_prop[o])
                sigma[f'{name}:additive'] = float(pop.sigma_add[o])
        return dict(theta=theta, omega=omega, omega_covariance=covariance,
                    iov={n: float(pop.iov_sd[j] ** 2) for j, n in enumerate(self.iov_names)},
                    coefficients={e['label']: float(pop.beta[k]) for k, e in enumerate(self.effects)},
                    sigma=sigma)
