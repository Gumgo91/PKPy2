"""Independent check of PKPy2 general-engine marginal likelihoods.

For each test model a small dataset is simulated with PKPy2, then the exact
marginal OFV at a fixed parameter point is computed twice:
  * by PKPy2 (two independent importance banks, 2^16 particles per subject);
  * by a separate implementation in this file that shares no code with PKPy2:
    its own CSV parsing, predictions (scipy.linalg.expm / closed forms /
    scipy.integrate.solve_ivp), likelihood terms (including M3 censoring and
    lognormal residuals) and adaptive Gauss-Hermite quadrature over all random
    effects (IIV and IOV) centred at the conditional mode.
Output: output/pkpy2_extended_validation/likelihood_agreement.json
"""
from pathlib import Path
import csv
import hashlib
import itertools
import json
import math
import sys

import numpy as np
from numpy.polynomial.hermite_e import hermegauss
from scipy.integrate import solve_ivp
from scipy.linalg import expm
from scipy.optimize import minimize
from scipy.special import log_ndtr

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'packages/pkpy2/src'))
OUT = ROOT / 'output/pkpy2_extended_validation/likelihood'
LOG2PI = math.log(2 * math.pi)


# ------------------------------------------------------------------ independent reference

def read_rows(path):
    rows = list(csv.DictReader(open(path)))
    subjects = {}
    for r in rows:
        subjects.setdefault(r['ID'], []).append({k: (float(v) if v not in ('.', '') else math.nan) for k, v in r.items()})
    for recs in subjects.values():
        recs.sort(key=lambda r: r['TIME'])
    return list(subjects.values())


def log_mvn(x, cov):
    chol = np.linalg.cholesky(cov)
    z = np.linalg.solve(chol, x)
    return -.5 * (len(x) * LOG2PI + 2 * np.sum(np.log(np.diag(chol))) + z @ z)


def obs_loglik(y, f, sd, cens, lognormal):
    if lognormal:
        if f <= 0:
            return -np.inf
        m, t = math.log(f), math.log(y)
    else:
        m, t = f, y
    z = (t - m) / sd
    if cens == 1:
        return float(log_ndtr(z))
    return -.5 * (LOG2PI + 2 * math.log(sd) + z * z)


def agq_subject(logjoint, d, nodes):
    """log integral of exp(logjoint(b)) over R^d by adaptive Gauss-Hermite quadrature."""
    fit = minimize(lambda b: -logjoint(b), np.zeros(d), method='BFGS', options=dict(gtol=1e-9, maxiter=2000))
    mode = fit.x
    h = 1e-4
    H = np.zeros((d, d))
    f0 = logjoint(mode)
    for i in range(d):
        for j in range(i, d):
            ei, ej = np.eye(d)[i] * h, np.eye(d)[j] * h
            H[i, j] = H[j, i] = -(logjoint(mode + ei + ej) - logjoint(mode + ei - ej) - logjoint(mode - ei + ej)
                                  + logjoint(mode - ei - ej)) / (4 * h * h)
    cov = np.linalg.inv(H)
    L = np.linalg.cholesky(cov)
    z, w = hermegauss(nodes)
    logw = np.log(w / math.sqrt(2 * math.pi))
    terms = []
    for idx in itertools.product(range(nodes), repeat=d):
        zz = z[list(idx)]
        b = mode + L @ zz
        terms.append(logjoint(b) + np.sum(logw[list(idx)]) + .5 * zz @ zz + .5 * d * LOG2PI)
    terms = np.array(terms)
    peak = terms.max()
    return peak + math.log(np.sum(np.exp(terms - peak))) + np.sum(np.log(np.diag(L)))


def linear_predictions(recs, matrix_fn, params_fn, outputs_fn, n_state, dose_fn):
    """Piecewise propagation with scipy expm; parameters of a record apply after it (LOCF)."""
    x = np.zeros(n_state)
    t = recs[0]['TIME']
    p = params_fn(recs[0])
    preds = []
    for r in recs:
        if r['TIME'] > t:
            x = expm(matrix_fn(p) * (r['TIME'] - t)) @ x
            t = r['TIME']
        p = params_fn(r)
        if r['EVID'] == 3:
            x = np.zeros(n_state)
        if r['EVID'] == 1:
            cmt, amount = dose_fn(r, p)
            x[cmt] += amount
        if r['EVID'] == 0:
            preds.append(outputs_fn(x, p, r))
    return preds


# ------------------------------------------------------------------ test models

def model_block_tv(pk):
    """L1: 1-cmt oral, block Omega(CL,V), categorical SEX on CL, time-varying WT (power) on V, combined error."""
    from pkpy2 import Parameter as P, Covariate, Model, Residual, structures as S
    model = Model(S.pk(1, 'first_order'),
                  theta=dict(CL=P(3.), V=P(30.), Ka=P(1.2)), omega=dict(CL=P(.1), V=P(.08)),
                  omega_blocks=[('CL', 'V')], omega_covariance={('CL', 'V'): P(.05)},
                  covariates=(Covariate('CL', 'SEX', 0, P(.3), 'categorical', 1), Covariate('V', 'WT', 70., P(1.))),
                  residual=Residual(proportional=P(.15), additive=P(.1)))
    x = dict(CL=3.2, V=28., Ka=1.1, O11=.12, O22=.07, O21=.045, bSEX=.25, bWT=.9, sp=.14, sa=.12)

    def ofv(subjects):
        total = 0.
        cov = np.array([[x['O11'], x['O21']], [x['O21'], x['O22']]])
        for recs in subjects:
            sex = recs[0]['SEX']

            def logjoint(eta):
                cl = x['CL'] * math.exp(x['bSEX'] * (sex == 1) + eta[0])
                par = lambda r: dict(CL=cl, V=x['V'] * (r['WT'] / 70.) ** x['bWT'] * math.exp(eta[1]), Ka=x['Ka'])
                A = lambda p: np.array([[-p['Ka'], 0], [p['Ka'], -p['CL'] / p['V']]])
                preds = linear_predictions(recs, A, par, lambda s, p, r: s[1] / p['V'], 2, lambda r, p: (0, r['AMT']))
                ll = 0.
                obs = [r for r in recs if r['EVID'] == 0]
                for r, f in zip(obs, preds):
                    ll += obs_loglik(r['DV'], f, math.sqrt((x['sp'] * f) ** 2 + x['sa'] ** 2), r.get('CENS', 0), False)
                return ll + log_mvn(np.asarray(eta), cov)
            total += -2 * agq_subject(logjoint, 2, 21)
        return total
    xvec = lambda prob: _coords(prob, dict(CL=x['CL'], V=x['V'], Ka=x['Ka']), [x['bSEX'], x['bWT']],
                                chol=np.linalg.cholesky(np.array([[x['O11'], x['O21']], [x['O21'], x['O22']]])),
                                sigma=dict(CP=(x['sp'], x['sa'], None)))
    return model, ofv, xvec


def model_iov_blq_lognormal(pk):
    """L2: 1-cmt IV bolus, IIV on CL and V, IOV on CL (2 occasions), lognormal residual, M3 censoring."""
    from pkpy2 import Parameter as P, Model, Residual, structures as S
    model = Model(S.pk(1), theta=dict(CL=P(4.), V=P(40.)), omega=dict(CL=P(.09), V=P(.04)),
                  iov=dict(CL=P(.04)), residual=Residual(lognormal=P(.2)))
    x = dict(CL=4.4, V=37., OCL=.1, OV=.05, K=.03, sl=.22)

    def ofv(subjects):
        total = 0.
        for recs in subjects:
            occ = [int(r['OCC']) for r in recs]
            labels = list(dict.fromkeys(occ))

            def logjoint(b):
                eta_cl, eta_v = b[0], b[1]
                kap = {o: b[2 + k] for k, o in enumerate(labels)}
                par = lambda r: dict(CL=x['CL'] * math.exp(eta_cl + kap[int(r['OCC'])]), V=x['V'] * math.exp(eta_v))
                A = lambda p: np.array([[-p['CL'] / p['V']]])
                preds = linear_predictions(recs, A, par, lambda s, p, r: s[0] / p['V'], 1, lambda r, p: (0, r['AMT']))
                ll = 0.
                obs = [r for r in recs if r['EVID'] == 0]
                for r, f in zip(obs, preds):
                    ll += obs_loglik(r['DV'], f, x['sl'], r['CENS'], True)
                prior = -.5 * (LOG2PI + math.log(x['OCL']) + eta_cl ** 2 / x['OCL'])
                prior += -.5 * (LOG2PI + math.log(x['OV']) + eta_v ** 2 / x['OV'])
                for o in labels:
                    prior += -.5 * (LOG2PI + math.log(x['K']) + kap[o] ** 2 / x['K'])
                return ll + prior
            total += -2 * agq_subject(logjoint, 2 + len(labels), 11)
        return total
    xvec = lambda prob: _coords(prob, dict(CL=x['CL'], V=x['V']), [], chol=np.diag(np.sqrt([x['OCL'], x['OV']])),
                                iov=dict(CL=x['K']), sigma=dict(CP=(None, None, x['sl'])))
    return model, ofv, xvec


def model_bioavailability_linear_cov(pk):
    """L4: IV and oral doses, logit F with IIV, linear AGE effect on CL, IIV on CL, proportional error."""
    from pkpy2 import Parameter as P, Covariate, Model, Residual, structures as S
    model = Model(S.pk(1, 'first_order', bioavailability=True),
                  theta=dict(CL=P(5.), V=P(50.), Ka=P(1.), F=P(.6)), omega=dict(CL=P(.09), F=P(.3)),
                  transforms=dict(F='logit'), covariates=(Covariate('CL', 'AGE', 40., P(-.01), 'linear'),),
                  residual=Residual(proportional=P(.15)))
    x = dict(CL=5.4, V=47., Ka=.9, F=.55, OCL=.08, OF=.25, bAGE=-.008, sp=.16)

    def ofv(subjects):
        total = 0.
        for recs in subjects:
            age = recs[0]['AGE']

            def logjoint(b):
                cl = x['CL'] * (1 + x['bAGE'] * (age - 40)) * math.exp(b[0])
                lf = math.log(x['F'] / (1 - x['F'])) + b[1]
                f = 1 / (1 + math.exp(-lf))
                par = lambda r: dict(CL=cl, V=x['V'], Ka=x['Ka'], F=f)
                A = lambda p: np.array([[-p['Ka'], 0], [p['Ka'], -p['CL'] / p['V']]])
                dose = lambda r, p: (0, r['AMT'] * p['F']) if int(r['CMT']) == 1 else (1, r['AMT'])
                preds = linear_predictions(recs, A, par, lambda s, p, r: s[1] / p['V'], 2, dose)
                ll = 0.
                obs = [r for r in recs if r['EVID'] == 0]
                for r, fp in zip(obs, preds):
                    ll += obs_loglik(r['DV'], fp, x['sp'] * fp, 0, False)
                prior = -.5 * (LOG2PI + math.log(x['OCL']) + b[0] ** 2 / x['OCL'])
                prior += -.5 * (LOG2PI + math.log(x['OF']) + b[1] ** 2 / x['OF'])
                return ll + prior
            total += -2 * agq_subject(logjoint, 2, 21)
        return total
    xvec = lambda prob: _coords(prob, dict(CL=x['CL'], V=x['V'], Ka=x['Ka'], F=x['F']), [x['bAGE']],
                                chol=np.diag(np.sqrt([x['OCL'], x['OF']])), sigma=dict(CP=(x['sp'], 0., None)))
    return model, ofv, xvec


def model_idr_two_outputs(pk):
    """L3: 1-cmt oral PK driving an indirect response (type III); IIV on CL and KOUT; two outputs."""
    from pkpy2 import Parameter as P, Model, Residual, structures as S
    model = Model(S.indirect_response(3), theta=dict(CL=P(2.), V=P(25.), Ka=P(1.3), R0=P(80.), KOUT=P(.15),
                                                     EMAX=P(1.5), EC50=P(1.)),
                  omega=dict(CL=P(.08), KOUT=P(.1)),
                  residual=dict(CP=Residual(proportional=P(.12)), R=Residual(additive=P(4.))))
    x = dict(CL=2.2, V=24., Ka=1.2, R0=78., KOUT=.14, EMAX=1.6, EC50=1.1, OCL=.07, OK=.12, sp=.13, sa=4.5)

    def ofv(subjects):
        total = 0.
        for recs in subjects:
            obs = [r for r in recs if r['EVID'] == 0]
            doses = [r for r in recs if r['EVID'] == 1]

            def logjoint(b):
                cl = x['CL'] * math.exp(b[0])
                kout = x['KOUT'] * math.exp(b[1])

                def rhs(t, s):
                    c = s[1] / x['V']
                    eff = x['EMAX'] * c / (x['EC50'] + c)
                    return [-x['Ka'] * s[0], x['Ka'] * s[0] - cl / x['V'] * s[1],
                            x['R0'] * kout * (1 + eff) - kout * s[2]]
                state = np.array([0., 0., x['R0']])
                t = 0.
                events = sorted([(r['TIME'], 0, r) for r in doses] + [(r['TIME'], 1, r) for r in obs], key=lambda e: (e[0], e[1]))
                ll = 0.
                for te, kind, r in events:
                    if te > t:
                        sol = solve_ivp(rhs, (t, te), state, method='Radau', rtol=1e-10, atol=1e-12)
                        state = sol.y[:, -1]
                        t = te
                    if kind == 0:
                        state[0] += r['AMT']
                    else:
                        if int(r['DVID']) == 1:
                            f = state[1] / x['V']
                            ll += obs_loglik(r['DV'], f, x['sp'] * f, 0, False)
                        else:
                            ll += obs_loglik(r['DV'], state[2], x['sa'], 0, False)
                prior = -.5 * (LOG2PI + math.log(x['OCL']) + b[0] ** 2 / x['OCL'])
                prior += -.5 * (LOG2PI + math.log(x['OK']) + b[1] ** 2 / x['OK'])
                return ll + prior
            total += -2 * agq_subject(logjoint, 2, 13)
        return total
    xvec = lambda prob: _coords(prob, dict(CL=x['CL'], V=x['V'], Ka=x['Ka'], R0=x['R0'], KOUT=x['KOUT'],
                                           EMAX=x['EMAX'], EC50=x['EC50']), [],
                                chol=np.diag(np.sqrt([x['OCL'], x['OK']])),
                                sigma=dict(CP=(x['sp'], 0., None), R=(0., x['sa'], None)))
    return model, ofv, xvec


def _coords(problem, theta, beta, chol=None, iov=None, sigma=None):
    """PKPy2 coordinates for the test point (only used to query PKPy2, never by the reference)."""
    x = np.zeros(len(problem.labels))
    for j, lab in enumerate(problem.labels):
        kind, _, name = lab.partition(':')
        if kind == 'theta':
            v = theta[name]
            t = problem.transform_names[problem.names.index(name)]
            x[j] = math.log(v) if t == 'log' else math.log(v / (1 - v)) if t == 'logit' else v
        elif kind == 'beta':
            k = [e['label'] for e in problem.effects].index(name)
            x[j] = beta[k]
        elif kind == 'omega':
            if ',' in name:
                a, b = name.split(',')
                x[j] = chol[problem.iiv_names.index(a), problem.iiv_names.index(b)]
            else:
                a = problem.iiv_names.index(name)
                x[j] = math.log(chol[a, a])
        elif kind == 'iov':
            x[j] = .5 * math.log(iov[name])
        elif kind == 'sigma':
            out, comp = name.split(':')
            col = dict(proportional=0, additive=1, lognormal=2)[comp]
            x[j] = math.log(sigma[out][col])
    return x


def designs():
    rng = np.random.default_rng(20260930)
    d = {}
    rows = ['ID,TIME,EVID,AMT,CMT,DV,DVID,CENS,WT,SEX']
    for i in range(1, 13):
        sex = int(rng.random() < .5)
        wt = [rng.uniform(55, 95) for _ in range(2)]
        rows.append(f'{i},0,1,300,1,.,1,0,{wt[0]:.1f},{sex}')
        for t in (.5, 1, 2, 4, 8):
            rows.append(f'{i},{t},0,0,1,0,1,0,{wt[0]:.1f},{sex}')
        rows.append(f'{i},10,2,0,1,.,1,0,{wt[1]:.1f},{sex}')
        for t in (12, 16, 24):
            rows.append(f'{i},{t},0,0,1,0,1,0,{wt[1]:.1f},{sex}')
    d['block_tv_categorical'] = ('\n'.join(rows), dict(covariates=['WT', 'SEX']), None)
    rows = ['ID,TIME,EVID,AMT,CMT,DV,DVID,CENS,OCC']
    for i in range(1, 11):
        for occ, t0 in ((1, 0), (2, 72)):
            if occ == 2:
                rows.append(f'{i},{t0},3,0,1,.,1,0,{occ}')
            rows.append(f'{i},{t0},1,500,1,.,1,0,{occ}')
            for t in (1, 4, 12, 24, 36):
                rows.append(f'{i},{t0 + t},0,0,1,0,1,0,{occ}')
    d['iov_blq_lognormal'] = ('\n'.join(rows), dict(occasion='OCC'), {'CP': 1.5})
    rows = ['ID,TIME,EVID,AMT,CMT,DV,DVID,CENS,AGE']
    for i in range(1, 13):
        age = rng.uniform(20, 70)
        rows.append(f'{i},0,1,100,2,.,1,0,{age:.1f}')
        for t in (.25, 1, 3, 8):
            rows.append(f'{i},{t},0,0,2,0,1,0,{age:.1f}')
        rows.append(f'{i},24,1,200,1,.,1,0,{age:.1f}')
        for t in (25, 26, 28, 32, 40):
            rows.append(f'{i},{t},0,0,2,0,1,0,{age:.1f}')
    d['bioavailability_linear'] = ('\n'.join(rows), dict(covariates=['AGE']), None)
    rows = ['ID,TIME,EVID,AMT,CMT,DV,DVID,CENS']
    for i in range(1, 9):
        rows.append(f'{i},0,1,150,1,.,1,0')
        for t in (1, 3, 8, 24):
            rows.append(f'{i},{t},0,0,1,0,1,0')
        for t in (2, 12, 24, 48, 72):
            rows.append(f'{i},{t + .01},0,0,1,0,2,0')
    d['idr_two_outputs'] = ('\n'.join(rows), dict(), None)
    return d


MODELS = dict(block_tv_categorical=model_block_tv, iov_blq_lognormal=model_iov_blq_lognormal,
              bioavailability_linear=model_bioavailability_linear_cov, idr_two_outputs=model_idr_two_outputs)


def main():
    import pkpy2
    from pkpy2._general.problem import GeneralProblem
    from pkpy2._general.importance import Bank
    OUT.mkdir(parents=True, exist_ok=True)
    report = []
    for name, (text, read_opts, lloq) in designs().items():
        model, reference_ofv, coords = MODELS[name](None)
        data_path = OUT / f'{name}_data.csv'
        if not data_path.exists():      # simulated once; later runs reuse the same data
            design_path = OUT / f'{name}_design.csv'
            design_path.write_text(text)
            design = pkpy2.read_nonmem(design_path, **read_opts)
            data = pkpy2.simulate_data(model, design, seed=7, lloq=lloq)
            # write the simulated data for the reference implementation
            cols = text.splitlines()[0].split(',')
            lines = [','.join(cols)]
            for ind in data:
                for k in range(len(ind.time)):
                    row = dict(ID=ind.id, TIME=ind.time[k], EVID=ind.evid[k], AMT=ind.amt[k], CMT=ind.cmt[k],
                               DV=ind.dv[k] if ind.evid[k] == 0 else '.', DVID=ind.dvid[k], CENS=ind.cens[k])
                    for c in data.covariate_names:
                        row[c] = ind.covariates[c][k]
                    if 'OCC' in cols:
                        row['OCC'] = ind.occasion[k]
                    lines.append(','.join(str(row[c]) for c in cols))
            data_path.write_text('\n'.join(lines))
        data = pkpy2.read_nonmem(data_path, **read_opts)
        problem = GeneralProblem(data, model)
        x = coords(problem)
        banks = [Bank(problem, x, power=16, seed=s).evaluate(x, gradient=False)[0] for s in (101, 202)]
        # the reference is cached per data file content (the quadrature is slow for ODE models)
        digest = hashlib.sha256(data_path.read_bytes()).hexdigest()
        cache = OUT / f'{name}_reference.json'
        cached = json.loads(cache.read_text()) if cache.exists() else {}
        if cached.get('sha256') == digest:
            ref = cached['reference_ofv']
        else:
            ref = reference_ofv(read_rows(data_path))
            cache.write_text(json.dumps(dict(sha256=digest, reference_ofv=float(ref))))
        censored = sum(int(np.sum(i.cens != 0)) for i in data)
        row = dict(model=name, subjects=len(data), observations=data.n_observations, censored=censored,
                   pkpy2_ofv=[float(b) for b in banks], reference_ofv=float(ref),
                   difference=float(np.mean(banks) - ref), bank_range=float(abs(banks[0] - banks[1])))
        report.append(row)
        print(json.dumps(row))
    (OUT.parent / 'likelihood_agreement.json').write_text(json.dumps(report, indent=1))


if __name__ == '__main__':
    main()
