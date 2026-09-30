"""PKPy2 (general engine) and nlmixr2 FOCEi on simulated datasets with the new model features.

Usage:
  python validate_pkpy2_extended_vs_nlmixr2.py write     # simulate the datasets (known truth)
  Rscript validate_pkpy2_extended_vs_nlmixr2.R           # nlmixr2 FOCEi fits
  python validate_pkpy2_extended_vs_nlmixr2.py fit       # PKPy2 fits + exact OFV at both sets of estimates
"""
from pathlib import Path
import csv
import json
import zlib
import math
import sys
import time

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'packages/pkpy2/src'))
import pkpy2                                                                          # noqa: E402
from pkpy2 import Parameter as P, Covariate, Model, Residual, structures as S          # noqa: E402

OUT = ROOT / 'output/pkpy2_extended_validation/vs_nlmixr2'


def design_infusion(rng):
    rows = ['ID,TIME,EVID,AMT,CMT,RATE,II,SS,DV,CENS,WT,SEX']
    for i in range(1, 61):
        wt, sex = rng.uniform(50, 100), int(rng.random() < .5)
        rows.append(f'{i},0,1,500,1,250,12,1,.,0,{wt:.1f},{sex}')
        for t in (.5, 1, 2, 3, 6, 9, 11.5):
            rows.append(f'{i},{t},0,0,1,0,0,0,0,0,{wt:.1f},{sex}')
        rows.append(f'{i},12,1,500,1,250,0,0,.,0,{wt:.1f},{sex}')
        for t in (13, 16, 24, 36):
            rows.append(f'{i},{t},0,0,1,0,0,0,0,0,{wt:.1f},{sex}')
    return rows, dict(covariates=['WT', 'SEX'])


def design_oral(rng, n=50, times=(.5, 1, 2, 4, 8, 12, 24, 36, 48)):
    rows = ['ID,TIME,EVID,AMT,CMT,RATE,II,SS,DV,CENS']
    for i in range(1, n + 1):
        rows.append(f'{i},0,1,100,1,0,0,0,.,0')
        for t in times:
            rows.append(f'{i},{t},0,0,2,0,0,0,0,0')
    return rows, {}


def design_iv(rng, n=40, amt=100, times=(.25, .5, 1, 2, 4, 6, 8, 12, 24)):
    rows = ['ID,TIME,EVID,AMT,CMT,RATE,II,SS,DV,CENS']
    for i in range(1, n + 1):
        rows.append(f'{i},0,1,{amt * (1 + (i % 3))},1,0,0,0,.,0')
        for t in times:
            rows.append(f'{i},{t},0,0,1,0,0,0,0,0')
    return rows, {}


SCENARIOS = {
    'infusion_block_covariates': dict(
        design=design_infusion,
        truth=lambda: Model(S.pk(2), theta=dict(CL=P(5.), V1=P(20.), Q=P(8.), V2=P(40.)),
                            omega=dict(CL=P(.09), V1=P(.06)), omega_blocks=[('CL', 'V1')],
                            omega_covariance={('CL', 'V1'): P(.04)},
                            covariates=(Covariate('CL', 'WT', 70., P(.75)), Covariate('V1', 'SEX', 0, P(-.2), 'categorical', 1)),
                            residual=Residual(proportional=P(.12), additive=P(.05))),
        start=lambda: Model(S.pk(2), theta=dict(CL=P(4.), V1=P(25.), Q=P(6.), V2=P(50.)),
                            omega=dict(CL=P(.1), V1=P(.1)), omega_blocks=[('CL', 'V1')],
                            covariates=(Covariate('CL', 'WT', 70., P(.5)), Covariate('V1', 'SEX', 0, P(-.01), 'categorical', 1)),
                            residual=Residual(proportional=P(.2), additive=P(.1))),
        lloq=None,
        mapping=dict(theta=dict(CL='tcl', V1='tv1', Q='tq', V2='tv2'), beta={'CL~WT': 'b.wt', 'V1~SEX=1': 'b.sex'},
                     omega=dict(CL='eta.cl', V1='eta.v1'), cov={('V1', 'CL'): ('eta.cl', 'eta.v1')},
                     sigma={'CP:proportional': 'prop.sd', 'CP:additive': 'add.sd'})),
    'oral_blq_m3': dict(
        design=design_oral,
        truth=lambda: Model(S.pk(1, 'first_order'), theta=dict(CL=P(3.), V=P(35.), Ka=P(1.2)),
                            omega=dict(CL=P(.09), V=P(.04), Ka=P(.2)), residual=Residual(proportional=P(.15))),
        start=lambda: Model(S.pk(1, 'first_order'), theta=dict(CL=P(2.5), V=P(30.), Ka=P(1.)),
                            omega=dict(CL=P(.1), V=P(.1), Ka=P(.1)), residual=Residual(proportional=P(.2))),
        lloq={'CP': .25},
        mapping=dict(theta=dict(CL='tcl', V='tv', Ka='tka'), omega=dict(CL='eta.cl', V='eta.v', Ka='eta.ka'),
                     sigma={'CP:proportional': 'prop.sd'})),
    'lognormal_residual': dict(
        design=design_iv,
        truth=lambda: Model(S.pk(1), theta=dict(CL=P(4.), V=P(40.)), omega=dict(CL=P(.09), V=P(.04)),
                            residual=Residual(lognormal=P(.2))),
        start=lambda: Model(S.pk(1), theta=dict(CL=P(3.), V=P(30.)), omega=dict(CL=P(.1), V=P(.1)),
                            residual=Residual(lognormal=P(.3))),
        lloq=None,
        mapping=dict(theta=dict(CL='tcl', V='tv'), omega=dict(CL='eta.cl', V='eta.v'), sigma={'CP:lognormal': 'lnorm.sd'})),
    'michaelis_menten': dict(
        design=lambda rng: design_iv(rng, 40, 300, (.25, .5, 1, 2, 4, 6, 8, 12, 16, 24)),
        truth=lambda: Model(S.michaelis_menten(1), theta=dict(V=P(20.), VMAX=P(40.), KM=P(4.)),
                            omega=dict(V=P(.05), VMAX=P(.09)), residual=Residual(proportional=P(.1), additive=P(.1))),
        start=lambda: Model(S.michaelis_menten(1), theta=dict(V=P(25.), VMAX=P(30.), KM=P(3.)),
                            omega=dict(V=P(.1), VMAX=P(.1)), residual=Residual(proportional=P(.2), additive=P(.2))),
        lloq=None,
        mapping=dict(theta=dict(V='tv', VMAX='tvmax', KM='tkm'), omega=dict(V='eta.v', VMAX='eta.vmax'),
                     sigma={'CP:proportional': 'prop.sd', 'CP:additive': 'add.sd'})),
}


def write():
    OUT.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(20260930)
    for name, sc in SCENARIOS.items():
        rows, opts = sc['design'](rng)
        path = OUT / f'{name}_design.csv'
        path.write_text('\n'.join(rows))
        design = pkpy2.read_nonmem(path, **opts)
        data = pkpy2.simulate_data(sc['truth'](), design, seed=zlib.crc32(name.encode()), lloq=sc['lloq'])
        cols = rows[0].split(',')
        lines = [rows[0]]
        for ind in data:
            for k in range(len(ind.time)):
                row = dict(ID=ind.id, TIME=ind.time[k], EVID=ind.evid[k], AMT=ind.amt[k], CMT=ind.cmt[k],
                           RATE=ind.rate[k], II=ind.ii[k], SS=ind.ss[k],
                           DV=f'{ind.dv[k]:.6g}' if ind.evid[k] == 0 else '.', CENS=ind.cens[k])
                for c in data.covariate_names:
                    row[c] = ind.covariates[c][k]
                lines.append(','.join(str(row[c]) for c in cols))
        (OUT / f'{name}.csv').write_text('\n'.join(lines))
        cens = sum(int(np.sum(i.cens == 1)) for i in data)
        print(name, 'subjects', len(data), 'observations', data.n_observations, 'censored', cens)


def nlmixr2_x(problem, mapping, r):
    th, om = r['theta'], r['omega']
    x = np.zeros(len(problem.labels))
    pop_cov = r.get('omega_matrix')
    for j, lab in enumerate(problem.labels):
        kind, _, name = lab.partition(':')
        if kind == 'theta':
            x[j] = th[mapping['theta'][name]]
        elif kind == 'beta':
            x[j] = th[mapping['beta'][name]]
        elif kind == 'sigma':
            x[j] = math.log(th[mapping['sigma'][name]])
    # omega: Cholesky of the nlmixr2 covariance in PKPy2's parameter order
    names = problem.iiv_names
    cov = np.zeros((len(names), len(names)))
    for a, na in enumerate(names):
        cov[a, a] = om[mapping['omega'][na]]
    for (a, b), (ea, eb) in mapping.get('cov', {}).items():
        v = r['omega_cov'][f'{ea},{eb}']
        ia, ib = names.index(a), names.index(b)
        cov[ia, ib] = cov[ib, ia] = v
    L = np.zeros_like(cov)
    for g in problem.omega_groups:
        sub = np.linalg.cholesky(cov[np.ix_(g, g)])
        for p_, a in enumerate(g):
            for q_, b in enumerate(g):
                L[a, b] = sub[p_, q_]
    for row, col, kind, coord, _ in problem.chol_plan:
        if coord >= 0:
            x[coord] = math.log(L[row, col]) if kind == 'log' else L[row, col]
    return x


def fit(names=None):
    from pkpy2._general.importance import Bank
    path_report = OUT.parent / 'vs_nlmixr2.json'
    previous = {r['scenario']: r for r in json.loads(path_report.read_text())} if names and path_report.exists() else {}
    report = []
    for name, sc in SCENARIOS.items():
        if names and name not in names:
            if name in previous:
                report.append(previous[name])
            continue
        opts = sc['design'](np.random.default_rng(0))[1]
        data = pkpy2.read_nonmem(OUT / f'{name}.csv', **opts)
        t0 = time.time()
        res = pkpy2.fit(data, sc['start'](), seed=20260930)
        seconds = time.time() - t0
        res.save(OUT / f'{name}_pkpy2.json')
        truth = pkpy2._general.problem.GeneralProblem(data, sc['truth']()).describe(
            pkpy2._general.problem.GeneralProblem(data, sc['truth']()).x0)
        row = dict(scenario=name, pkpy2_status=res.status, pkpy2_ofv=float(res.ofv), pkpy2_seconds=seconds,
                   pkpy2=res.problem.describe(res.x), truth=truth)
        path = OUT / f'{name}_nlmixr2.json'
        if path.exists():
            r = json.loads(path.read_text())
            x = nlmixr2_x(res.problem, sc['mapping'], r)
            vals = [Bank(res.problem, x, power=14, seed=s).evaluate(x, gradient=False)[0] for s in (5, 6)]
            row.update(nlmixr2_objective=r['objective'], nlmixr2_seconds=r['seconds'],
                       nlmixr2=res.problem.describe(x), exact_ofv_at_nlmixr2=float(np.mean(vals)),
                       exact_ofv_difference=float(np.mean(vals) - res.ofv))
        report.append(row)
        print(json.dumps(row, default=float), flush=True)
    order = list(SCENARIOS)
    report.sort(key=lambda r: order.index(r['scenario']))
    path_report.write_text(json.dumps(report, indent=1, default=float))


if __name__ == '__main__':
    if sys.argv[1] == 'write':
        write()
    else:
        fit(sys.argv[2:] or None)
