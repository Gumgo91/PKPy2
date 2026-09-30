"""Parameter recovery of the general engine for the new model features (known truth).

Usage: python validate_pkpy2_extended_recovery.py <scenario> <replicates> [first_replicate]
Each replicate simulates a dataset from the true model, fits it from perturbed
starting values and stores the record; existing records are skipped.
Summaries: python validate_pkpy2_extended_recovery.py summarize
"""
from pathlib import Path
import json
import math
import sys
import time
import zlib

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'packages/pkpy2/src'))
sys.path.insert(0, str(ROOT / 'scripts'))
import pkpy2                                                                          # noqa: E402
from pkpy2 import Parameter as P, Covariate, Model, Residual, structures as S          # noqa: E402

OUT = ROOT / 'output/pkpy2_extended_validation/recovery'


def complex_design(rng, n=40):
    rows = ['ID,TIME,EVID,AMT,CMT,RATE,II,SS,DV,OCC,WT,SEX']
    for i in range(1, n + 1):
        wt1 = rng.uniform(50, 100)
        wt2 = wt1 * rng.uniform(.85, 1.15)
        sex = int(rng.random() < .5)
        rows.append(f'{i},0,1,500,1,250,12,1,.,1,{wt1:.2f},{sex}')
        for t in (.5, 1, 2, 4, 6, 9, 11.9):
            rows.append(f'{i},{t},0,0,1,0,0,0,0,1,{wt1:.2f},{sex}')
        rows.append(f'{i},168,3,0,1,0,0,0,.,2,{wt2:.2f},{sex}')
        rows.append(f'{i},168,1,800,1,400,0,0,.,2,{wt2:.2f},{sex}')
        for t in (.5, 1, 2, 4, 8, 12, 24, 36):
            rows.append(f'{i},{168 + t},0,0,1,0,0,0,0,2,{wt2:.2f},{sex}')
    return rows, dict(covariates=['WT', 'SEX'], occasion='OCC')


def pkpd_design(rng, n=40):
    rows = ['ID,TIME,EVID,AMT,CMT,RATE,II,SS,DV,DVID']
    for i in range(1, n + 1):
        rows.append(f'{i},0,1,150,1,0,0,0,.,1')
        for t in (1, 3, 8, 24, 48):
            rows.append(f'{i},{t},0,0,1,0,0,0,0,1')
        for t in (0, 12, 24, 48, 72, 96, 120):
            rows.append(f'{i},{t + .01},0,0,1,0,0,0,0,2')
    return rows, {}


SCENARIOS = {
    'complex_linear': dict(
        design=complex_design, lloq={'CP': .4},
        truth=lambda: Model(S.pk(2), theta=dict(CL=P(5.), V1=P(20.), Q=P(8.), V2=P(40.)),
                            omega=dict(CL=P(.09), V1=P(.06)), omega_blocks=[('CL', 'V1')],
                            omega_covariance={('CL', 'V1'): P(.04)}, iov=dict(CL=P(.03)),
                            covariates=(Covariate('CL', 'WT', 70., P(.75)), Covariate('V1', 'SEX', 0, P(-.2), 'categorical', 1)),
                            residual=Residual(proportional=P(.12), additive=P(.05))),
        start=lambda: Model(S.pk(2), theta=dict(CL=P(4.), V1=P(25.), Q=P(6.), V2=P(50.)),
                            omega=dict(CL=P(.1), V1=P(.1)), omega_blocks=[('CL', 'V1')], iov=dict(CL=P(.05)),
                            covariates=(Covariate('CL', 'WT', 70., P(.5)), Covariate('V1', 'SEX', 0, P(0.), 'categorical', 1)),
                            residual=Residual(proportional=P(.2), additive=P(.1)))),
    'pkpd_idr': dict(
        design=pkpd_design, lloq=None,
        truth=lambda: Model(S.indirect_response(3), theta=dict(CL=P(2.), V=P(25.), Ka=P(1.3), R0=P(80.), KOUT=P(.15),
                                                               EMAX=P(1.5), EC50=P(1.)),
                            omega=dict(CL=P(.09), V=P(.04), KOUT=P(.09), EC50=P(.16)),
                            residual=dict(CP=Residual(proportional=P(.12)), R=Residual(additive=P(4.)))),
        start=lambda: Model(S.indirect_response(3), theta=dict(CL=P(1.6), V=P(30.), Ka=P(1.), R0=P(70.), KOUT=P(.1),
                                                               EMAX=P(1.), EC50=P(1.5)),
                            omega=dict(CL=P(.1), V=P(.1), KOUT=P(.1), EC50=P(.1)),
                            residual=dict(CP=Residual(proportional=P(.2)), R=Residual(additive=P(6.))))),
}


def run(name, replicates, first=0, descending=False):
    sc = SCENARIOS[name]
    out = OUT / name
    out.mkdir(parents=True, exist_ok=True)
    order = range(first, first + replicates)
    for r in (reversed(order) if descending else order):
        path = out / f'rep_{r:03d}.json'
        if path.exists():
            continue
        rng = np.random.default_rng(zlib.crc32(f'{name}:{r}'.encode()))
        rows, opts = sc['design'](rng)
        dpath = out / f'rep_{r:03d}_design.csv'
        dpath.write_text('\n'.join(rows))
        design = pkpy2.read_nonmem(dpath, **opts)
        data = pkpy2.simulate_data(sc['truth'](), design, seed=zlib.crc32(f'{name}:{r}:data'.encode()), lloq=sc['lloq'])
        t0 = time.time()
        try:
            res = pkpy2.fit(data, sc['start'](), seed=20260930 + r)
            record = dict(replicate=r, status=res.status, converged=res.converged, ofv=float(res.ofv),
                          estimates=res.problem.describe(res.x), seconds=time.time() - t0,
                          censored=int(sum(int(np.sum(i.cens == 1)) for i in data)))
        except Exception as error:     # recorded as a failed replicate
            record = dict(replicate=r, status=f'error: {type(error).__name__}: {error}', converged=False,
                          seconds=time.time() - t0)
        path.write_text(json.dumps(record, indent=1, default=float))
        print(name, r, record['status'], round(record['seconds'], 1), flush=True)


def flatten(d):
    out = {}
    for group, values in d.items():
        for k, v in values.items():
            out[f'{group}:{k}'] = v
    return out


def summarize():
    from pkpy2._general.problem import GeneralProblem
    report = {}
    for name, sc in SCENARIOS.items():
        recs = [json.loads(p.read_text()) for p in sorted((OUT / name).glob('rep_[0-9][0-9][0-9].json'))]
        if not recs:
            continue
        rows, opts = sc['design'](np.random.default_rng(0))
        dpath = OUT / name / 'rep_000_design.csv'
        design = pkpy2.read_nonmem(dpath, **opts)
        data = pkpy2.simulate_data(sc['truth'](), design, seed=1)
        prob = GeneralProblem(data, sc['truth']())
        truth = flatten(prob.describe(prob.x0))
        ok = [r for r in recs if r.get('converged')]
        rows = []
        for key, true in truth.items():
            vals = np.array([flatten(r['estimates'])[key] for r in ok])
            if not len(vals) or true == 0:
                continue
            rel = (vals - true) / abs(true)
            rows.append(dict(quantity=key, truth=true, n=len(vals), relative_bias_pct=float(100 * rel.mean()),
                             relative_rmse_pct=float(100 * np.sqrt(np.mean(rel ** 2)))))
        report[name] = dict(replicates=len(recs), converged=len(ok),
                            median_seconds=float(np.median([r['seconds'] for r in recs])), estimates=rows)
    (OUT.parent / 'recovery_summary.json').write_text(json.dumps(report, indent=1))
    print(json.dumps(report, indent=1))


if __name__ == '__main__':
    if sys.argv[1] == 'summarize':
        summarize()
    else:
        run(sys.argv[1], int(sys.argv[2]), int(sys.argv[3]) if len(sys.argv) > 3 else 0,
            descending='--descending' in sys.argv)
