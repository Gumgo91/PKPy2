"""Uncertainty and model-building tools of PKPy2 (general engine).

theophylline: 95% intervals of the population parameters of the theophylline
    model by Wald (observed information), sandwich, profile likelihood,
    nonparametric bootstrap and SIR.
scm: stepwise covariate modeling (forward p < 0.05, backward p < 0.01) on
    replicate simulated datasets with two true effects (CL~CRCL, V~WT) and five
    null candidates; selection frequency of each candidate.

Usage: python validate_pkpy2_tools.py theophylline [bootstrap_replicates]
       python validate_pkpy2_tools.py scm <replicates>
       python validate_pkpy2_tools.py summarize
"""
from pathlib import Path
import json
import sys
import time
import zlib

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'packages/pkpy2/src'))
import pkpy2                                                                          # noqa: E402
from pkpy2 import Subject, Parameter as P, Covariate, Model, Residual, structures as S  # noqa: E402
from pkpy2.data import Dataset                                                        # noqa: E402

DEV = ROOT / 'output/pkpy2_development'
OUT = ROOT / 'output/pkpy2_extended_validation/tools'


def theophylline_data():
    rows = json.loads((DEV / 'theophylline_v3/data.json').read_text())
    return Dataset.from_subjects([Subject(r['sid'], np.array(r['time']), np.array(r['obs']), r['dose'],
                                          r['covariates']) for r in rows])


def theophylline_model():
    return Model(S.pk(1, 'first_order', lag=True), theta=dict(CL=P(3.), V=P(30.), Ka=P(1.), ALAG=P(0., True)),
                 omega=dict(CL=P(.1), V=P(.03), Ka=P(.3)),
                 covariates=tuple(Covariate(n, 'WT', 70., P(1., True)) for n in ('CL', 'V')),
                 residual=Residual(proportional=P(.15)))


def _jsonable(v):
    if isinstance(v, np.ndarray):
        return v.tolist()
    if isinstance(v, np.generic):
        return v.item()
    return str(v)


def theophylline(n_boot=200):
    OUT.mkdir(parents=True, exist_ok=True)
    data = theophylline_data()
    t0 = time.time()
    res = pkpy2.fit(data, theophylline_model(), seed=20260930)
    report = dict(fit=dict(status=res.status, ofv=res.ofv, seconds=time.time() - t0, theta=res.theta,
                           omega=res.omega, sigma=res.sigma))
    print('fit', res.status, round(res.ofv, 4), round(time.time() - t0, 1), flush=True)
    t0 = time.time()
    unc = res.uncertainty()
    wald = {r['quantity']: r for r in unc['intervals']}
    robust = {r['quantity']: r for r in unc['robust_intervals']}
    report['wald'] = wald
    report['sandwich'] = robust
    report['uncertainty_seconds'] = time.time() - t0
    print('uncertainty', unc['status'], round(time.time() - t0, 1), flush=True)
    profiles = {}
    for key in ('theta:CL', 'theta:V', 'theta:Ka', 'omega:CL', 'omega:V', 'omega:Ka'):
        t0 = time.time()
        est = wald[key]['estimate']
        se_log = wald[key]['se'] / est
        values = est * np.exp(np.linspace(-3.2, 3.2, 13) * se_log)
        prof = pkpy2.profile(res, key, values)
        prof['seconds'] = time.time() - t0
        profiles[key] = prof
        print('profile', key, [None if v is None else round(v, 4) for v in prof['interval']],
              'wald', [round(v, 4) for v in wald[key]['interval']], round(prof['seconds'], 1), flush=True)
    report['profile'] = profiles
    t0 = time.time()
    s = pkpy2.sir(res, samples=1000, resamples=500, iterations=4, covariance=np.array(unc['covariance']))
    s['seconds'] = time.time() - t0
    report['sir'] = s
    print('sir', round(s['seconds'], 1), 'ess', round(s['effective_samples'], 1), flush=True)
    (OUT / 'theophylline_intervals.json').write_text(json.dumps(report, indent=1, default=_jsonable))
    t0 = time.time()
    boot = pkpy2.bootstrap(res, n_boot, seed=20261002,
                           callback=lambda r: print(' boot', r['replicate'], r['status'], flush=True))
    boot['seconds'] = time.time() - t0
    report['bootstrap'] = dict(summary=boot['summary'], converged=boot['converged'], requested=boot['requested'],
                               seconds=boot['seconds'])
    (OUT / 'theophylline_bootstrap_replicates.json').write_text(json.dumps(boot['replicates'], indent=1,
                                                                           default=_jsonable))
    (OUT / 'theophylline_intervals.json').write_text(json.dumps(report, indent=1, default=_jsonable))
    print('bootstrap', boot['converged'], 'of', n_boot, round(boot['seconds'], 1), flush=True)


# ------------------------------------------------------------------ stepwise covariate modeling

TRUE_EFFECTS = {'CL~CRCL[power]', 'V~WT[power]'}


def scm_candidates():
    return [Covariate('CL', 'WT', 70., P(.1)), Covariate('CL', 'AGE', 50., P(.1)),
            Covariate('CL', 'SEX', 0, P(0.), 'categorical', 1), Covariate('CL', 'CRCL', 80., P(.1)),
            Covariate('V', 'WT', 70., P(.1)), Covariate('V', 'AGE', 50., P(.1)),
            Covariate('V', 'SEX', 0, P(0.), 'categorical', 1)]


def scm_dataset(r, n=60):
    rng = np.random.default_rng(zlib.crc32(f'scm:{r}'.encode()))
    rows = ['ID,TIME,EVID,AMT,CMT,DV,WT,AGE,SEX,CRCL']
    for i in range(1, n + 1):
        wt, age, sex, crcl = rng.uniform(50, 100), rng.uniform(20, 80), int(rng.random() < .5), rng.uniform(30, 130)
        cov = f'{wt:.1f},{age:.0f},{sex},{crcl:.0f}'
        rows.append(f'{i},0,1,100,1,.,{cov}')
        for t in (.5, 1, 2, 4, 6, 8, 12, 24):
            rows.append(f'{i},{t},0,0,2,0,{cov}')
    path = OUT / 'scm' / f'design_{r:03d}.csv'
    path.write_text('\n'.join(rows))
    design = pkpy2.read_nonmem(path, covariates=['WT', 'AGE', 'SEX', 'CRCL'])
    truth = Model(S.pk(1, 'first_order'), theta=dict(CL=P(3.), V=P(35.), Ka=P(1.2)),
                  omega=dict(CL=P(.09), V=P(.04), Ka=P(.2)),
                  covariates=(Covariate('CL', 'CRCL', 80., P(.7)), Covariate('V', 'WT', 70., P(1.))),
                  residual=Residual(proportional=P(.15)))
    return pkpy2.simulate_data(truth, design, seed=zlib.crc32(f'scm:{r}:data'.encode()))


def scm(replicates, descending=False, first=0):
    (OUT / 'scm').mkdir(parents=True, exist_ok=True)
    base = Model(S.pk(1, 'first_order'), theta=dict(CL=P(2.5), V=P(30.), Ka=P(1.)),
                 omega=dict(CL=P(.1), V=P(.1), Ka=P(.1)), residual=Residual(proportional=P(.2)))
    for r in (reversed(range(first, replicates)) if descending else range(first, replicates)):
        path = OUT / 'scm' / f'rep_{r:03d}.json'
        if path.exists():
            continue
        t0 = time.time()
        data = scm_dataset(r)
        try:
            # candidate models start from the current estimates; one Laplace start per fit
            out = pkpy2.stepwise_covariates(data, base, scm_candidates(),
                                            fit_options=dict(seed=r, laplace_options=dict(starts=1)))
            record = dict(replicate=r, selected=out['covariates'], history=out['history'], seconds=time.time() - t0,
                          final_ofv=float(out['final'].ofv), final_status=out['final'].status)
        except Exception as error:     # recorded as a failed replicate
            record = dict(replicate=r, selected=None, error=f'{type(error).__name__}: {error}',
                          seconds=time.time() - t0)
        path.write_text(json.dumps(record, indent=1, default=_jsonable))
        print('scm', r, record['selected'], round(record['seconds'], 1), flush=True)


def summarize():
    report = {}
    path = OUT / 'theophylline_intervals.json'
    if path.exists():
        d = json.loads(path.read_text())
        rows = []
        for key in ('theta:CL', 'theta:V', 'theta:Ka', 'omega:CL', 'omega:V', 'omega:Ka', 'sigma:CP:proportional'):
            row = dict(quantity=key, estimate=d['wald'][key]['estimate'], wald=d['wald'][key]['interval'],
                       sandwich=d['sandwich'][key]['interval'], sir=d['sir']['summary'][key]['interval'])
            if key in d.get('profile', {}):
                row['profile'] = d['profile'][key]['interval']
            if 'bootstrap' in d and key in d['bootstrap']['summary']:
                row['bootstrap'] = d['bootstrap']['summary'][key]['interval']
            rows.append(row)
        report['theophylline'] = dict(intervals=rows, bootstrap_converged=d.get('bootstrap', {}).get('converged'),
                                      bootstrap_requested=d.get('bootstrap', {}).get('requested'),
                                      sir_effective_samples=d['sir']['effective_samples'])
    recs = [json.loads(p.read_text()) for p in sorted((OUT / 'scm').glob('rep_*.json'))]
    failed = [r for r in recs if r.get('selected') is None]
    recs = [r for r in recs if r.get('selected') is not None]
    if recs:
        labels = sorted({f"{c.parameter}~{c.covariate}" + (f'={c.level:g}' if c.level is not None else '') +
                         f'[{c.form}]' for c in scm_candidates()})
        freq = {c: float(np.mean([c in r['selected'] for r in recs])) for c in labels}
        report['scm'] = dict(replicates=len(recs), failed=len(failed), selection_frequency=freq,
                             exact_true_model=float(np.mean([set(r['selected']) == TRUE_EFFECTS for r in recs])),
                             median_seconds=float(np.median([r['seconds'] for r in recs])))
    (OUT.parent / 'tools_summary.json').write_text(json.dumps(report, indent=1))
    print(json.dumps(report, indent=1))


if __name__ == '__main__':
    if sys.argv[1] == 'theophylline':
        theophylline(int(sys.argv[2]) if len(sys.argv) > 2 else 200)
    elif sys.argv[1] == 'scm':
        first = int(sys.argv[sys.argv.index('--first') + 1]) if '--first' in sys.argv else 0
        scm(int(sys.argv[2]), descending='--descending' in sys.argv, first=first)
    else:
        summarize()
