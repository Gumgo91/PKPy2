"""PKPy2 fit of the warfarin PK/PD turnover model (the nlmixr2 example model, nlmixr2data::warfarin).

Structure: depot -> gut (KTR) -> central (Ka), linear elimination CL/V; the
prothrombin complex activity (PCA) follows an indirect response with inhibition
of production, I(C) = IMAX*C/(IC50 + C), baseline R0 and KOUT. IMAX is logit-
transformed; all eight parameters have IIV. Residuals: CP proportional +
additive; PCA additive. Starting values as in the nlmixr2 example.
Also evaluates the exact marginal OFV (PKPy2 importance sampling) at the nlmixr2
FOCEi and SAEM estimates when their JSON files are present.
"""
from pathlib import Path
import json
import math
import sys
import time

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'packages/pkpy2/src'))
import pkpy2                                             # noqa: E402
from pkpy2 import Parameter as P, Model, Residual, structures as S   # noqa: E402

DIR = ROOT / 'output/pkpy2_extended_validation/warfarin_pkpd'


def structure():
    pk = S.linear(['KTR', 'Ka', 'CL', 'V'], ['depot', 'gut', 'central'],
                  [('depot', 'depot', 'p', ('KTR',), -1.), ('gut', 'depot', 'p', ('KTR',), 1.),
                   ('gut', 'gut', 'p', ('Ka',), -1.), ('central', 'gut', 'p', ('Ka',), 1.),
                   ('central', 'central', 'ratio', ('CL', 'V'), -1.)],
                  [S.Output('CP', 'central', 'V')], name='warfarin_pk')
    return S.indirect_response(1, pk)


def model(start=None):
    s = structure()
    v = dict(KTR=1., Ka=1., CL=.1, V=10., R0=100., KOUT=.05, IMAX=.8, IC50=.5)
    om = dict(KTR=1., Ka=1., CL=2., V=1., IMAX=.5, IC50=.5, KOUT=.5, R0=.5)
    sig = dict(prop=.1, add=.1, pd=10.)
    if start:
        v, om, sig = start
    return Model(s, theta={k: P(v[k]) for k in s.parameters}, omega={k: P(om[k]) for k in om},
                 transforms=dict(IMAX='logit'),
                 residual=dict(CP=Residual(proportional=P(sig['prop']), additive=P(sig['add'])),
                               R=Residual(additive=P(sig['pd']))))


def nlmixr2_point(problem, path):
    r = json.loads(path.read_text())
    th, om = r['theta'], r['omega']
    names = dict(KTR='tktr', Ka='tka', CL='tcl', V='tv', IMAX='temax', IC50='tec50', KOUT='tkout', R0='te0')
    etas = dict(KTR='eta.ktr', Ka='eta.ka', CL='eta.cl', V='eta.v', IMAX='eta.emax', IC50='eta.ec50',
                KOUT='eta.kout', R0='eta.e0')
    x = np.zeros(len(problem.labels))
    for j, lab in enumerate(problem.labels):
        kind, _, name = lab.partition(':')
        if kind == 'theta':
            x[j] = th[names[name]]
        elif kind == 'omega':
            x[j] = .5 * math.log(om[etas[name]])
        elif lab == 'sigma:CP:proportional':
            x[j] = math.log(th['prop.err'])
        elif lab == 'sigma:CP:additive':
            x[j] = math.log(th['pkadd.err'])
        elif lab == 'sigma:R:additive':
            x[j] = math.log(th['pdadd.err'])
    return x, r


def restore(data):
    """The saved PKPy2 fit as a result object (for diagnostics without refitting)."""
    from pkpy2._general.problem import GeneralProblem
    from pkpy2._general.fit import GeneralFitResult
    saved = json.loads((DIR / 'pkpy2_fit.json').read_text())
    problem = GeneralProblem(data, model())
    assert saved['data_sha256'] == problem.data_sha256 and saved['coordinates'] == problem.labels
    return GeneralFitResult(problem, np.array(saved['x']), saved['ofv'], saved['status'], saved['audit'],
                            saved['estimation'], saved['seconds'])


def diagnose(res):
    """Diagnostics, VPC and local intervals of the PKPy2 fit."""
    t0 = time.time()
    table, summary = pkpy2.diagnostics(res, npde_samples=1000)
    check = pkpy2.vpc(res, n=500, bins=8)
    unc = res.uncertainty() if res.converged else None
    out = dict(summary=summary, vpc={k: {n: (v.tolist() if hasattr(v, 'tolist') else v) for n, v in e.items()
                                          if n != 'observations'} for k, e in check.items()},
               observations={k: dict(time=e['observations']['time'].tolist(), dv=e['observations']['dv'].tolist())
                             for k, e in check.items()},
               table={k: (v.tolist() if hasattr(v, 'tolist') else v) for k, v in table.items()},
               uncertainty=None if unc is None else dict(status=unc['status'], intervals=unc.get('intervals', []),
                                                         robust_intervals=unc.get('robust_intervals', []),
                                                         eigenvalues=unc.get('eigenvalues'),
                                                         coordinates=unc.get('coordinates')),
               seconds=time.time() - t0)
    (DIR / 'pkpy2_diagnostics.json').write_text(json.dumps(out, indent=1, default=float))
    print('diagnostics', json.dumps(summary, default=float), 'uncertainty', None if unc is None else unc['status'],
          round(time.time() - t0, 1), 's', flush=True)


def main():
    data = pkpy2.read_nonmem(DIR / 'warfarin_pkpd.csv', covariates=['WT', 'AGE', 'SEX'])
    if '--diagnostics-only' in sys.argv:
        diagnose(restore(data))
        return
    t0 = time.time()
    res = pkpy2.fit(data, model(), seed=20260930,
                    callback=lambda r: print({k: (round(v, 3) if isinstance(v, float) else v) for k, v in r.items()
                                              if k in ('phase', 'start', 'ofv', 'stage', 'accepted', 'independent_ofv',
                                                       'projected_score_max', 'power', 'minimum_ess')}, flush=True))
    res.save(DIR / 'pkpy2_fit.json')
    print('status', res.status, res.ofv, round(time.time() - t0, 1), 's', flush=True)
    print(json.dumps(dict(theta=res.theta, omega=res.omega, sigma=res.sigma), indent=1), flush=True)
    from pkpy2._general.importance import Bank
    rows = [dict(source='PKPy2', ofv=float(res.ofv), status=res.status)]
    for method in ('focei', 'saem'):
        path = DIR / f'nlmixr2_{method}.json'
        if not path.exists():
            continue
        x, r = nlmixr2_point(res.problem, path)
        values = [Bank(res.problem, x, power=14, seed=s).evaluate(x, gradient=False)[0] for s in (11, 12)]
        rows.append(dict(source=f'nlmixr2 {method}', ofv=float(np.mean(values)), bank_range=float(abs(values[0] - values[1])),
                         nlmixr2_objective=r['objective'], seconds=r['seconds']))
    (DIR / 'exact_ofv.json').write_text(json.dumps(rows, indent=1))
    print(json.dumps(rows, indent=1))
    diagnose(res)

if __name__ == '__main__':
    main()
