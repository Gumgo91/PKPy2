"""The general engine reproduces the classic engine on the classic model set.

Refits, with the general engine (pkpy2.Model), the theophylline and warfarin
analyses and ten primary simulation datasets (five rich, five sparse) whose
classic-engine estimates are stored in the v3 records, from the same starting
values. Output: output/pkpy2_extended_validation/general_vs_classic.json
"""
from pathlib import Path
import json
import sys
import time

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'packages/pkpy2/src'))
import pkpy2                                                             # noqa: E402
from pkpy2 import Subject, Parameter as P, Covariate, Model, Residual, structures as S   # noqa: E402
from pkpy2.data import Dataset                                          # noqa: E402

DEV = ROOT / 'output/pkpy2_development'
OUT = ROOT / 'output/pkpy2_extended_validation'


def theophylline():
    rows = json.loads((DEV / 'theophylline_v3/data.json').read_text())
    subjects = [Subject(r['sid'], np.array(r['time']), np.array(r['obs']), r['dose'], r['covariates']) for r in rows]
    model = Model(S.pk(1, 'first_order', lag=True), theta=dict(CL=P(3.), V=P(30.), Ka=P(1.), ALAG=P(0., True)),
                  omega=dict(CL=P(.1), V=P(.03), Ka=P(.3)),
                  covariates=tuple(Covariate(n, 'WT', 70., P(1., True)) for n in ('CL', 'V')),
                  residual=Residual(proportional=P(.15)))
    return subjects, model, json.loads((DEV / 'theophylline_v3/proportional.json').read_text())


def warfarin():
    rows = json.loads((ROOT / 'output/pkpy2_nonmem_reference/warfarin_data.json').read_text())
    subjects = [Subject(r['sid'], np.array(r['time']), np.array(r['obs']), r['dose'], r['covariates']) for r in rows]
    model = Model(S.pk(1, 'first_order', lag=True), theta=dict(CL=P(.2), V=P(10.), Ka=P(.8), ALAG=P(.5)),
                  omega=dict(CL=P(.1), Ka=P(.5, True)),
                  covariates=(Covariate('CL', 'WT', 70., P(.75, True)), Covariate('V', 'WT', 70., P(1., True))),
                  residual=Residual(proportional=P(.2), additive=P(np.sqrt(.117), True)))
    return subjects, model, json.loads((DEV / 'warfarin_v3/warfarin_start_1.json').read_text())


def primary(name):
    rec = json.loads((DEV / f'confirmatory_v3/{name}.json').read_text())
    data = json.loads((DEV / f"confirmatory_v3/{rec['data_file']}").read_text())
    subjects = [Subject(i + 1, np.array(t), np.array(y), data['dose']) for i, (t, y) in
                enumerate(zip(data['time'], data['observations']))]
    tr = rec['truth']
    model = Model(S.pk(1), theta={k: P(.8 * v) for k, v in tr['theta'].items()},
                  omega={k: P(.08) for k in tr['omega']}, residual=Residual(proportional=P(.2)))
    return subjects, model, rec['pkpy2']


def compare(name, subjects, model, classic):
    t0 = time.time()
    res = pkpy2.fit(Dataset.from_subjects(subjects), model, seed=20260930)
    general = dict(theta=res.theta, omega=res.omega, sigma_prop=res.sigma['CP:proportional'])
    rel = {}
    for k, v in classic['theta'].items():
        if v > 0 and not (k == 'ALAG' and v == 0):
            rel[f'theta:{k}'] = 100 * (general['theta'][k] / v - 1)
    for k, v in classic['omega'].items():
        rel[f'omega:{k}'] = 100 * (general['omega'][k] / v - 1)
    rel['sigma_prop'] = 100 * (general['sigma_prop'] / classic['sigma']['sigma_prop'] - 1)
    row = dict(dataset=name, general_status=res.status, general_ofv=float(res.ofv), classic_ofv=float(classic['ofv']),
               ofv_difference=float(res.ofv - classic['ofv']), relative_difference_pct=rel,
               max_abs_relative_difference_pct=float(max(abs(v) for v in rel.values())),
               general_seconds=time.time() - t0)
    print(json.dumps(row), flush=True)
    return row


def main():
    rows = [compare('theophylline', *theophylline()), compare('warfarin', *warfarin())]
    for s in ('rich', 'sparse'):
        for r in range(5):
            name = f'1cmt_iv__{s}__{r:03d}'
            rows.append(compare(name, *primary(name)))
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / 'general_vs_classic.json').write_text(json.dumps(rows, indent=1))


if __name__ == '__main__':
    main()
