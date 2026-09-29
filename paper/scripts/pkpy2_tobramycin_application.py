"""PKPy2 application to the tobramycin dataset used in the PKGPT expert benchmark.

Data: PKGPT repository dataset/tobramycin.csv (97 subjects, 858 intravenous doses recorded
as bolus events, 236 observations). Model: the published expert model, i.e. two-compartment
intravenous, CL = CL58 x (CLCR/58)^b1 with IIV, V1 = V1_62 x (WT/62)^b2, Q and V2 without
IIV, combined residual error with the additive variance fixed at 1.85e-6.

Two starting points were fixed before fitting: start 1 (primary) uses the initial guesses
documented with the expert model (CL 3.4 L/h, V1 20.3 L, exponents 1) with Q = 5 L/h and
V2 = 20 L; start 2 uses the published expert estimates. Fit and uncertainty options are
those of the other PKPy2 analyses.
"""
from pathlib import Path
import csv
import hashlib
import json
import math
import sys

import numpy as np
from pkpy2 import Subject, ModelSpec, Parameter as P, Covariate, fit

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / 'output/pkpy2_software_comparison/source/tobramycin_github.csv'
OUT = ROOT / 'output/pkpy2_tobramycin_v3'
GITHUB_BLOB = 'de8f700e8a30601ec3350bec2ce3a348dc5ca231'
FIT_OPTIONS = dict(workers=2, saem_options=dict(cpu_budget_seconds=60.),
                   refinement_options=dict(cpu_budget_seconds=300., max_stages=90, analytic_non_eta=True))
CI_OPTIONS = dict(power=16, step=.002, workers=2, confidence=.95)
ADD_SD = math.sqrt(1.85e-6)
STARTS = {
    'start_1': dict(CL=3.4, V1=20.3, Q=5.0, V2=20.0, b_clcr=1.0, b_wt=1.0, omega_CL=0.1, sigma_prop=0.2),
    'start_2': dict(CL=2.95, V1=4.59, Q=6.85, V2=13.2, b_clcr=0.236, b_wt=1.07, omega_CL=0.028, sigma_prop=0.2),
}
SEEDS = dict(start_1=260930101, start_2=260930102, uncertainty=260930201)


def subjects():
    raw = DATA.read_bytes()
    assert hashlib.sha1(b'blob %d\x00' % len(raw) + raw).hexdigest() == GITHUB_BLOB
    rows = list(csv.DictReader(raw.decode('utf-8').lstrip('#').splitlines()))
    out = []
    for sid in dict.fromkeys(r['ID'] for r in rows):
        rr = [r for r in rows if r['ID'] == sid]
        doses = [(float(r['TIME']), float(r['AMT'])) for r in rr if r['EVID'] == '1']
        obs = [r for r in rr if r['EVID'] == '0']
        assert all(r['RATE'] == '0' for r in rr)
        out.append(Subject(int(sid), np.array([float(r['TIME']) for r in obs]), np.array([float(r['DV']) for r in obs]),
                           doses[0][1], covariates=dict(WT=float(rr[0]['WT']), CLCR=float(rr[0]['CLCR'])),
                           dose_history=doses))
    return out


def spec(start):
    return ModelSpec('2cmt_iv', {n: P(start[n]) for n in ('CL', 'V1', 'Q', 'V2')}, {'CL': P(start['omega_CL'])},
                     P(start['sigma_prop']), P(ADD_SD, True),
                     (Covariate('CL', 'CLCR', 58., P(start['b_clcr'])), Covariate('V1', 'WT', 62., P(start['b_wt']))))


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    subs = subjects()
    assert len(subs) == 97 and sum(len(s.obs) for s in subs) == 236 and sum(len(s.dose_history) for s in subs) == 858
    (OUT / 'protocol.json').write_text(json.dumps(dict(data=str(DATA.relative_to(ROOT)), github_blob=GITHUB_BLOB, starts=STARTS,
        seeds=SEEDS, fit_options=FIT_OPTIONS, uncertainty_options=CI_OPTIONS, additive_sd_fixed=ADD_SD,
        primary='start_1'), indent=1), encoding='utf-8')
    which = sys.argv[1:] or list(STARTS)
    for name in which:
        path = OUT / f'{name}.json'
        if path.exists():
            continue
        fitted = fit(subs, spec(STARTS[name]), seed=SEEDS[name], **FIT_OPTIONS)
        print(name, fitted.status, fitted.to_dict()['theta'], flush=True)
        if name == 'start_1' and fitted.converged:
            fitted.uncertainty(seed=SEEDS['uncertainty'], **CI_OPTIONS)
        path.write_text(json.dumps(fitted.to_dict(), indent=1, default=lambda x: x.tolist() if hasattr(x, 'tolist') else float(x)),
                        encoding='utf-8')


if __name__ == '__main__':
    main()
