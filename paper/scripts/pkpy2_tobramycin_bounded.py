"""Tobramycin with the parameter bounds of the expert NONMEM control stream.

The published expert model is fitted with the $THETA bounds of the expert analysis:
CL (0, 20) L/h, CLCR exponent (0, 2), V1 (0, 100) L, WT exponent (0, 2), Q (0, 50) L/h,
V2 (0, 100) L. Everything else is as in pkpy2_tobramycin_application.py. Starts: documented
initial values (start 1, primary) and the published expert estimates (start 2).
"""
from pathlib import Path
import json
import sys

from pkpy2 import ModelSpec, Parameter as P, Covariate, fit

sys.path.insert(0, str(Path(__file__).resolve().parent))
import pkpy2_tobramycin_application as T  # noqa: E402

OUT = T.ROOT / 'output/pkpy2_tobramycin_v3'
BOUNDS = dict(CL=(0., 20.), V1=(0., 100.), Q=(0., 50.), V2=(0., 100.), b_clcr=(0., 2.), b_wt=(0., 2.))
STARTS = {'bounded_start_1': T.STARTS['start_1'], 'bounded_start_2': T.STARTS['start_2']}
SEEDS = dict(bounded_start_1=260930121, bounded_start_2=260930122, uncertainty=260930221)


def spec(start):
    return ModelSpec('2cmt_iv', {n: P(start[n], lower=BOUNDS[n][0], upper=BOUNDS[n][1]) for n in ('CL', 'V1', 'Q', 'V2')},
                     {'CL': P(start['omega_CL'])}, P(start['sigma_prop']), P(T.ADD_SD, True),
                     (Covariate('CL', 'CLCR', 58., P(start['b_clcr'], lower=BOUNDS['b_clcr'][0], upper=BOUNDS['b_clcr'][1])),
                      Covariate('V1', 'WT', 62., P(start['b_wt'], lower=BOUNDS['b_wt'][0], upper=BOUNDS['b_wt'][1]))))


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    subs = T.subjects()
    (OUT / 'bounded_protocol.json').write_text(json.dumps(dict(model='published expert model with expert $THETA bounds',
        bounds=BOUNDS, starts=STARTS, seeds=SEEDS, fit_options=T.FIT_OPTIONS, uncertainty_options=T.CI_OPTIONS,
        primary='bounded_start_1'), indent=1), encoding='utf-8')
    names = sys.argv[1:] or list(STARTS)
    for name in names:
        path = OUT / f'{name}.json'
        if path.exists():
            continue
        fitted = fit(subs, spec(STARTS[name]), seed=SEEDS[name], **T.FIT_OPTIONS)
        print(name, fitted.status, round(fitted.ofv, 3), fitted.theta, fitted.coefficients, flush=True)
        if name == 'bounded_start_1' and fitted.converged:
            fitted.uncertainty(seed=SEEDS['uncertainty'], **T.CI_OPTIONS)
        path.write_text(json.dumps(fitted.to_dict(), indent=1, default=lambda x: x.tolist() if hasattr(x, 'tolist') else float(x)),
                        encoding='utf-8')


if __name__ == '__main__':
    main()
