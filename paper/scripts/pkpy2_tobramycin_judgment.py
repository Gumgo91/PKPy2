"""Tobramycin with pharmacological judgment encoded as fixed values and bounds.

Pre-specified before fitting:
  1. V1 proportional to body weight (WT exponent fixed at 1).
  2. Peripheral volume V2 <= 30 L (about 0.5 L/kg at 62 kg; aminoglycosides distribute
     mainly in extracellular fluid).
  3. The remaining expert $THETA bounds: CL (0, 20) L/h, V1 (0, 100) L, Q (0, 50) L/h,
     CLCR exponent (0, 2).
Starts: documented initial values (judgment_start_1, primary; V2 start 20 L is within the
bound) and the published expert estimates (judgment_start_2).
"""
from pathlib import Path
import json
import sys

from pkpy2 import ModelSpec, Parameter as P, Covariate, fit

sys.path.insert(0, str(Path(__file__).resolve().parent))
import pkpy2_tobramycin_application as T  # noqa: E402

OUT = T.ROOT / 'output/pkpy2_tobramycin_v3'
BOUNDS = dict(CL=(0., 20.), V1=(0., 100.), Q=(0., 50.), V2=(0., 30.), b_clcr=(0., 2.))
STARTS = {'judgment_start_1': T.STARTS['start_1'], 'judgment_start_2': T.STARTS['start_2']}
SEEDS = dict(judgment_start_1=260930131, judgment_start_2=260930132, uncertainty=260930231)


def spec(start):
    return ModelSpec('2cmt_iv', {n: P(start[n], lower=BOUNDS[n][0], upper=BOUNDS[n][1]) for n in ('CL', 'V1', 'Q', 'V2')},
                     {'CL': P(start['omega_CL'])}, P(start['sigma_prop']), P(T.ADD_SD, True),
                     (Covariate('CL', 'CLCR', 58., P(start['b_clcr'], lower=BOUNDS['b_clcr'][0], upper=BOUNDS['b_clcr'][1])),
                      Covariate('V1', 'WT', 62., P(1.0, True))))


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    subs = T.subjects()
    (OUT / 'judgment_protocol.json').write_text(json.dumps(dict(
        model='published expert model; WT exponent on V1 fixed at 1; V2 <= 30 L; other expert bounds',
        bounds=BOUNDS, starts=STARTS, seeds=SEEDS, fit_options=T.FIT_OPTIONS, uncertainty_options=T.CI_OPTIONS,
        primary='judgment_start_1'), indent=1), encoding='utf-8')
    for name in sys.argv[1:] or list(STARTS):
        path = OUT / f'{name}.json'
        if path.exists():
            continue
        fitted = fit(subs, spec(STARTS[name]), seed=SEEDS[name], **T.FIT_OPTIONS)
        print(name, fitted.status, round(fitted.ofv, 3), fitted.theta, fitted.coefficients, flush=True)
        if name == 'judgment_start_1' and fitted.converged:
            fitted.uncertainty(seed=SEEDS['uncertainty'], **T.CI_OPTIONS)
        path.write_text(json.dumps(fitted.to_dict(), indent=1, default=lambda x: x.tolist() if hasattr(x, 'tolist') else float(x)),
                        encoding='utf-8')


if __name__ == '__main__':
    main()
