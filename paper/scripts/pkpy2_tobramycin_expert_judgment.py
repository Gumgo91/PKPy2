"""Tobramycin with the expert's modeling judgment encoded as fixed values and bounds (fidelity test).

Pre-specified before fitting:
  1. V1 proportional to body weight (WT exponent fixed at 1).
  2. A small central compartment with a distinct distribution phase: V1 <= 10 L.
  3. Peripheral volume V2 <= 30 L.
  4. The remaining expert $THETA bounds: CL (0, 20) L/h, Q (0, 50) L/h, CLCR exponent (0, 2).
Starts: documented initial values with V1 = 8 L to lie within the constraint (expert_start_1,
primary) and the published expert estimates (expert_start_2).
"""
from pathlib import Path
import json
import sys

from pkpy2 import ModelSpec, Parameter as P, Covariate, fit

sys.path.insert(0, str(Path(__file__).resolve().parent))
import pkpy2_tobramycin_application as T  # noqa: E402

OUT = T.ROOT / 'output/pkpy2_tobramycin_v3'
BOUNDS = dict(CL=(0., 20.), V1=(0., 10.), Q=(0., 50.), V2=(0., 30.), b_clcr=(0., 2.))
STARTS = {'expert_start_1': dict(T.STARTS['start_1'], V1=8.0), 'expert_start_2': T.STARTS['start_2']}
SEEDS = dict(expert_start_1=260930141, expert_start_2=260930142, uncertainty=260930241)


def spec(start):
    return ModelSpec('2cmt_iv', {n: P(start[n], lower=BOUNDS[n][0], upper=BOUNDS[n][1]) for n in ('CL', 'V1', 'Q', 'V2')},
                     {'CL': P(start['omega_CL'])}, P(start['sigma_prop']), P(T.ADD_SD, True),
                     (Covariate('CL', 'CLCR', 58., P(start['b_clcr'], lower=BOUNDS['b_clcr'][0], upper=BOUNDS['b_clcr'][1])),
                      Covariate('V1', 'WT', 62., P(1.0, True))))


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    subs = T.subjects()
    (OUT / 'expert_judgment_protocol.json').write_text(json.dumps(dict(
        model='published expert model with expert judgment: WT exponent on V1 fixed at 1; V1 <= 10 L; V2 <= 30 L; other expert bounds',
        bounds=BOUNDS, starts=STARTS, seeds=SEEDS, fit_options=T.FIT_OPTIONS, uncertainty_options=T.CI_OPTIONS,
        primary='expert_start_1'), indent=1), encoding='utf-8')
    for name in sys.argv[1:] or list(STARTS):
        path = OUT / f'{name}.json'
        if path.exists():
            continue
        fitted = fit(subs, spec(STARTS[name]), seed=SEEDS[name], **T.FIT_OPTIONS)
        print(name, fitted.status, round(fitted.ofv, 3), fitted.theta, fitted.coefficients, flush=True)
        if name == 'expert_start_1' and fitted.converged:
            fitted.uncertainty(seed=SEEDS['uncertainty'], **T.CI_OPTIONS)
        path.write_text(json.dumps(fitted.to_dict(), indent=1, default=lambda x: x.tolist() if hasattr(x, 'tolist') else float(x)),
                        encoding='utf-8')


if __name__ == '__main__':
    main()
