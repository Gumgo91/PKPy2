"""The classic engine is unchanged by the general-engine additions.

Refits primary simulation datasets with the recorded protocol and seeds and
compares OFV and estimates with the stored v3 records (expected: identical).
Output: output/pkpy2_extended_validation/classic_unchanged.json
"""
from pathlib import Path
import json
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'packages/pkpy2/src'))
sys.path.insert(0, str(ROOT / 'experiments'))
from pkpy2 import fit                                  # noqa: E402
import pkpy2_paper_experiments as E                    # noqa: E402

D = ROOT / 'output/pkpy2_development/confirmatory_v3'


def main(keys):
    proto = json.loads((D / 'protocol.json').read_text())
    rows = []
    for key in keys:
        task = next(t for t in proto['tasks'] if f"{t['model']}__{t['sampling']}__{t['replicate']:03d}" == key)
        subjects, spec, truth, eta = E.generate(task['model'], task['sampling'], task['seed'])
        old = json.loads((D / f'{key}.json').read_text())['pkpy2']
        t0 = time.perf_counter()
        r = fit(subjects, spec, seed=task['seed'] + 500000, **proto['fit_options'])
        diffs = [abs(r.theta[k] / old['theta'][k] - 1) for k in r.theta]
        diffs += [abs(r.omega[k] / old['omega'][k] - 1) for k in r.omega]
        diffs += [abs(r.sigma[k] / old['sigma'][k] - 1) for k in r.sigma if old['sigma'][k] > 0]
        row = dict(dataset=key, status=r.status, recorded_status=old['status'], ofv=float(r.ofv),
                   recorded_ofv=float(old['ofv']), ofv_difference=float(r.ofv - old['ofv']),
                   max_relative_estimate_difference=float(max(diffs)), seconds=time.perf_counter() - t0)
        rows.append(row)
        print(json.dumps(row), flush=True)
    out = ROOT / 'output/pkpy2_extended_validation/classic_unchanged.json'
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(rows, indent=1))


if __name__ == '__main__':
    main(sys.argv[1:] or ['1cmt_iv__rich__000', '1cmt_iv__sparse__000', '1cmt_iv__rich__001', '1cmt_iv__sparse__001'])
