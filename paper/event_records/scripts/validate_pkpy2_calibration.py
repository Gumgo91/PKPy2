"""Calibration of the diagnostics at the true parameter values (simulated data).

Datasets are simulated from each scenario's true model, which is then evaluated
at its true values without estimation (pkpy2.evaluate). Under the true model the
NPDE are standard normal, and each observed VPC percentile lies within the 95%
interval of its simulated distribution with probability 0.95.

Usage: python validate_pkpy2_calibration.py <scenario> <replicates>
       python validate_pkpy2_calibration.py summarize
"""
from pathlib import Path
import json
import sys
import time
import zlib

import numpy as np
from scipy import stats

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'packages/pkpy2/src'))
sys.path.insert(0, str(ROOT / 'scripts'))
import pkpy2                                                           # noqa: E402
from validate_pkpy2_extended_recovery import SCENARIOS                 # noqa: E402

OUT = ROOT / 'output/pkpy2_extended_validation/calibration'


def run(name, replicates, descending=False):
    sc = SCENARIOS[name]
    out = OUT / name
    out.mkdir(parents=True, exist_ok=True)
    for r in (reversed(range(replicates)) if descending else range(replicates)):
        path = out / f'rep_{r:03d}.json'
        if path.exists():
            continue
        t0 = time.time()
        rng = np.random.default_rng(zlib.crc32(f'calibration:{name}:{r}'.encode()))
        rows, opts = sc['design'](rng)
        dpath = out / 'design.csv'
        dpath.write_text('\n'.join(rows))
        design = pkpy2.read_nonmem(dpath, **opts)
        data = pkpy2.simulate_data(sc['truth'](), design, seed=zlib.crc32(f'calibration:{name}:{r}:data'.encode()),
                                   lloq=sc['lloq'])
        result = pkpy2.evaluate(data, sc['truth'](), power=12, seed=r)
        table, summary = pkpy2.diagnostics(result, npde_samples=1000, seed=1000 + r)
        npde = table['NPDE'][np.isfinite(table['NPDE'])]
        cwres = table['CWRES'][np.isfinite(table['CWRES'])]
        check = pkpy2.vpc(result, n=500, bins=6, lloq=sc['lloq'], seed=2000 + r)
        record = dict(replicate=r, ofv=result.ofv, audit_passed=result.audit['passed'],
                      censored=int(np.sum(table['CENS'] == 1)), n_npde=int(len(npde)),
                      npde_mean=float(npde.mean()), npde_variance=float(npde.var(ddof=1)),
                      npde_ks_p=float(stats.kstest(npde, 'norm').pvalue),
                      cwres_mean=float(cwres.mean()), cwres_variance=float(cwres.var(ddof=1)),
                      vpc_within={k: v['observed_within_interval'] for k, v in check.items()},
                      vpc_inside={k: ((v['observed'] >= v['lower']) & (v['observed'] <= v['upper'])).tolist()
                                  for k, v in check.items()},
                      eta_shrinkage=summary['eta_shrinkage'], seconds=time.time() - t0)
        path.write_text(json.dumps(record, indent=1))
        print(name, r, round(record['npde_mean'], 3), round(record['npde_variance'], 3),
              {k: round(v, 3) for k, v in record['vpc_within'].items()}, round(record['seconds'], 1), flush=True)


def summarize():
    report = {}
    for name in SCENARIOS:
        recs = [json.loads(p.read_text()) for p in sorted((OUT / name).glob('rep_*.json'))]
        if not recs:
            continue
        coverage = {}
        for k in recs[0]['vpc_inside']:
            inside = np.array([r['vpc_inside'][k] for r in recs], dtype=float)
            coverage[k] = dict(overall=float(np.nanmean(inside)),
                               by_quantile=np.nanmean(inside, axis=(0, 1)).tolist())
        report[name] = dict(
            replicates=len(recs), audit_passed=sum(r['audit_passed'] for r in recs),
            npde_mean=float(np.mean([r['npde_mean'] for r in recs])),
            npde_variance=float(np.mean([r['npde_variance'] for r in recs])),
            npde_ks_rejections_5pct=float(np.mean([r['npde_ks_p'] < .05 for r in recs])),
            cwres_mean=float(np.mean([r['cwres_mean'] for r in recs])),
            cwres_variance=float(np.mean([r['cwres_variance'] for r in recs])),
            vpc_coverage=coverage, median_censored=float(np.median([r['censored'] for r in recs])),
            median_seconds=float(np.median([r['seconds'] for r in recs])))
    (OUT.parent / 'calibration_summary.json').write_text(json.dumps(report, indent=1))
    print(json.dumps(report, indent=1))


if __name__ == '__main__':
    if sys.argv[1] == 'summarize':
        summarize()
    else:
        run(sys.argv[1], int(sys.argv[2]), descending='--descending' in sys.argv)
