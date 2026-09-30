"""Run the package example examples/theophylline_diagnostics.py on the manuscript data and keep its outputs.

The example reads the NONMEM-format theophylline file exported for the software comparison, fits the
theophylline model through the event-record interface, and draws its diagnostic plots with pkpy2.plots
(Supplementary Figure S1, Listing 2). The fit is compared with the compact-interface analysis.

Outputs in output/pkpy2_extended_validation/example_theophylline/: gof.png, individual_fits.png, vpc.png,
diagnostics.csv (Supplementary Figure S1 data) and summary.json.
"""
from pathlib import Path
import csv
import importlib.util
import json
import sys

import numpy as np
import matplotlib
matplotlib.use('Agg')

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'packages/pkpy2/src'))
EXAMPLE = ROOT / 'packages/pkpy2/examples/theophylline_diagnostics.py'
DATA = ROOT / 'output/pkpy2_software_comparison/data/theophylline.csv'
COMPACT = ROOT / 'output/pkpy2_development/theophylline_v3/proportional.json'
OUT = ROOT / 'output/pkpy2_extended_validation/example_theophylline'
COLUMNS = ['ID', 'TIME', 'DV', 'CENS', 'PRED', 'IPRED', 'IWRES', 'CWRES', 'NPDE']


def main():
    spec = importlib.util.spec_from_file_location('theophylline_diagnostics', EXAMPLE)
    example = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(example)
    result, table, summary, check = example.main(OUT, DATA)
    with open(OUT / 'diagnostics.csv', 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(COLUMNS)
        for row in zip(*(np.asarray(table[c]) for c in COLUMNS)):
            w.writerow([int(v) if c in ('ID', 'CENS') else float(v) for c, v in zip(COLUMNS, row)])
    compact = json.loads(COMPACT.read_text(encoding='utf-8'))
    record = dict(status=result.status, ofv=result.ofv, compact_ofv=compact['ofv'], ofv_difference=result.ofv - compact['ofv'],
                  theta=result.theta, omega=result.omega, sigma=result.sigma,
                  compact_theta=compact['theta'], compact_omega=compact['omega'],
                  max_relative_difference_pct=max(100 * abs(result.theta[k] / compact['theta'][k] - 1)
                                                  for k in ('CL', 'V', 'Ka')),
                  eta_shrinkage=summary['eta_shrinkage'], vpc_within={k: v['observed_within_interval'] for k, v in check.items()})
    (OUT / 'summary.json').write_text(json.dumps(record, indent=1, default=float), encoding='utf-8')
    print(json.dumps(record, indent=1, default=float))
    assert result.status == 'converged' and abs(record['ofv_difference']) < .05, record


if __name__ == '__main__':
    main()
