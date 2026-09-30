"""Event-record workflow on the theophylline data: fit, diagnostics and plots.

Reads the NONMEM-format theophylline data of the manuscript (paper/data/theophylline.csv), fits the
one-compartment oral model with clearance and volume scaled to 70 kg, and saves the goodness-of-fit,
individual-fit and VPC plots drawn by pkpy2.plots (install matplotlib with: python -m pip install .[plots]).
The fit takes one to two minutes on a desktop computer.

Usage: python examples/theophylline_diagnostics.py [output directory]
"""
from pathlib import Path
import sys

import pkpy2
from pkpy2 import Model, Residual, Covariate, Parameter as P, structures as S, plots

ROOT = Path(__file__).resolve().parents[1]


def model():
    return Model(
        S.pk(1, "first_order"),                         # depot (CMT 1) and central (CMT 2) compartments
        theta={"CL": P(3.), "V": P(30.), "Ka": P(1.)},  # CL and V of a 70 kg subject
        omega={"CL": P(.1), "V": P(.03), "Ka": P(.3)},
        covariates=(Covariate("CL", "WT", 70., P(1., fixed=True)),
                    Covariate("V", "WT", 70., P(1., fixed=True))),
        residual=Residual(proportional=P(.15)),
    )


def main(out, data_path=ROOT / "paper/data/theophylline.csv"):
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    data = pkpy2.read_nonmem(data_path, covariates=["WT"])
    result = pkpy2.fit(data, model(), seed=20260930)
    table, summary = pkpy2.diagnostics(result)       # PRED, IPRED, CWRES, NPDE, shrinkage
    check = pkpy2.vpc(result)
    plots.gof(table).savefig(out / "gof.png", dpi=300)
    plots.individual_fits(table).savefig(out / "individual_fits.png", dpi=300)
    plots.vpc(check).savefig(out / "vpc.png", dpi=300)
    print(result.status, f"OFV {result.ofv:.3f}", {k: round(v, 4) for k, v in result.theta.items()})
    return result, table, summary, check


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else ROOT / "output")
