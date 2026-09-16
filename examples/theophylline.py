"""Theophylline population analysis matching the model in the PKPy2 manuscript.

Reads ``data/theo.csv`` (12 subjects, first-order oral dosing), fits a
one-compartment oral model with fixed allometric weight scaling of clearance
and volume (exponent 1 at a 70 kg reference), interindividual variability on
CL, V and Ka, and a proportional residual error model, then reports local
95% confidence intervals.

The dose column is mg/kg and is multiplied by subject weight to obtain mg;
bioavailability is fixed to 1, so CL and V are apparent (CL/F, V/F).
This takes on the order of a few minutes on a desktop CPU.
"""
import csv
from pathlib import Path

import numpy as np

from pkpy2 import Subject, ModelSpec, Parameter as P, Covariate, fit

ROOT = Path(__file__).resolve().parents[1]

FIT_OPTIONS = dict(
    workers=2,
    saem_options=dict(cpu_budget_seconds=60.),
    refinement_options=dict(cpu_budget_seconds=300., max_stages=90,
                            analytic_non_eta=True),
)
CI_OPTIONS = dict(power=16, step=0.002, workers=2, confidence=0.95)


def load_subjects(path):
    rows = list(csv.DictReader(path.open(encoding="utf-8-sig", newline="")))
    subjects = []
    for sid in sorted({int(r["ID"]) for r in rows}):
        records = [r for r in rows if int(r["ID"]) == sid]
        wt = float(records[0]["WT"])
        dose_history = [(float(r["TIME"]), float(r["AMT"]) * wt)
                        for r in records if float(r["AMT"]) > 0]
        observed = [r for r in records if r["DV"] and float(r["TIME"]) > 0]
        subjects.append(Subject(
            sid,
            np.array([float(r["TIME"]) for r in observed]),
            np.array([float(r["DV"]) for r in observed]),
            dose=0.,
            covariates={"WT": wt},
            dose_history=dose_history,
        ))
    return subjects


def main():
    subjects = load_subjects(ROOT / "data" / "theo.csv")
    spec = ModelSpec(
        "1cmt_oral",
        theta={"CL": P(3.), "V": P(30.), "Ka": P(1.), "ALAG": P(0., True)},
        omega={"CL": P(0.1), "V": P(0.03), "Ka": P(0.3)},
        sigma_prop=P(0.15),
        sigma_add=P(0., True),
        covariates=tuple(
            Covariate(n, "WT", 70., P(1., True)) for n in ("CL", "V")),
    )
    result = fit(subjects, spec, seed=260920001, **FIT_OPTIONS)
    print(f"status={result.status} converged={result.converged} "
          f"ofv={result.ofv:.3f} seconds={result.seconds:.1f}")
    print("theta:", {k: round(v, 4) for k, v in result.theta.items()})
    print("omega:", {k: round(v, 4) for k, v in result.omega.items()})
    print("sigma:", {k: round(v, 4) for k, v in result.sigma.items()})
    if result.converged:
        report = result.uncertainty(seed=260920002, **CI_OPTIONS)
        print(f"uncertainty status={report['status']}")
        for row in report["information"]["data"]["intervals"]:
            interval = row["interval"]
            bounds = (f"[{interval[0]:.4g}, {interval[1]:.4g}]"
                      if interval else "unresolved")
            print(f"  {row['coordinate']:>22s} estimate={row['estimate']:.4g} "
                  f"95% CI {bounds}")


if __name__ == "__main__":
    main()
