"""Minimal PKPy2 workflow on a small simulated one-compartment IV dataset.

Generates 20 subjects, fits the declared population model by marginal
likelihood, and requests local 95% confidence intervals. The dataset is small
on purpose so the script finishes quickly; it illustrates the interface, not
a powered analysis.
"""
from pathlib import Path

import numpy as np

from pkpy2 import Subject, ModelSpec, Parameter as P, fit


def simulate(seed=1, n_subjects=20):
    rng = np.random.default_rng(seed)
    times = np.array([0.25, 0.5, 1., 2., 4., 8., 12., 24.])
    subjects = []
    for i in range(1, n_subjects + 1):
        cl = 4. * np.exp(rng.normal(0., 0.3))
        v = 40. * np.exp(rng.normal(0., 0.2))
        true = 100. / v * np.exp(-cl / v * times)
        obs = true * np.exp(rng.normal(0., 0.15, len(times)))
        subjects.append(Subject(i, times, obs, dose=100.))
    return subjects


def main():
    subjects = simulate()
    spec = ModelSpec(
        "1cmt_iv",
        theta={"CL": P(3.2), "V": P(32.)},
        omega={"CL": P(0.08), "V": P(0.08)},
        sigma_prop=P(0.20),
    )
    result = fit(subjects, spec, seed=1, workers=2)
    print(f"status={result.status} converged={result.converged} "
          f"ofv={result.ofv:.3f} seconds={result.seconds:.1f}")
    print("theta:", {k: round(v, 4) for k, v in result.theta.items()})
    print("omega:", {k: round(v, 4) for k, v in result.omega.items()})
    print("sigma:", {k: round(v, 4) for k, v in result.sigma.items()})
    if result.converged:
        report = result.uncertainty(workers=2)
        print(f"uncertainty status={report['status']}")
        for row in report["information"]["data"]["intervals"]:
            interval = row["interval"]
            bounds = (f"[{interval[0]:.4g}, {interval[1]:.4g}]"
                      if interval else "unresolved")
            print(f"  {row['coordinate']:>22s} estimate={row['estimate']:.4g} "
                  f"95% CI {bounds}")
        out = Path(__file__).resolve().parents[1] / "output"
        out.mkdir(exist_ok=True)
        result.save(out / "quickstart_fit.json")
        print(f"saved {out / 'quickstart_fit.json'}")


if __name__ == "__main__":
    main()
