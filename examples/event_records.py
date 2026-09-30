"""Event-record workflow: NONMEM-format data, a two-compartment infusion model, diagnostics and a VPC.

A small study is simulated with pkpy2.simulate_data: 24 subjects on a steady-state
infusion regimen (occasion 1) and a single infusion after washout (occasion 2), with
body weight changing between occasions and a sex effect on the central volume.
Concentrations below 0.4 mg/L are censored (M3). The declared model is then fitted
from perturbed starting values, and diagnostics, a VPC and intervals are reported.
The fit and the interval calculation take a few minutes on a desktop computer.
"""
from pathlib import Path

import numpy as np

import pkpy2
from pkpy2 import Model, Residual, Covariate, Parameter as P, structures as S

OUT = Path(__file__).resolve().parents[1] / "output"


def design(path, n=24, seed=3):
    rng = np.random.default_rng(seed)
    rows = ["ID,TIME,EVID,AMT,CMT,RATE,II,SS,DV,OCC,WT,SEX"]
    for i in range(1, n + 1):
        wt1 = rng.uniform(50, 100)
        wt2 = wt1 * rng.uniform(.85, 1.15)
        sex = int(rng.random() < .5)
        rows.append(f"{i},0,1,500,1,250,12,1,.,1,{wt1:.1f},{sex}")          # steady state, 250 mg/h for 2 h every 12 h
        rows += [f"{i},{t},0,0,1,0,0,0,.,1,{wt1:.1f},{sex}" for t in (.5, 1, 2, 4, 6, 9, 11.9)]
        rows.append(f"{i},168,3,0,1,0,0,0,.,2,{wt2:.1f},{sex}")               # washout: reset
        rows.append(f"{i},168,1,800,1,400,0,0,.,2,{wt2:.1f},{sex}")           # single 2-h infusion
        rows += [f"{i},{168 + t},0,0,1,0,0,0,.,2,{wt2:.1f},{sex}" for t in (.5, 1, 2, 4, 8, 12, 24, 36)]
    path.write_text("\n".join(rows))
    return pkpy2.read_nonmem(path, covariates=["WT", "SEX"], occasion="OCC")


def model(theta, omega, covariance, iov, beta, residual):
    return Model(S.pk(2), theta={k: P(v) for k, v in theta.items()},
                 omega={k: P(v) for k, v in omega.items()}, omega_blocks=[("CL", "V1")],
                 omega_covariance={("CL", "V1"): P(covariance)}, iov={"CL": P(iov)},
                 covariates=(Covariate("CL", "WT", 70., P(beta[0])),
                             Covariate("V1", "SEX", 0, P(beta[1]), "categorical", 1)),
                 residual=Residual(proportional=P(residual[0]), additive=P(residual[1])))


def main():
    OUT.mkdir(exist_ok=True)
    data = design(OUT / "event_records_design.csv")
    truth = model(dict(CL=5., V1=20., Q=8., V2=40.), dict(CL=.09, V1=.06), .04, .03, (.75, -.2), (.12, .05))
    data = pkpy2.simulate_data(truth, data, seed=11, lloq={"CP": .4})
    start = model(dict(CL=4., V1=25., Q=6., V2=50.), dict(CL=.1, V1=.1), 0., .05, (.5, 0.), (.2, .1))
    result = pkpy2.fit(data, start, seed=1)
    print(f"status={result.status} ofv={result.ofv:.3f} seconds={result.seconds:.0f}")
    print("theta:", {k: round(v, 3) for k, v in result.theta.items()})
    print("omega:", {k: round(v, 4) for k, v in result.omega.items()}, "covariance:", result.omega_covariance)
    print("iov:", result.iov, "coefficients:", result.coefficients)
    table, summary = pkpy2.diagnostics(result)
    print("shrinkage:", summary["eta_shrinkage"], "NPDE mean/variance:",
          round(summary["npde_mean"], 3), round(summary["npde_variance"], 3))
    check = pkpy2.vpc(result, n=200, lloq=.4)
    print("observed percentiles inside the 95% VPC intervals:", check["CP"]["observed_within_interval"])
    if result.converged:
        report = result.uncertainty(power=12)      # 2^12 particles per subject keeps this example short
        for row in report["intervals"]:
            print(f"  {row['quantity']:>24s} {row['estimate']:.4g}  95% CI {row['interval'][0]:.4g}-{row['interval'][1]:.4g}")
    result.save(OUT / "event_records_fit.json")


if __name__ == "__main__":
    main()
