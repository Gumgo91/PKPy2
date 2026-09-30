"""PK/PD with two outputs: one-compartment oral PK driving an indirect-response (turnover) model.

Plasma concentrations (DVID 1) and a response (DVID 2) are simulated for 30 subjects
from an indirect-response model with stimulation of production (type III), then the
declared model is fitted jointly with a residual model per output, and a VPC is
computed for each output. The ODE kernels are compiled on the first run, and the fit
takes a few minutes on a desktop computer.
"""
from pathlib import Path

import numpy as np

import pkpy2
from pkpy2 import Model, Residual, Parameter as P, structures as S

OUT = Path(__file__).resolve().parents[1] / "output"


def design(path, n=30):
    rows = ["ID,TIME,EVID,AMT,CMT,DV,DVID"]
    for i in range(1, n + 1):
        rows.append(f"{i},0,1,150,1,.,1")
        rows += [f"{i},{t},0,0,1,.,1" for t in (1, 3, 8, 24, 48)]           # concentrations
        rows += [f"{i},{t + .01},0,0,1,.,2" for t in (0, 12, 24, 48, 72, 96, 120)]   # response
    path.write_text("\n".join(rows))
    return pkpy2.read_nonmem(path)


def model(values, omega, residual):
    structure = S.indirect_response(3)          # outputs CP and R
    return Model(structure, theta={k: P(v) for k, v in values.items()},
                 omega={k: P(v) for k, v in omega.items()},
                 residual={"CP": Residual(proportional=P(residual[0])), "R": Residual(additive=P(residual[1]))})


def main():
    OUT.mkdir(exist_ok=True)
    data = design(OUT / "pkpd_design.csv")
    truth = model(dict(CL=2., V=25., Ka=1.3, R0=80., KOUT=.15, EMAX=1.5, EC50=1.),
                  dict(CL=.09, V=.04, KOUT=.09, EC50=.16), (.12, 4.))
    data = pkpy2.simulate_data(truth, data, seed=5)
    start = model(dict(CL=1.6, V=30., Ka=1., R0=70., KOUT=.1, EMAX=1., EC50=1.5),
                  dict(CL=.1, V=.1, KOUT=.1, EC50=.1), (.2, 6.))
    result = pkpy2.fit(data, start, seed=1)
    print(f"status={result.status} ofv={result.ofv:.3f} seconds={result.seconds:.0f}")
    print("theta:", {k: round(v, 3) for k, v in result.theta.items()})
    print("omega:", {k: round(v, 4) for k, v in result.omega.items()})
    print("sigma:", {k: round(v, 4) for k, v in result.sigma.items()})
    check = pkpy2.vpc(result, n=200)
    for output, entry in check.items():
        print(f"{output}: observed percentiles inside the 95% VPC intervals {entry['observed_within_interval']:.2f}")
    result.save(OUT / "pkpd_fit.json")


if __name__ == "__main__":
    main()
