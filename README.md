# PKPy2

PKPy2 is a standalone Python package for explicitly specified nonlinear
mixed-effects pharmacokinetic (PopPK) and pharmacokinetic-pharmacodynamic
(PK/PD) models. It estimates structural parameters, interindividual and
interoccasion variability, residual error, and covariate effects jointly by
marginal likelihood, checks convergence with independent numerical audits, and
provides model diagnostics and interval estimates. Fixed values and parameter
bounds let the analyst encode pharmacological judgment directly in the model
declaration.

It is the successor to [PKPy](https://doi.org/10.7717/peerj.20258), replacing
summaries of separately fitted individual parameters with joint population
inference under a declared model.

$$\psi_{ik}(t) = g_k^{-1}\!\left(\tilde\theta_k + \textstyle\sum_c f_{kc}(z_{ic}(t)) + \eta_{ik} + \kappa_{ik,\mathrm{occ}(t)}\right), \qquad \eta_i \sim \mathcal{N}(0, \Omega)$$

$$L = \prod_i \int \left[\, \prod_j p\!\left(y_{ij} \mid \eta_i, \kappa_i\right) \right] p(\eta_i)\, p(\kappa_i)\; d\eta_i\, d\kappa_i, \qquad \mathrm{OFV} = -2 \log L$$

## Features

- **Two model interfaces, one estimator.** `ModelSpec` declares the four
  closed-form one- and two-compartment models (analytical sensitivities and
  dose-state recurrence). `Model` declares event-record models read from
  NONMEM-format data (see below). Both use the same marginal likelihood,
  integration and convergence audit.
- **Event records** — EVID 0-4, AMT, CMT, RATE (including modeled rate and
  duration), II, SS 1/2 (including constant-infusion steady state), ADDL, MDV,
  DVID and censoring (CENS/LIMIT, M3/M4).
- **Structural library** — one- to three-compartment disposition; bolus,
  infusion, first-order, zero-order and transit absorption; bioavailability and
  lag; effect compartments with linear or Emax-type responses; parent-metabolite;
  Michaelis-Menten; target-mediated drug disposition (full, QSS); indirect
  response I-IV; user-defined linear systems and ODEs.
- **Statistical model** — log, logit or identity parameter scales; power,
  exponential, linear and categorical covariate effects, including time-varying
  covariates; block Ω and interoccasion variability; additive, proportional,
  combined or log-normal residual error for each output.
- **Joint marginal-likelihood estimation** — multi-start Laplace exploration,
  then importance-sampling refinement with adaptive proposals and quasi-Newton
  steps validated on independent integration banks.
- **Independent convergence audit** — two independently generated quasi-Monte
  Carlo banks must agree on the OFV (difference <= 0.05) with at least 100
  effective particles per subject, and projected scores must be stationary.
- **Diagnostics** — empirical Bayes estimates, PRED/IPRED, WRES, CWRES, IWRES,
  NPDE, shrinkage, simulation, VPC (prediction-corrected, censored), plots.
- **Uncertainty and model building** — Wald and sandwich intervals, profile
  likelihood, bootstrap, SIR, likelihood-ratio tests, information criteria and
  stepwise covariate selection; `evaluate` computes the OFV and diagnostics of
  a model at fixed values (no estimation).

## Installation

Python 3.13 is required.

```powershell
git clone https://github.com/Gumgo91/PKPy2.git
cd PKPy2
python -m pip install .            # add [plots] for matplotlib-based plots
```

Dependencies: NumPy >= 2.2, SciPy >= 1.16, Numba >= 0.65, threadpoolctl >= 3.6.
The first call of each model class compiles its numerical kernels (about a
minute); compiled kernels are cached.

## Quick start (compact interface)

```python
import numpy as np
from pkpy2 import Subject, ModelSpec, Parameter as P, fit

subjects = [
    Subject(1, np.array([1., 4., 12.]), np.array([2.1, 1.6, .75]), dose=100.),
    Subject(2, np.array([1., 4., 12.]), np.array([2.3, 1.4, .68]), dose=100.),
]
spec = ModelSpec("1cmt_iv", theta={"CL": P(4.), "V": P(40.)},
                 omega={"CL": P(.09), "V": P(.04)}, sigma_prop=P(.15))
result = fit(subjects, spec, seed=1, workers=2)
print(result.theta, result.omega, result.status)
```

## Event-record models

```python
import pkpy2
from pkpy2 import Model, Residual, Covariate, Parameter as P, structures as S

data = pkpy2.read_nonmem("study.csv", covariates=["WT", "SEX"], occasion="OCC")
model = Model(
    S.pk(2),                                   # dosing follows CMT, AMT, RATE, II, SS, ADDL
    theta={"CL": P(5.), "V1": P(20.), "Q": P(8.), "V2": P(40.)},
    omega={"CL": P(.09), "V1": P(.06)}, omega_blocks=[("CL", "V1")],
    iov={"CL": P(.03)},
    covariates=(Covariate("CL", "WT", 70., P(.75)),
                Covariate("V1", "SEX", 0, P(-.2), "categorical", 1)),
    residual=Residual(proportional=P(.12), additive=P(.05)),
)
result = pkpy2.fit(data, model, seed=1)
table, summary = pkpy2.diagnostics(result)     # PRED, IPRED, CWRES, NPDE, shrinkage
check = pkpy2.vpc(result, lloq=.4)             # censored VPC
report = result.uncertainty()                  # Wald and sandwich intervals
```

PK/PD models have one residual model per output (`DVID` 1, 2, ...):

```python
model = Model(S.indirect_response(3), theta=..., omega=...,
              residual={"CP": Residual(proportional=P(.12)), "R": Residual(additive=P(4.))})
```

A user-defined ODE is written as plain Python that Numba can compile:

```python
def rhs(t, x, p, dx):              # p: parameters in declared order
    dx[0] = -p[0] * x[0]
    dx[1] = p[0] * x[0] - p[1] / p[2] * x[1]

structure = S.ode(["KA", "CL", "V"], ["depot", "central"], [("CP", "central", "V")], rhs)
```

Inference tools: `pkpy2.profile`, `pkpy2.bootstrap`, `pkpy2.sir`,
`pkpy2.robust_covariance`, `pkpy2.likelihood_ratio_test`, `pkpy2.compare`,
`pkpy2.stepwise_covariates`, `pkpy2.simulate`, `pkpy2.simulate_data`,
`pkpy2.evaluate`; plots in `pkpy2.plots` (`gof`, `vpc`, `individual_fits`).

Runnable scripts are in [`examples/`](examples/):

- `examples/quickstart.py` — simulated one-compartment IV fit with intervals.
- `examples/repeated_dosing.py` — analytical prediction over 60 dose events.
- `examples/theophylline.py` — population fit of `data/theo.csv` as in the manuscript.
- `examples/event_records.py` — steady-state infusions, block Ω, IOV, covariates and BLQ data; diagnostics, VPC, intervals.
- `examples/pkpd_turnover.py` — indirect-response PK/PD model with two outputs and VPCs.

## Statistical contract

- `theta`: typical values (population medians for log-scale parameters)
  before covariate effects.
- `omega`: random-effect **variances** on the parameter's scale; block
  covariances in `omega_covariance`, interoccasion variances in `iov`.
- Residual SDs: combined variance `(proportional * f)**2 + additive**2`;
  log-normal outputs use `log y ~ N(log f, sd**2)`.
- `Parameter(value, fixed=True)` fixes a quantity exactly and excludes it from
  optimization, parameter counts, and information matrices;
  `Parameter(value, lower=..., upper=...)` bounds an estimated quantity on the
  reporting scale.
- `Covariate("CL", "WT", 70., P(.75))` is a power effect `(WT/70)**beta`;
  forms `"exponential"`, `"linear"` and `"categorical"` (with a level) are
  also available.
- Convergence is decided by the independent audit. A returned point estimate
  may carry an unresolved status and is not relabeled as success; intervals
  are conditional on the declared model.

## Evaluation summary

The accompanying manuscript evaluated this implementation as follows.

*Compact interface (one- and two-compartment models)*

- Analytical predictions agreed with independent matrix-exponential and DOP853 calculations under 48 conditions (maximum scaled discrepancy 4.1e-10 for concentrations, 3.4e-08 for sensitivities); marginal likelihoods agreed with an independent Gaussian-quadrature implementation (maximum same-point OFV difference 4.3e-06).
- All 200 primary simulation fits converged. Under sparse sampling, the relative RMSE of the interindividual variance in volume was 32.49% with PKPy2 vs. 51.66% with the original PKPy fitting components; recovery was similar to that of nlmixr2 (FOCEi, SAEM) and saemix on the same datasets.
- Empirical coverage of nominal 95% local intervals ranged from 90% to 100%.
- Theophylline and warfarin estimates differed by at most 3.37% and 4.56% from the published expert NONMEM estimates. For tobramycin, with the expert's pharmacological judgment declared as a fixed weight exponent and volume bounds, PKPy2 reproduced the expert estimates within 6.0%.

*Event-record interface*

- Predictions agreed with rxode2 in 24 dosing and model scenarios (maximum relative difference 2.3e-11 for linear systems, 1.1e-06 for ODE models).
- Marginal OFVs of four models with block Ω, interoccasion variability, time-varying and categorical covariates, M3 censoring, log-normal residuals and two outputs agreed with independent adaptive Gauss-Hermite quadrature within 1.3e-04; refits of 12 compact-interface analyses agreed within 0.0012 OFV units.
- PRED, IPRED, IWRES and CWRES agreed with nlmixr2, and NPDE with the npde package (4.9e-15 on identical replicates).
- On four simulated datasets, the exact OFV at the nlmixr2 FOCEi estimates was 0.01 to 2.59 units above that at the PKPy2 estimates; for the warfarin PK/PD turnover model, 334.3 (FOCEi) and 17.9 (SAEM) units above.
- In recovery simulations (20/20 and 20/20 fits converged), relative bias was within ±3.1% for typical values; at the true parameters NPDE had variance 1.02 and 1.01, and 95% and 94% of observed VPC percentiles lay within their 95% intervals.
- Stepwise covariate selection retained both true effects in 19 of 20 simulated datasets.

`validation/numerical_validation.py` and `validation/event_record_checks.py` reproduce independent checks with only the installed package, NumPy and SciPy (both run in continuous integration). [`paper/`](paper/) contains the analysis data, scripts, fit records and raw data of the manuscript (see [paper/README.md](paper/README.md)).

## Repository layout

```text
src/pkpy2/            package source (compact interface in api.py and _numerics/, event-record engine in _general/)
examples/             runnable example scripts
data/                 public benchmark CSVs used in the manuscript (see data/README.md)
validation/           independent numerical checks (run in continuous integration)
paper/                manuscript materials: comparisons, analyses, raw data and validation of the event-record interface
```

## Citation

If you use PKPy2, please cite the software and the associated manuscript:

> Kong H, Kim I. *PKPy2: A Python framework for joint population
> pharmacokinetic estimation and uncertainty assessment.* Manuscript
> submitted for publication, 2026.

The scientific predecessor:

> Kong H, Kim I, Zhang B-T. *PKPy: a Python-based framework for automated
> population pharmacokinetic analysis.* PeerJ. 2025;13:e20258.
> https://doi.org/10.7717/peerj.20258

See [CITATION.cff](CITATION.cff) for machine-readable citation metadata.

## License

MIT — see [LICENSE](LICENSE). Benchmark CSVs under `data/` are third-party
data; the software license does not grant rights to them (see
[data/README.md](data/README.md)).
