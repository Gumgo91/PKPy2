# PKPy2

PKPy2 is a standalone Python package for explicitly specified nonlinear
mixed-effects pharmacokinetic (PopPK) models. It estimates structural
parameters, interindividual variability, residual error, and covariate effects
jointly by marginal likelihood, and adds analytical repeated-dose prediction,
independent numerical convergence checks, and local confidence intervals.

It is the successor to [PKPy](https://doi.org/10.7717/peerj.20258), replacing
summaries of separately fitted individual parameters with joint population
inference under a declared model.

$$\log p_{ik} = \log \theta_k + X_{ik}\,\beta_k + \eta_{ik}, \qquad \eta_i \sim \mathcal{N}(0, \Omega)$$

$$y_{ij} \mid \eta_i \sim \mathcal{N}\left(f_{ij},\; \sigma_{\mathrm{prop}}^{2}\, f_{ij}^{2} + \sigma_{\mathrm{add}}^{2}\right)$$

$$L = \prod_i \int \left[\, \prod_j p\!\left(y_{ij} \mid \eta_i\right) \right] p(\eta_i)\; d\eta_i, \qquad \mathrm{OFV} = -2 \log L$$

## Features

- **Joint marginal-likelihood estimation** of population parameters,
  interindividual variances, residual error, and covariate effects. Staged
  procedure: marginal-likelihood refinement, SAEM when required, then
  importance-sampling refinement.
- **Explicit model interface** — structural model, fixed vs. estimated
  parameters, random effects, residual error, and power covariates are
  declared through a common API. No data-dependent model selection occurs
  inside `fit`.
- **Analytical predictions and sensitivities** for one- and two-compartment
  models with IV bolus or first-order oral input, deterministic absorption
  lag, and repeated dosing by linear superposition. Dose-state recurrence
  accelerates repeated-dose evaluation (up to ~900x at 1,000 dose events in
  the accompanying evaluation).
- **Independent convergence checks** — final acceptance requires agreement
  between two independently generated quasi-Monte Carlo integration banks
  (OFV difference <= 0.05, maximum absolute projected score <= 0.2,
  effective sample size >= 100 per subject).
- **Local uncertainty** — full marginal-likelihood Hessian from two
  independent integration banks and two finite-difference step sizes;
  Wald intervals in logarithmic coordinates, transformed back to the
  reporting scale.
- **Auditable results** — every fit records its estimation path, termination
  messages, and numerical phases; fits can be saved to and reloaded from a
  portable JSON format that verifies the analysis data.

## Installation

Python 3.13 is required.

```powershell
git clone https://github.com/Gumgo91/PKPy2.git
cd PKPy2
python -m pip install .
```

Dependencies: NumPy >= 2.2, SciPy >= 1.16, Numba >= 0.65, threadpoolctl >= 3.6.

## Quick start

```python
import numpy as np
from pkpy2 import Subject, ModelSpec, Parameter as P, fit

# Each subject supplies observation times, concentrations, and a dose.
subjects = [
    Subject(1, np.array([1., 4., 12.]), np.array([2.1, 1.6, .75]), dose=100.),
    Subject(2, np.array([1., 4., 12.]), np.array([2.3, 1.4, .68]), dose=100.),
]
spec = ModelSpec(
    "1cmt_iv",
    theta={"CL": P(4.), "V": P(40.)},
    omega={"CL": P(.09), "V": P(.04)},
    sigma_prop=P(.15),
)
result = fit(subjects, spec, seed=1, workers=2)
print(result.theta, result.omega, result.status)
# This tiny example illustrates syntax, not a sufficiently powered analysis.
if result.converged:
    report = result.uncertainty(workers=2)
```

Runnable scripts are in [`examples/`](examples/):

- `examples/quickstart.py` — simulated one-compartment IV fit with intervals.
- `examples/repeated_dosing.py` — analytical prediction over 60 dose events.
- `examples/theophylline.py` — population fit of `data/theo.csv` matching the
  manuscript specification (one-compartment oral, fixed WT exponents on CL and
  V at a 70 kg reference, IIV on CL/V/Ka, proportional residual error).

## Statistical contract

- `theta`: population medians, before power covariate effects.
- `omega`: independent lognormal random-effect **variances**, not SD or CV.
- `sigma_prop`, `sigma_add`: residual SDs in the Gaussian variance
  `(sigma_prop * prediction)**2 + sigma_add**2`.
- `Parameter(value, fixed=True)` fixes a quantity exactly and excludes it from
  optimization, parameter counts, and information matrices.
- `Covariate("CL", "WT", 70., P(.75))` estimates the exponent in
  `CL_i = CL * (WT_i / 70)**beta`; a fixed coefficient is also supported.
  Power covariates must be positive and observed for every subject.
- Oral clearances and volumes are apparent quantities (CL/F, V/F) when
  bioavailability is not independently specified. Lag is fixed at zero with
  `P(0., True)`.
- A nonempty `dose_history=[(time, amount), ...]` is authoritative. Otherwise
  `dose` is administered at time zero. Infusion rates/durations are not part
  of this interface.
- At least one random effect and one free parameter are required.

## Supported scope

The public model set comprises `1cmt_iv`, `1cmt_oral`, `2cmt_iv`, and
`2cmt_oral`, with optional fixed or estimated oral lag and repeated doses.
Concentrations and natural-parameter sensitivities use analytic expressions.
The first three models also use event recurrence for ordered repeated doses;
two-compartment oral dosing uses direct analytic superposition.

The current estimator supports diagonal interindividual-variability structure
and observed positive power covariates. Correlated random effects, infusion,
arbitrary ODE models, censored observations, automatic model selection, and
clinical dosing decisions are outside this API.

Convergence is decided by the independent numerical checks above. A returned
point estimate may carry an unresolved status and is not relabeled as success.
Uncertainty may remain unresolved even after a fit converges; local Wald
intervals are conditional on the declared model and do not establish
structural identifiability or guarantee small-sample coverage.
`result.estimation` records termination messages and numerical phases,
including for unresolved fits. `result.save("fit.json")` and
`load_fit("fit.json", subjects)` support portable result storage; loading
verifies the supplied analysis data.

## Evaluation summary

The accompanying manuscript evaluated this implementation as follows:

- Analytical predictions agreed with independent matrix-exponential and
  DOP853 calculations under 48 conditions (maximum scaled discrepancy
  4.10e-10 for concentrations, 3.41e-8 for sensitivities). Marginal-likelihood
  calculations agreed with an independent Gaussian-quadrature implementation
  (maximum same-point OFV difference 1.93e-6; maximum relative difference in
  coordinate standard errors 0.0008%).
- All 200 primary simulation fits converged (100 rich- and 100
  sparse-sampling one-compartment IV datasets). Under sparse sampling, the
  relative RMSE of the interindividual variance in volume was 32.46% with
  PKPy2 vs. 51.66% with the original PKPy fitting components.
- Empirical coverage of nominal 95% local intervals ranged from 90% to 100%.
- Theophylline estimates differed by at most 3.39% from archived expert
  NONMEM results under an equivalent statistical model; five freely estimated
  warfarin parameters differed by at most 4.58% from published values.
- With 1,000 dose events, recurrence-based prediction calls were 440-906x
  faster than direct summation.

`validation/numerical_validation.py` reproduces the independent prediction,
sensitivity, and gradient checks (requires only the installed package, NumPy,
and SciPy). The full simulation, NONMEM-comparison, and benchmark code used
for the manuscript is described in the article and its supplement.

## Repository layout

```text
src/pkpy2/            package source (public API in api.py, numerics in _numerics/)
examples/             runnable example scripts
data/                 public benchmark CSVs used in the manuscript (see data/README.md)
validation/           independent numerical-check script
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
