"""Supplement sections S11-S12 (event-record interface), used by build_pkpy2_peerj_supplement.py."""
from pathlib import Path
import csv
import json

import numpy as np

from pkpy2_extended_raw_data import warfarin_nlmixr2

ROOT = Path(__file__).resolve().parents[1]
V = ROOT / 'output/pkpy2_extended_validation'
ODE = {'mm_iv_multiple', 'mm_oral_ss', 'idr1', 'idr2', 'idr3', 'idr4', 'tmdd_full', 'tmdd_qss'}
SCENARIO = {
    'iv1_infusion': 'One-compartment infusions', 'iv2_infusion_ss': 'Two-compartment infusion, steady state',
    'oral1_ss_lag_f': 'Oral, steady state, lag, F', 'oral2_addl_lag': 'Oral, additional doses, lag',
    'iv3_bolus_infusion': 'Three-compartment bolus and infusion', 'zero_order_d1': 'Zero-order absorption (D1)',
    'modeled_rate_r1': 'Modeled infusion rate (RATE = -1)', 'transit3': 'Three transit compartments',
    'reset_evid3_evid4': 'Resets (EVID 3 and 4)', 'timevarying_wt': 'Time-varying weight on CL',
    'effect_sigmoid_emax': 'Effect compartment, sigmoid Emax', 'parent_metabolite': 'Parent-metabolite',
    'ss2_superposition': 'Steady state with SS = 2', 'continuous_infusion_ss': 'Constant-infusion steady state',
    'infusion_bioavailability': 'Infusion with bioavailability', 'ss_infusion_lag_tail': 'Steady-state infusion past the interval',
    'mm_iv_multiple': 'Michaelis-Menten, repeated IV', 'mm_oral_ss': 'Michaelis-Menten, oral steady state',
    'idr1': 'Indirect response I', 'idr2': 'Indirect response II', 'idr3': 'Indirect response III',
    'idr4': 'Indirect response IV', 'tmdd_full': 'TMDD, full', 'tmdd_qss': 'TMDD, quasi-steady state'}
LIKELIHOOD = {'block_tv_categorical': 'Block Ω, time-varying and categorical covariates',
              'iov_blq_lognormal': 'IOV, M3 censoring, log-normal residual',
              'bioavailability_linear': 'Bioavailability (logit), linear covariate',
              'idr_two_outputs': 'Indirect response, two outputs'}
COMPARISON = {'infusion_block_covariates': 'Infusion, block Ω, WT and sex', 'oral_blq_m3': 'Oral, M3 censoring',
              'lognormal_residual': 'Log-normal residual', 'michaelis_menten': 'Michaelis-Menten'}


def read(path):
    return json.loads((V / path).read_text(encoding='utf-8'))


def table(headers, rows):
    return '\n'.join(['| ' + ' | '.join(headers) + ' |', '| ' + ' | '.join(['---'] * len(headers)) + ' |']
                     + ['| ' + ' | '.join(map(str, r)) + ' |' for r in rows])


def g(x, d=4):
    if x is None:
        return 'NR'
    return f'{x:.{d}g}'


def e(x):
    return f'{x:.1e}'


def one(x):
    """One decimal without a negative zero."""
    s = f'{x:.1f}'
    return '0.0' if s == '-0.0' else s


def interval(lo, hi):
    """Interval limits with four significant digits in positional notation."""
    fmt = lambda v: np.format_float_positional(v, precision=4, unique=False, fractional=False, trim='k').rstrip('.')
    return f'{fmt(lo)} to {fmt(hi)}'


def fixed_zero(key, value):
    """Residual components fixed at zero are not part of the fitted model."""
    return key.startswith('sigma:') and value == 0.


def label(key):
    group, _, name = key.partition(':')
    if group == 'theta':
        return name
    if group == 'omega':
        return f'ω²({name})'
    if group == 'omega_covariance':
        return f'ω({name})'
    if group == 'iov':
        return f'IOV({name})'
    if group == 'coefficients':
        return f'β({name})'
    if group == 'sigma':
        out, comp = name.split(':')
        return {'proportional': 'σ_prop', 'additive': 'σ_add', 'lognormal': 'σ_log'}[comp] + f' ({out})'
    return key


def flatten(d):
    out = {}
    for group in ('theta', 'coefficients', 'omega', 'omega_covariance', 'iov', 'sigma'):
        for k, v in d.get(group, {}).items():
            out[f'{group}:{k}'] = v
    return out


def markdown():
    pred = read('prediction_agreement.json')
    lik = read('likelihood_agreement.json')
    eng = read('general_vs_classic.json')
    diag = read('diagnostics_agreement.json')
    cmp = read('vs_nlmixr2.json')
    wofv = read('warfarin_pkpd/exact_ofv.json')
    wfit = read('warfarin_pkpd/pkpy2_fit.json')
    wdiag = read('warfarin_pkpd/pkpy2_diagnostics.json')
    rec = read('recovery_summary.json')
    cal = read('calibration_summary.json')
    tools = read('tools_summary.json')

    s18 = table(['Scenario', 'Model', 'Values', 'Max. relative difference'],
                [[SCENARIO.get(r['scenario'], r['scenario']), 'ODE' if r['scenario'] in ODE else 'Linear', r['values'],
                  e(r['max_relative_difference'])] for r in pred])
    s19a = table(['Model', 'Subjects', 'Observations', 'Censored', 'PKPy2 OFV (two banks)', 'Quadrature OFV', 'Difference'],
                 [[LIKELIHOOD[r['model']], r['subjects'], r['observations'], r['censored'],
                   ' / '.join(f'{v:.5f}' for v in r['pkpy2_ofv']), f"{r['reference_ofv']:.5f}", e(r['difference'])] for r in lik])
    s19b = table(['Dataset', 'Status', 'OFV difference', 'Max. estimate difference (%)'],
                 [[r['dataset'].replace('1cmt_iv__', 'Primary ').replace('__', ' ').capitalize(), r['general_status'],
                   f"{r['ofv_difference']:.5f}", f"{r['max_abs_relative_difference_pct']:.3f}"] for r in eng])
    s19c = table(['Quantity', 'n', 'Max. abs. difference', 'Correlation'],
                 [[k, diag[k]['n'], e(diag[k]['max_abs_difference']), f"{diag[k]['correlation']:.6f}"]
                  for k in ('PRED', 'IPRED', 'IWRES', 'CWRES', 'NPDE')]
                 + [['EBE (η)', diag['EBE']['n'], e(diag['EBE']['max_abs_difference']), '-'],
                    ['NPDE vs npde package (same replicates)', diag['NPDE_vs_npde_package_same_replicates']['n'],
                     e(diag['NPDE_vs_npde_package_same_replicates']['max_abs_difference']),
                     f"{diag['NPDE_vs_npde_package_same_replicates']['correlation']:.6f}"]])

    rows20 = []
    for r in cmp:
        a, b, t = flatten(r['pkpy2']), flatten(r.get('nlmixr2', {})), flatten(r['truth'])
        for k in a:
            if not fixed_zero(k, a[k]):
                rows20.append([COMPARISON[r['scenario']], label(k), g(t.get(k)), g(a[k]), g(b.get(k))])
        rows20.append([COMPARISON[r['scenario']], 'Exact OFV', '-', f"{r['pkpy2_ofv']:.3f}", f"{r['exact_ofv_at_nlmixr2']:.3f}"])
    s20 = table(['Dataset', 'Parameter', 'True', 'PKPy2', 'nlmixr2 FOCEi'], rows20)

    base = next(r['ofv'] for r in wofv if r['source'] == 'PKPy2')
    unc = {q['quantity']: q for q in (wdiag.get('uncertainty') or {}).get('intervals', [])}
    ustatus = (wdiag.get('uncertainty') or {}).get('status')
    fl = {k: v for k, v in flatten(wfit).items() if not fixed_zero(k, v)}
    focei, saem = warfarin_nlmixr2('focei'), warfarin_nlmixr2('saem')
    rows = []
    for k, v in fl.items():
        row = [label(k), g(v), g(focei.get(k)), g(saem.get(k))]
        if ustatus == 'computed':
            q = unc.get(k.replace('coefficients:', 'beta:').replace('omega_covariance', 'omega_cov'))
            row += ['-' if q is None or q.get('rse_pct') is None else f"{q['rse_pct']:.1f}",
                    '-' if q is None else interval(*q['interval'])]
        rows.append(row)
    head = ['Parameter', 'PKPy2', 'nlmixr2 FOCEi', 'nlmixr2 SAEM']
    s21 = table(head + (['PKPy2 RSE (%)', 'PKPy2 95% interval'] if ustatus == 'computed' else []), rows)
    s21_note = ('' if ustatus == 'computed' else
                'PKPy2 local intervals are not reported: IMAX approached 1 and the marginal Hessian was not positive definite. ')
    program = {'PKPy2': 'PKPy2', 'nlmixr2 focei': 'nlmixr2 FOCEi', 'nlmixr2 saem': 'nlmixr2 SAEM'}
    s21b = table(['Estimates', 'Exact OFV', 'ΔOFV vs PKPy2', 'Objective reported by the program'],
                 [[program.get(r['source'], r['source']), f"{r['ofv']:.3f}", f"{r['ofv'] - base:.3f}",
                   f"{wfit['ofv']:.3f}" if r['source'] == 'PKPy2' else f"{r['nlmixr2_objective']:.3f}"] for r in wofv])

    s22_rows = []
    for name, v in rec.items():
        for q in v['estimates']:
            s22_rows.append([{'complex_linear': 'Two-compartment infusion', 'pkpd_idr': 'Indirect-response PK/PD'}[name],
                             label(q['quantity']), g(q['truth']), q['n'], one(q['relative_bias_pct']),
                             one(q['relative_rmse_pct'])])
    s22 = table(['Design', 'Parameter', 'True', 'n', 'Relative bias (%)', 'Relative RMSE (%)'], s22_rows)
    s23 = table(['Design', 'Replicates', 'NPDE mean', 'NPDE variance', 'KS test p < 0.05', 'VPC percentiles within 95% interval'],
                [[{'complex_linear': 'Two-compartment infusion', 'pkpd_idr': 'Indirect-response PK/PD'}[k], v['replicates'],
                  f"{v['npde_mean']:.3f}", f"{v['npde_variance']:.3f}", f"{100 * v['npde_ks_rejections_5pct']:.0f}%",
                  '; '.join(f"{o} {100 * c['overall']:.1f}%" for o, c in v['vpc_coverage'].items())] for k, v in cal.items()])
    th = tools['theophylline']
    s24 = table(['Parameter', 'Estimate', 'Wald', 'Sandwich', 'Profile', 'Bootstrap', 'SIR'],
                [[label(r['quantity'].replace('beta:', 'coefficients:')), g(r['estimate']),
                  *[('-' if r.get(m) is None or any(v is None for v in r[m]) else interval(*r[m]))
                    for m in ('wald', 'sandwich', 'profile', 'bootstrap', 'sir')]] for r in th['intervals']])
    scm = tools['scm']
    s25 = table(['Candidate effect', 'True effect', 'Selected (%)'],
                [[k.split('[')[0], 'yes' if k in ('CL~CRCL[power]', 'V~WT[power]') else 'no', f"{100 * v:.0f}"]
                 for k, v in scm['selection_frequency'].items()])

    return f'''

## S11. Event-record interface: numerical methods

**Records and events.** Records are ordered by time within subject (data order at equal times) after additional doses (ADDL) are expanded at TIME + k·II. Dose records (EVID 1 and 4) are converted to events: a bolus at the record time plus the lag time, or an infusion start and end, with the amount multiplied by the bioavailability; for RATE > 0 the rate is kept and the duration changes (NONMEM convention), RATE = -1 takes the rate and RATE = -2 the duration from model parameters. EVID 3 resets all compartments, EVID 4 resets and doses. SS = 1 replaces the state by the steady state reached with the record's dose and interval before the dose is given, SS = 2 adds that steady state, and SS with II = 0 and RATE > 0 sets the steady state of a constant infusion, which then stops. Parameters and covariates change at their records (last observation carried forward).

**Linear systems.** The system matrix A(θ) is assembled from the declared transfer terms. Between events, x(t + Δ) = exp(AΔ) x(t) + J(Δ), where J(Δ) is the integral of exp(As) r over 0 ≤ s ≤ Δ for the active infusion rates r. When A has real eigenvalues separated by more than 10⁻⁶ in relative terms and an eigenvector basis with 1-norm condition number below 10⁶, propagation uses A = VΛV⁻¹, with the integral φ(λ, Δ) = (exp(λΔ) − 1)/λ per eigenvalue; otherwise the matrix exponential is computed by (6,6) Padé approximation with scaling and squaring [23], and the infusion integral from the exponential of an augmented block matrix. Steady states use closed forms, with u the unit vector of the dosed compartment: for a bolus of amount a with interval τ and lag L, x = (I − exp(Aτ))⁻¹ exp(A(τ − L)) u a; for an infusion of duration D, the state at the start of an infusion is z = (I − exp(Aτ))⁻¹ exp(A(τ − D)) J(D), including infusions that extend beyond the interval; for a constant infusion at rate ρ, x = −A⁻¹ u ρ.

**Nonlinear systems.** ODEs are integrated with the Dormand-Prince 5(4) method with first-same-as-last stages and adaptive steps (relative tolerance 10⁻⁸, absolute tolerance 10⁻¹⁰) [24]; when the step count exceeds 100,000, the interval is recomputed with a linearly implicit Rosenbrock method of order 2(3) and a finite-difference Jacobian [25]. Steady states are obtained by repeating the dosing interval until the relative change of the state is below 10⁻⁹ (at most 1,000 intervals). Linear PK parts of ODE models (indirect-response, Michaelis-Menten and TMDD models) use the same transfer terms as the linear library.

**Statistical model.** For parameter k of subject i at record r, ψᵢₖ(r) = g⁻¹(θ̃ₖ + Σ effects + ηᵢₖ + κᵢₖ,occ(r)), where g is the log, logit or identity function and θ̃ₖ = g(θₖ). Covariate effects are β log(z/c) (power), β(z − c) (exponential), log(1 + β(z − c)) (linear) and β·1[z = level] (categorical). Ω is parameterized by the Cholesky factor of each declared block (log diagonal) and interoccasion variances by log SDs. The combined residual variance is σ_prop² f² + σ_add²; for log-normal outputs log y ~ N(log f, σ²), and the OFV is evaluated on the log scale, as in log-transform-both-sides analyses. A censored observation contributes Φ((LLOQ − f)/SD) (on the log scale for log-normal outputs); CENS = −1 gives the upper tail and LIMIT the other end of an interval.

**Estimation.** Importance particles are drawn in the space of the transformed individual parameters, φᵢ = μᵢ + ηᵢ, so that typical values and time-invariant covariate effects of parameters with interindividual variability, variances, and residual SDs change only the importance weights; their score contributions are analytic. For the remaining coordinates (typical values without random effects and time-varying covariate effects), scores are central differences (step 10⁻⁴) of the objective with fixed particles; a time-varying effect on a parameter with random effects enters the typical value at the subject's mean covariate and the predictions through the within-subject deviation. The Laplace stage finds conditional modes by Newton steps with Fisher-type curvature H = JᵀWJ + Ω⁻¹ (J, prediction derivatives by central differences) and minimizes 2J(b*) + log det H − d log 2π with L-BFGS-B from the starting values and up to three scrambled-Sobol perturbations of the typical values and covariate coefficients (±log 3 and ±log 10), stopping when three starts agree within 0.1. Importance banks use 2ᵖ scrambled Sobol points per subject, half from the population distribution and half from a subject-specific Gaussian adapted twice to the weighted mean and 1.5 times the weighted covariance of a pilot sample of 2¹⁰ points. Refinement begins at p = 10 and takes quasi-Newton steps within a trust box, with curvature from central differences of the bank gradient updated by damped BFGS; each step is validated on an independent bank, and p increases up to 16 when the banks disagree or have fewer than 100 effective particles. The final assessment uses two independent banks with p = 14 (or the working p when higher). Acceptance requires their OFVs to agree within 0.05 and every subject ESS ≥ 100, as in S1, and each projected score to be at most 0.2 or, for a coordinate whose standard error from the quasi-Newton curvature, SE = √(2[H⁻¹]ⱼⱼ), is below one, at most 0.2/SE; such coordinates then lie within about 0.1 standard errors of the stationary point.

**Diagnostics and tools.** PRED uses η = 0; IPRED and IWRES use the conditional modes. WRES and CWRES use first-order linearizations at η = 0 and at the modes [32], decorrelated with the Cholesky factor of their covariance. NPDE use 1,000 simulated replicates of each subject's design, decorrelated with the Cholesky factor of the simulated covariance [33,34]. Shrinkage is 1 − SD(EBE)/ω for η and 1 − SD(IWRES) for ε [35]. VPC bins observations by nominal time (one bin per distinct time when there are at most twice as many distinct times as requested bins, otherwise by time quantiles) and compares the 5th, 50th and 95th observed percentiles with the 95% intervals of the same percentiles in 500 simulated replicates, optionally after prediction correction [36]; with a quantification limit, observed and simulated values below it are set to the limit. The sandwich covariance is I⁻¹ S I⁻¹ with I = H/2 and S the sum of outer products of the per-subject scores. Profile intervals are where the refitted OFV exceeds its minimum by 3.84, with linear interpolation on the log scale. The bootstrap resamples subjects with replacement and refits from the original estimates. SIR draws 1,000 vectors per iteration from a multivariate normal proposal centred at the estimate, starting from the local covariance and then using the covariance of the previous resample, and resamples 500 vectors per iteration [37,38]. Stepwise covariate selection adds the effect with the largest OFV decrease while p < 0.05 and removes effects while p ≥ 0.01 (likelihood-ratio tests, one degree of freedom) [39].

## S12. Evaluation of the event-record interface

**Table S18. PKPy2 predictions compared with rxode2.**

{s18}

rxode2 solved every scenario as an ODE system with relative and absolute tolerances of 10⁻¹² (including steady-state tolerances). Differences are divided by the rxode2 value, or by 10⁻⁶ times the subject's largest value when that is larger.

**Table S19. Likelihood, engine and diagnostic checks.**

(a) Marginal OFV at fixed parameters compared with independent adaptive Gauss-Hermite quadrature (PKPy2 banks with 2¹⁶ particles per subject).

{s19a}

(b) Refits through the event-record interface compared with the compact-interface fits (event-record minus compact).

{s19b}

(c) Diagnostics of the theophylline model at identical parameters compared with nlmixr2 (CWRES standardized by its own variance and NPDE with the upper Cholesky factor, as computed by nlmixr2) and with the npde package applied to the same simulated replicates.

{s19c}

**Table S20. Simulated datasets: true values and estimates of PKPy2 and nlmixr2 FOCEi.**

{s20}

Exact OFV: marginal OFV at each program's estimates evaluated by PKPy2 with two independent banks of 2¹⁴ particles per subject. Residual components fixed at zero are not shown.

**Table S21. Warfarin PK/PD: estimates of each program and exact OFV at these estimates.**

(a) Estimates of each program.

{s21}

(b) Exact marginal OFV at the estimates of each program, evaluated by PKPy2 as in Table S20.

{s21b}

{s21_note}The objectives reported by the programs use different approximations and constants. The model is the nlmixr2 example turnover model: transit (KTR) and first-order (Ka) absorption, CL and V, and an indirect response with inhibition of production by IMAX·C/(IC50 + C), baseline R0 and KOUT; IMAX is logit-transformed and all eight parameters have interindividual variability.

**Table S22. Parameter recovery with the event-record interface.**

{s22}

**Table S23. Calibration of NPDE and VPC at the true parameters.**

{s23}

KS, Kolmogorov-Smirnov test of the NPDE against N(0, 1).

**Table S24. Theophylline 95% intervals from five methods.**

{s24}

Profile likelihood was computed for the typical values and variances (-, not computed). Bootstrap: {th.get('bootstrap_converged')} of {th.get('bootstrap_requested')} replicates converged. SIR effective sample size in the final iteration: {th['sir_effective_samples']:.0f}.

**Table S25. Stepwise covariate selection in {scm['replicates']} simulated datasets.**

{s25}

Data: 60 subjects, one-compartment oral model, eight samples per subject; true effects on CL of (CRCL/80) to the power 0.7 and on V of (WT/70) to the power 1.
'''
