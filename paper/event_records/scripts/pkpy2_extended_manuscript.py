"""Manuscript text on the event-record interface (added by revise_pkpy2_peerj_manuscript.py).

All numbers come from docs/pkpy2_paper/extended_numbers.json (scripts/pkpy2_extended_numbers.py).
Citations use the PeerJ reference numbers; references 23-40 are listed in NEW_REFERENCES and
renumbered in order of first citation by build_pkpy2_jpkpd_submission.py.
"""
from pathlib import Path
import json
import os

ROOT = Path(__file__).resolve().parents[1]
E = json.loads(Path(os.environ.get('PKPY2_EXTENDED_NUMBERS', ROOT / 'docs/pkpy2_paper/extended_numbers.json'))
               .read_text(encoding='utf-8'))
SUP = str.maketrans('-0123456789', '⁻⁰¹²³⁴⁵⁶⁷⁸⁹')

NEW_REFERENCES = {
    23: 'Moler C, Van Loan C. Nineteen dubious ways to compute the exponential of a matrix, twenty-five years later. SIAM Review. '
        '2003;45(1):3-49. doi:10.1137/S00361445024180.',
    24: 'Dormand JR, Prince PJ. A family of embedded Runge-Kutta formulae. Journal of Computational and Applied Mathematics. '
        '1980;6(1):19-26. doi:10.1016/0771-050X(80)90013-3.',
    25: 'Shampine LF, Reichelt MW. The MATLAB ODE suite. SIAM Journal on Scientific Computing. 1997;18(1):1-22. '
        'doi:10.1137/S1064827594276424.',
    26: 'Savic RM, Jonker DM, Kerbusch T, Karlsson MO. Implementation of a transit compartment model for describing drug absorption '
        'in pharmacokinetic studies. Journal of Pharmacokinetics and Pharmacodynamics. 2007;34(5):711-726. doi:10.1007/s10928-007-9066-0.',
    27: 'Dayneka NL, Garg V, Jusko WJ. Comparison of four basic models of indirect pharmacodynamic responses. Journal of '
        'Pharmacokinetics and Biopharmaceutics. 1993;21(4):457-478. doi:10.1007/BF01061691.',
    28: 'Mager DE, Jusko WJ. General pharmacokinetic model for drugs exhibiting target-mediated drug disposition. Journal of '
        'Pharmacokinetics and Pharmacodynamics. 2001;28(6):507-532. doi:10.1023/A:1014414520282.',
    29: 'Gibiansky L, Gibiansky E, Kakkar T, Ma P. Approximations of the target-mediated drug disposition model and identifiability '
        'of model parameters. Journal of Pharmacokinetics and Pharmacodynamics. 2008;35(5):573-591. doi:10.1007/s10928-008-9102-8.',
    30: 'Karlsson MO, Sheiner LB. The importance of modeling interoccasion variability in population pharmacokinetic analyses. '
        'Journal of Pharmacokinetics and Biopharmaceutics. 1993;21(6):735-750. doi:10.1007/BF01113502.',
    31: 'Beal SL. Ways to fit a PK model with some data below the quantification limit. Journal of Pharmacokinetics and '
        'Pharmacodynamics. 2001;28(5):481-504. doi:10.1023/A:1012299115260.',
    32: 'Hooker AC, Staatz CE, Karlsson MO. Conditional weighted residuals (CWRES): a model diagnostic for the FOCE method. '
        'Pharmaceutical Research. 2007;24(12):2187-2197. doi:10.1007/s11095-007-9361-x.',
    33: 'Brendel K, Comets E, Laffont C, Laveille C, Mentré F. Metrics for external model evaluation with an application to the '
        'population pharmacokinetics of gliclazide. Pharmaceutical Research. 2006;23(9):2036-2049. doi:10.1007/s11095-006-9067-5.',
    34: 'Comets E, Brendel K, Mentré F. Computing normalised prediction distribution errors to evaluate nonlinear mixed-effect '
        'models: the npde add-on package for R. Computer Methods and Programs in Biomedicine. 2008;90(2):154-166. '
        'doi:10.1016/j.cmpb.2007.12.002.',
    35: 'Savic RM, Karlsson MO. Importance of shrinkage in empirical Bayes estimates for diagnostics: problems and solutions. '
        'The AAPS Journal. 2009;11(3):558-569. doi:10.1208/s12248-009-9133-0.',
    36: 'Bergstrand M, Hooker AC, Wallin JE, Karlsson MO. Prediction-corrected visual predictive checks for diagnosing nonlinear '
        'mixed-effects models. The AAPS Journal. 2011;13(2):143-151. doi:10.1208/s12248-011-9255-z.',
    37: 'Dosne AG, Bergstrand M, Harling K, Karlsson MO. Improving the estimation of parameter uncertainty distributions in '
        'nonlinear mixed effects models using sampling importance resampling. Journal of Pharmacokinetics and Pharmacodynamics. '
        '2016;43(6):583-596. doi:10.1007/s10928-016-9487-8.',
    38: 'Dosne AG, Bergstrand M, Karlsson MO. An automated sampling importance resampling procedure for estimating parameter '
        'uncertainty. Journal of Pharmacokinetics and Pharmacodynamics. 2017;44(6):509-520. doi:10.1007/s10928-017-9542-0.',
    39: 'Jonsson EN, Karlsson MO. Automated covariate model building within NONMEM. Pharmaceutical Research. 1998;15(9):1463-1468. '
        'doi:10.1023/A:1011970125687.',
    40: "O'Reilly RA, Aggeler PM. Studies on coumarin anticoagulant drugs: initiation of warfarin therapy without a loading dose. "
        'Circulation. 1968;38(1):169-177. doi:10.1161/01.CIR.38.1.169.',
}


def sci(x):
    m, e = f'{x:.1e}'.split('e')
    return f'{m} × 10{str(int(e)).translate(SUP)}'


def pct(x, d=1):
    return f'{x:.{d}f}'


WORDS = {0: 'none', 1: 'one', 2: 'two', 3: 'three', 4: 'four', 5: 'five', 6: 'six', 7: 'seven', 8: 'eight', 9: 'nine',
         10: 'ten', 11: 'eleven', 12: 'twelve', 13: 'thirteen', 14: 'fourteen', 15: 'fifteen', 16: 'sixteen',
         17: 'seventeen', 18: 'eighteen', 19: 'nineteen', 20: 'twenty'}


def abstract():
    return (
        'Population pharmacokinetic (PopPK) analysis usually relies on specialized, often commercial, software. PKPy introduced a '
        'Python workflow, but it derived population parameters from separately fitted individual models, so individual estimation '
        'error entered the estimated interindividual variability. We developed PKPy2, a standalone Python package that estimates '
        'structural parameters, interindividual variability, residual error, and covariate effects jointly by marginal likelihood, '
        'with optional fixed values and bounds, numerical convergence checks, and confidence intervals. An event-record interface '
        'extends estimation to NONMEM-format dosing records, nonlinear and pharmacokinetic-pharmacodynamic (PK/PD) models, correlated '
        'and interoccasion random effects, time-varying covariates, and censored observations, with standard diagnostics and '
        'uncertainty tools. In 200 simulated one-compartment intravenous datasets, all PKPy2 fits converged, and under sparse sampling '
        'the relative root mean squared error of the interindividual variance of volume was {sv}%, compared with {pv}% for PKPy. '
        'Recovery was similar to that of nlmixr2 and saemix on the same datasets, and coverage of nominal 95% intervals ranged from '
        '{cmin}% to {cmax}%. Predictions, likelihoods, and diagnostics of the '
        'event-record interface agreed with rxode2, independent quadrature, and nlmixr2. In clinical theophylline and warfarin data, '
        'PKPy2 estimates differed from published expert NONMEM estimates by at most {tm}% and {wm}%. In tobramycin data, the published '
        "model had a different maximum-likelihood solution; when the expert's pharmacological judgment was entered as fixed values and "
        'bounds, PKPy2 reproduced the expert estimates within {tob}%. PKPy2 provides joint PopPK and PK/PD estimation, diagnostics, and '
        'uncertainty assessment in Python.')


def extend(d, N, find, set_text, insert_after, blank_after, replace_in):
    """Apply the event-record sections to the revised manuscript document d (N: main manuscript numbers)."""
    set_text(find(d, 'Population pharmacokinetic (PopPK) analysis usually relies'),
             abstract().format(sv=N['sparse_rmse_pkpy2_omega_V'], pv=N['sparse_rmse_pkpy_omega_V'], tm=N['theo_max_diff'],
                               wm=N['warf_max_diff'], tob=N['tob_ej_max'], cmin=N['coverage_min'], cmax=N['coverage_max']))

    # ------------------------------------------------------------ Introduction
    intro = find(d, 'We developed PKPy2 to extend PKPy from separate individual fits')
    replace_in(intro, 'Figure 1 summarizes the transition from individual fits to joint population inference.',
               'An event-record interface reads NONMEM-format analysis data and extends the model set to multi-compartment, '
               'nonlinear, and pharmacokinetic-pharmacodynamic (PK/PD) models with correlated and interoccasion random effects, '
               'time-varying covariates, and censored observations, together with model diagnostics and tools for uncertainty '
               'assessment and covariate selection. Figure 1 summarizes the transition from individual fits to joint population '
               'inference.')
    replace_in(find(d, 'We evaluated the numerical calculations against independent implementations and examined'),
               'All evaluations used prespecified pharmacokinetic models.',
               'The event-record interface was verified against rxode2, an independent quadrature implementation, nlmixr2, and the '
               'npde package, and its estimation was evaluated in simulated datasets and a warfarin PK/PD analysis. All evaluations '
               'used prespecified models.')

    # ------------------------------------------------------------ Methods
    replace_in(find(d, 'where θₖ is the typical population value at the covariate reference'),
               'The evaluated implementation uses a diagonal Ω.',
               'Ω is diagonal in the compact interface used for the primary evaluation; the event-record interface described below '
               'also allows correlated blocks.')
    replace_in(find(d, 'Normal-density constants were retained.'),
               'The evaluated model set comprised one- and two-compartment structures with intravenous bolus or first-order oral input, '
               'deterministic absorption lag, explicit dose histories, and diagonal random effects.',
               'The compact interface comprises one- and two-compartment structures with intravenous bolus or first-order oral input, '
               'deterministic absorption lag, explicit dose histories, and diagonal random effects.')
    heading = find(d, 'Local uncertainty assessment')
    body = find(d, 'Wald intervals for positive parameters were calculated in logarithmic coordinates')
    last = find(d, 'Intervals were reported when the information matrix was positive definite')
    h = insert_after(blank_after(last), 'Event-record interface', like=heading)
    p = insert_after(h,
        'Models beyond the four closed-form structures are declared through an event-record interface. Analysis data follow NONMEM '
        'conventions: each record carries the subject, time, event type, dose amount and compartment, infusion rate or a modeled rate '
        'or duration, steady-state flag and dosing interval, additional doses, the observed output, and an optional censoring flag '
        '[12,13]. A structural library provides one- to three-compartment disposition with intravenous bolus or infusion and '
        'first-order, zero-order, or transit-compartment absorption [26], bioavailability and lag time, effect-compartment models with '
        'linear or Emax-type responses, parent-metabolite models, Michaelis-Menten elimination, target-mediated drug disposition in '
        'full and quasi-steady-state forms [28,29], and indirect-response models I-IV [27]; linear compartmental systems and ordinary '
        'differential equations (ODEs) can also be declared directly. Linear systems were advanced exactly between events through the '
        'eigendecomposition of the system matrix or, when the matrix was not diagonalizable, the matrix exponential [23], and their '
        'steady states were obtained in closed form. Nonlinear systems were integrated with an adaptive Dormand-Prince method [24] or, '
        'for stiff systems, a Rosenbrock method [25], and their steady states were obtained by repeating the dosing interval to '
        'convergence.', like=body)
    p = insert_after(p,
        'Individual parameters were defined on the log, logit, or identity scale as the typical value plus covariate and random '
        'effects. Covariate effects could be power, exponential, linear, or categorical and could change within a subject; random '
        'effects could be correlated within declared blocks of Ω and combined with interoccasion variability [30]. Each observed output '
        'had an additive, proportional, combined, or log-normal residual model, the last corresponding to a log-transform-both-sides '
        'analysis, and observations below the quantification limit contributed the probability of censoring (M3 method) [31]. '
        'Estimation followed the procedure described above. A Laplace approximation at the conditional modes was minimized from '
        'several starts, and the importance-sampled marginal likelihood was then refined in stages by quasi-Newton steps validated on '
        'independent integration banks. Each subject\'s importance proposal was adapted to the weighted mean and covariance of a pilot '
        'sample. Convergence required the two-bank OFV agreement and effective sample sizes of the compact interface, and a projected '
        'score of at most 0.2 in each coordinate, or 0.2 divided by the local standard error for coordinates with standard errors below '
        'one, which places the estimate within about 0.1 standard errors of the stationary point in these coordinates (Supplementary '
        'Section S11).', like=body)
    h = insert_after(blank_after(p), 'Diagnostics, uncertainty tools, and covariate selection', like=heading)
    insert_after(h,
        'Both interfaces share the diagnostic and inference tools. PKPy2 reports conditional modes (empirical Bayes estimates) and '
        'conditional means of the random effects, population and individual predictions, weighted, conditional weighted [32], and '
        'individual weighted residuals, normalized prediction distribution errors (NPDE) [33,34], η- and ε-shrinkage [35], simulations '
        'from the fitted model, and visual predictive checks (VPC) with optional prediction correction and censoring [36]. In addition '
        'to local Wald intervals, parameter uncertainty can be assessed with a sandwich covariance, profile likelihood, a nonparametric '
        'case bootstrap [20], and sampling importance resampling (SIR) with iterative proposal updates [37,38]. Nested models are '
        'compared by likelihood-ratio tests, and covariate effects can be selected by stepwise forward inclusion and backward '
        'elimination [39].', like=body)

    comp_head = find(d, 'Comparison with established open-source NLME software')
    comp_last = find(d, 'Simulation estimates were summarized with the metrics used in the primary evaluation.')
    h = insert_after(blank_after(comp_last), 'Evaluation of the event-record interface', like=comp_head)
    m3 = next(r for r in E['vs_nlmixr2'] if r['scenario'] == 'oral_blq_m3')
    p = insert_after(h,
        f"Predictions were compared with rxode2 [9] in {E['pred_scenarios']} scenarios covering infusions, steady state with bolus, "
        'infusion, and constant-infusion dosing, additional doses, lag time and bioavailability, modeled rates and durations, resets, '
        'time-varying covariates, three-compartment, transit, effect-compartment, and parent-metabolite models, and the '
        'Michaelis-Menten, indirect-response, and target-mediated disposition models. Marginal OFVs of four models combining block Ω, '
        'interoccasion variability, time-varying and categorical covariates, M3 censoring, a log-normal residual model, bioavailability, '
        'and two outputs were compared at fixed parameters with an independent adaptive Gauss-Hermite quadrature implementation [3,18]. '
        'The theophylline and warfarin analyses and ten primary simulation datasets were refitted through the event-record interface for '
        'comparison with the compact interface. Diagnostics of the theophylline model were compared with nlmixr2 and the npde package at '
        'identical parameter values.', like=comp_last)
    insert_after(p,
        'Estimation was compared with nlmixr2 FOCEi in four simulated datasets: a two-compartment infusion model at steady state with '
        f"block Ω and effects of body weight and sex (60 subjects); a one-compartment oral model with {m3['censored']} of "
        f"{m3['observations']} concentrations below the quantification limit (50 subjects); a one-compartment model with log-normal "
        'residual error; and Michaelis-Menten elimination (40 subjects each). The warfarin turnover model of the nlmixr2 examples, with '
        'transit absorption, inhibition of the production of prothrombin complex activity (PCA), and eight random effects, was fitted '
        'to the concentrations and PCA of 32 subjects [15,40] with PKPy2 and with nlmixr2 FOCEi and SAEM. The exact marginal OFV of '
        "each program's estimates was evaluated with PKPy2 importance sampling. Parameter recovery was assessed in "
        f"{E['recovery']['complex_linear']['replicates']} datasets each of a two-compartment steady-state infusion model with block Ω, "
        'interoccasion variability, time-varying weight, sex, and BLQ data (40 subjects) and of an indirect-response PK/PD model with '
        'two outputs (40 subjects), and the calibration of NPDE and VPC in '
        f"{E['calibration']['complex_linear']['replicates']} and {E['calibration']['pkpd_idr']['replicates']} datasets of the same "
        'designs evaluated at the true parameters. The interval methods were compared on the theophylline model, and stepwise '
        f"covariate selection was applied to {E['tools']['scm']['replicates']} simulated datasets with two true and five null covariate "
        'effects (Supplementary Section S12).', like=comp_last)

    # ------------------------------------------------------------ Results
    rhead = find(d, 'Comparison with established software in the primary simulation')
    rlast = find(d, 'Coverage of nominal 95% intervals for CL and V was')
    h = insert_after(blank_after(rlast), 'Event-record interface', like=rhead)
    diag_max = max(E['diag_ipred_max'], E['diag_iwres_max'], E['diag_cwres_max'])
    p = insert_after(h,
        f"Predictions agreed with rxode2 in all {E['pred_scenarios']} scenarios, with maximum relative differences of "
        f"{sci(E['pred_linear_max'])} for linear systems and {sci(E['pred_ode_max'])} for ODE models (Figure 6A). The four marginal "
        f"OFVs differed from the independent quadrature values by at most {sci(E['lik_max_abs_diff'])}. All "
        f"{WORDS[E['engine_datasets']]} refits through the event-record interface converged, with OFVs within "
        f"{E['engine_max_abs_ofv_diff']:.4f} and estimates within {E['engine_max_param_pct']:.2f}% of the compact-interface fits "
        f"(Figure 6B). At identical parameters, PRED agreed with nlmixr2 to {sci(E['diag_pred_max'])}, and IPRED, IWRES, and CWRES "
        f"agreed to within {diag_max:.4f}; NPDE computed from the same simulated replicates agreed with the npde package to "
        f"{sci(E['diag_npde_pkg_max'])} (Figure 6C).", like=rlast)
    p = insert_after(p, results_estimation(), like=rlast)
    insert_after(p, results_recovery(), like=rlast)
    replace_in(find(d, 'Prediction-call times after compilation were similar'), '(Figure 6)', '(Supplementary Table S10)')

    # ------------------------------------------------------------ Discussion
    rec = find(d, 'The analytical recurrence for repeated dosing provided a substantial computational advantage')
    insert_after(rec, discussion(), like=rec)
    lim = find(d, 'The evaluation used 30-subject datasets, diagonal random effects')
    set_text(lim,
        'The primary evaluation used 30-subject datasets, diagonal random effects, specified linear compartmental models, and ten '
        'replicates for each additional structure, and the event-record interface was evaluated with '
        f"{E['recovery']['complex_linear']['replicates']} replicates per design, datasets of up to 60 subjects, and one PK/PD "
        'application. The NONMEM comparisons used the expert estimates as reported, and nlmixr2 and saemix were run with default '
        'settings. The tobramycin constraints were formulated with knowledge of the expert solution; the data alone supported a '
        'different solution. Larger simulation studies across population sizes, sparse PK/PD designs, and stiff systems will further '
        'characterize precision and computing time for complex models.')
    over = find(d, 'Overall, PKPy2 provides a unified Python workflow')
    set_text(over,
        'Overall, PKPy2 provides a unified Python workflow for explicit PopPK and PK/PD model specification with fixed values and '
        'bounds, NONMEM-format event records, joint estimation, numerical convergence assessment, model diagnostics, and uncertainty '
        'quantification. Its improved recovery of interindividual variability under sparse sampling, close agreement with independent '
        'calculations, expert NONMEM estimates, rxode2, and nlmixr2, and efficient handling of long dose histories demonstrate the '
        'practical value of nonlinear mixed-effects analysis in a standalone Python environment.')

    # ------------------------------------------------------------ Conclusions and availability
    concl = find(d, 'PKPy2 extends PKPy from separate individual fits to joint nonlinear')
    replace_in(concl, 'and analytical dose-state propagation accelerated prediction for long dose histories.',
               'and analytical dose-state propagation accelerated prediction for long dose histories. An event-record interface extends '
               'estimation, diagnostics, and uncertainty assessment to multi-compartment, nonlinear, and PK/PD models with correlated '
               'and interoccasion random effects and censored data, and its calculations agreed with rxode2, an independent quadrature '
               'implementation, nlmixr2, and the npde package.')
    set_text(find(d, 'PKPy2 provides an integrated Python environment for explicit PopPK model specification'),
             'PKPy2 provides an integrated Python environment for explicit PopPK and PK/PD model specification, joint estimation of '
             'structural parameters and variability, numerical convergence assessment, model diagnostics, and uncertainty '
             'quantification. The framework supports reproducible population analysis with a transparent model interface and a '
             'standalone estimation engine.')
    replace_in(find(d, 'PKPy2, installation instructions, runnable examples, public benchmark data'),
               'and the numerical validation script are available', 'the numerical validation script, and the validation scripts of the '
               'event-record interface are available')

    # ------------------------------------------------------------ References 23-40 (PeerJ numbering)
    anchor = find(d, '[22] R Core Team.')
    for k in sorted(NEW_REFERENCES):
        anchor = insert_after(anchor, f'[{k}] {NEW_REFERENCES[k]}')


def results_estimation():
    rows = {r['scenario']: r for r in E['vs_nlmixr2']}
    diffs = [r['exact_ofv_difference'] for r in rows.values()]
    w = E['warfarin_pkpd']
    converged = all(r['status'] == 'converged' for r in rows.values()) and w['status'] == 'converged'
    text = ('PKPy2 converged for the four simulated comparison datasets and the warfarin PK/PD model. ' if converged else
            f"PKPy2 converged for {sum(r['status'] == 'converged' for r in rows.values())} of the four simulated comparison datasets. ")
    lo, hi = min(diffs), max(diffs)
    if lo >= -.05:
        text += (f"The exact OFV at the nlmixr2 FOCEi estimates was {max(lo, 0):.2f} to {hi:.2f} units higher than at the PKPy2 "
                 'estimates in the simulated datasets')
    else:
        text += (f"The exact OFV at the nlmixr2 FOCEi estimates differed from that at the PKPy2 estimates by {lo:.2f} to {hi:.2f} "
                 'units in the simulated datasets')
    text += (f", and for warfarin PK/PD the OFV was {w['focei_minus_pkpy2']:.2f} units higher at the FOCEi estimates and "
             f"{w['saem_minus_pkpy2']:.2f} units higher at the SAEM estimates (Figure 6D). Estimates of all programs are listed in "
             'Supplementary Tables S20 and S21. ')
    we = E.get('warfarin_pkpd_estimates')
    if we:
        text += (f"For warfarin PK/PD, PKPy2 estimated IMAX at {we['imax']:.4f} (nlmixr2 SAEM, {we['imax_saem']:.3f}; FOCEi, "
                 f"{we['imax_focei']:.3f}), CL at {we['cl']:.3f} L/h, and V at {we['v']:.2f} L (visual predictive checks of both "
                 'outputs in Figure 7).')
    return text


def results_recovery():
    rc, rp = E['recovery']['complex_linear'], E['recovery']['pkpd_idr']
    cc, cp = E['calibration']['complex_linear'], E['calibration']['pkpd_idr']
    scm = E['tools']['scm']
    freq = scm['selection_frequency']
    n = scm['replicates']
    true = ['CL~CRCL[power]', 'V~WT[power]']
    null = [k for k in freq if k not in true]
    both = round(scm['exact_true_model'] * n)
    tr = min(freq[k] for k in true)
    fp = sum(round(freq[k] * n) for k in null)
    cov_c = min(cc['vpc_coverage'].values())
    cov_p = min(cp['vpc_coverage'].values())
    conv = ('all fits converged in both designs' if rc['converged'] == rc['replicates'] and rp['converged'] == rp['replicates']
            else f"{rc['converged']} of {rc['replicates']} and {rp['converged']} of {rp['replicates']} fits converged")
    each = ('each true effect was selected in every dataset' if tr == 1 else
            f'each true effect was selected in at least {100 * tr:.0f}% of datasets')
    theta = max(rc['theta']['max_abs_bias'], rp['theta']['max_abs_bias'])
    coef = rc['coefficients']
    var_lo = min(rc['variance']['bias_min'], rp['variance']['bias_min'])
    var_hi = max(rc['variance']['bias_max'], rp['variance']['bias_max'])
    add = rc['sigma']
    return (f"In the recovery simulations, {conv}. Relative bias was within ±{theta:.1f}% for typical values, "
            f"{coef['bias_min']:.1f}% to {coef['bias_max']:.1f}% for the covariate coefficients, and {var_lo:.1f}% to "
            f"{var_hi:+.1f}% for variance components; the additive residual SD of the infusion model, which the concentrations "
            f"informed little, had a relative bias of {add['bias_min']:.0f}% (Supplementary Table S22). At the true parameters of "
            'the two designs, the mean NPDE was '
            f"{cc['npde_mean']:.3f} and {cp['npde_mean']:.3f}, the NPDE variance {cc['npde_variance']:.2f} and "
            f"{cp['npde_variance']:.2f}, and {100 * cov_c:.0f}% and {100 * cov_p:.0f}% of observed VPC percentiles lay within their "
            '95% intervals. For theophylline, the Wald, sandwich, profile-likelihood, bootstrap, and SIR intervals were similar for the '
            'typical values and differed more for the variance components (Figure 7). Stepwise covariate selection retained both true '
            f"effects in {both} of {n} datasets, {each}, and a null effect was selected in {fp} of {len(null) * n} null-candidate "
            'evaluations.')


def discussion():
    rows = E['vs_nlmixr2']
    lo = min(r['exact_ofv_difference'] for r in rows)
    w = E['warfarin_pkpd']
    both_lower = lo >= -.05 and w['focei_minus_pkpy2'] >= -.05 and w['saem_minus_pkpy2'] >= -.05
    est = ('Its estimates reached exact OFVs at least as low as those of nlmixr2 in all comparisons. ' if both_lower else
           'Exact OFVs identified where the programs reached different solutions. ')
    return ('The event-record interface extends the same estimation and audit to the data structures and models of routine '
            'population PK/PD analysis. Its predictions reproduced rxode2 to near machine precision for linear systems and to the '
            'integration tolerance for ODE models, its likelihoods reproduced an independent quadrature, and its diagnostics matched '
            'nlmixr2 and the npde package. ' + est + 'These checks cover the numerical components that users cannot inspect directly: '
            'event handling, likelihood integration, and residual definitions.')
