"""Apply the PeerJ editorial-check revision to the submitted PKPy2 manuscript and figure legends.

Edits are made in place on copies of the submitted Word files so that the original
formatting is preserved. Every replaced paragraph is located by its exact original text
prefix, so the script fails loudly if the source document differs from the one reviewed.
All quoted results are taken from docs/pkpy2_paper/manuscript_numbers.json
(scripts/pkpy2_manuscript_numbers.py), computed from the current records.
"""
from pathlib import Path
import copy
import json
import sys

import docx
from docx.enum.text import WD_COLOR_INDEX

ROOT = Path(__file__).resolve().parents[1]
SRC = Path(sys.argv[1])
DST = Path(sys.argv[2])
N = json.loads((ROOT / 'docs/pkpy2_paper/manuscript_numbers.json').read_text(encoding='utf-8'))
WORDS = {0: 'None', 1: 'One', 2: 'Two', 3: 'Three', 4: 'Four', 5: 'Five', 6: 'Six', 7: 'Seven', 8: 'Eight', 9: 'Nine', 10: 'Ten'}
SUP = str.maketrans('-0123456789', '⁻⁰¹²³⁴⁵⁶⁷⁸⁹')
LAB = {'pkpy2': 'PKPy2', 'nlmixr2_focei': 'nlmixr2 FOCEi', 'nlmixr2_saem': 'nlmixr2 SAEM', 'saemix': 'saemix'}
PAR = {'theta_CL': 'CL', 'theta_V': 'V', 'omega_CL': 'ω²(CL)', 'omega_V': 'ω²(V)', 'sigma_prop': 'σ_prop'}


def sci(s):
    mantissa, exponent = str(s).split('e')
    return f'{mantissa} × 10{str(int(exponent)).translate(SUP)}'


def find(doc, prefix):
    hits = [p for p in doc.paragraphs if p.text.startswith(prefix)]
    assert len(hits) == 1, (prefix, len(hits))
    return hits[0]


def set_text(p, text):
    runs = p.runs
    runs[0].text = text
    for r in runs[1:]:
        r._r.getparent().remove(r._r)


def insert_after(p, text, like=None):
    """Insert a new paragraph after p, formatted like `like` (default p)."""
    proto = like if like is not None else p
    new = copy.deepcopy(proto._p)
    p._p.addnext(new)
    para = docx.text.paragraph.Paragraph(new, p._parent)
    set_text(para, text)
    return para


def blank_after(p):
    """Insert an empty paragraph after p, copied from the section spacing used in the original."""
    proto = next(q for q in p._parent.paragraphs if not q.text.strip() and q.style.name == p.style.name)
    new = copy.deepcopy(proto._p)
    p._p.addnext(new)
    return docx.text.paragraph.Paragraph(new, p._parent)


def replace_in(p, old, new):
    assert old in p.text, (old, p.text[:80])
    for r in p.runs:
        if old in r.text:
            r.text = r.text.replace(old, new)
            return
    set_text(p, p.text.replace(old, new))


def converged_phrase(k, total, what):
    return f'All {WORDS[total].lower()} {what}' if k == total else f'{WORDS[k]} of the {WORDS[total].lower()} {what}'


def sim_comparison_text():
    """Results text of the simulation comparison with nlmixr2 and saemix."""
    ez = N['excluding_zero']
    rich = [e for e in ez if e['sampling'] == 'rich']
    if len(rich) == 1:
        e = rich[0]
        rich_text = (f"; the only paired difference whose bootstrap interval excluded zero was for {PAR[e['parameter']]} against "
                     f"{LAB[e['comparator']]} (PKPy2 minus {LAB[e['comparator']].split()[-1]}, {e['diff']:.2f} percentage points; 95% interval "
                     f"{e['lo']:.2f} to {e['hi']:.2f})")
    elif rich:
        parts = [f"{PAR[e['parameter']]} against {LAB[e['comparator']]} ({e['diff']:.2f} percentage points; 95% interval {e['lo']:.2f} to "
                 f"{e['hi']:.2f})" for e in rich]
        rich_text = '; paired differences (PKPy2 minus comparator) whose bootstrap intervals excluded zero were ' + ', '.join(parts)
    else:
        rich_text = '; no paired difference had a bootstrap interval excluding zero'
    sparse = {(e['comparator'], e['parameter']): e for e in ez if e['sampling'] == 'sparse'}
    omega = N['sparse_rmse_omegaV_tools']
    omega_pairs = [f"{sparse[(c, 'omega_V')]['diff']:.2f} ({sparse[(c, 'omega_V')]['lo']:.2f} to {sparse[(c, 'omega_V')]['hi']:.2f})"
                   for c in ['nlmixr2_saem', 'saemix'] if (c, 'omega_V') in sparse]
    focei = [p for p in ['theta_CL', 'theta_V'] if ('nlmixr2_focei', p) in sparse]
    text = (f"nlmixr2 FOCEi, nlmixr2 SAEM, and saemix returned estimates for all 200 primary datasets, and parameter recovery was similar "
            f"across programs (Table 6 and Figure 5). Under rich sampling, relative RMSEs differed from PKPy2 by at most "
            f"{N['rich_max_rmse_gap']} percentage points for every parameter{rich_text}. Under sparse sampling, nlmixr2 FOCEi overestimated "
            f"CL by {N['sparse_bias_CL_focei']}% on average, compared with {N['sparse_bias_CL_pkpy2_cmp']}% for PKPy2")
    if focei:
        text += (', and its relative RMSEs for ' + ' and '.join(PAR[p] for p in focei) + ' were higher by '
                 + ' and '.join(f"{-sparse[('nlmixr2_focei', p)]['diff']:.2f}" for p in focei) + ' percentage points (95% intervals for PKPy2 '
                 'minus FOCEi, ' + ' and '.join(f"{sparse[('nlmixr2_focei', p)]['lo']:.2f} to {sparse[('nlmixr2_focei', p)]['hi']:.2f}" for p in focei)
                 + ')')
    text += (f". The sparse-sampling relative RMSE of ω²(V) was {omega['pkpy2']}% with PKPy2, {omega['nlmixr2_focei']}% with nlmixr2 FOCEi, "
             f"{omega['nlmixr2_saem']}% with nlmixr2 SAEM, and {omega['saemix']}% with saemix")
    if len(omega_pairs) == 2:
        text += f"; the paired differences from nlmixr2 SAEM and saemix were {omega_pairs[0]} and {omega_pairs[1]} percentage points"
    text += '. All paired differences are listed in Supplementary Table S12.'
    return text


def manuscript():
    d = docx.Document(SRC / 'PKPy2_PeerJ_manuscript.docx')
    assert N['converged_rich'] == 100 and N['converged_sparse'] == 100
    assert N['theo_pkpy2_lowest'] or float(N['others_within']) < .1
    cov = N['coverage_tools']
    sec_total = N['secondary_total_converged']

    # ---------------------------------------------------------------- Abstract
    set_text(find(d, 'Population pharmacokinetic (PopPK) analysis often relies'),
        'Population pharmacokinetic (PopPK) analysis usually relies on specialized, often commercial, software. PKPy introduced a Python workflow, '
        'but it derived population parameters from separately fitted individual models, so individual estimation error entered the estimated '
        'interindividual variability. We developed PKPy2, a standalone Python package that estimates structural parameters, interindividual '
        'variability, residual error, and covariate effects jointly by marginal likelihood for declared compartmental models, with optional fixed '
        'values and bounds. PKPy2 also provides analytical repeated-dose predictions, numerical convergence checks, and local confidence intervals, '
        'and its calculations agreed closely with independent implementations. In 200 simulated one-compartment intravenous datasets, all PKPy2 fits '
        f'converged, and under sparse sampling the relative root mean squared error (RMSE) of the interindividual variance of volume was '
        f"{N['sparse_rmse_pkpy2_omega_V']}%, compared with {N['sparse_rmse_pkpy_omega_V']}% for PKPy. Parameter recovery was similar to that of "
        f"nlmixr2 and saemix on the same datasets. Empirical coverage of nominal 95% intervals ranged from {N['coverage_min']}% to "
        f"{N['coverage_max']}%, and {sec_total} of 30 datasets with three additional model structures converged. In clinical theophylline and "
        f"warfarin data, PKPy2 estimates differed from published expert NONMEM estimates by at most {N['theo_max_diff']}% and "
        f"{N['warf_max_diff']}%. In tobramycin data, the published model had a different maximum-likelihood solution; when the expert's "
        f"pharmacological judgment was entered as fixed values and bounds, PKPy2 reproduced the expert estimates within {N['tob_ej_max']}%. "
        'PKPy2 provides joint PopPK estimation and uncertainty assessment in Python, with performance comparable to established open-source software.')

    # ---------------------------------------------------------------- Introduction
    set_text(find(d, 'We evaluated the numerical calculations against independent implementations'),
        'We evaluated the numerical calculations against independent implementations and examined parameter recovery and interval coverage in repeated '
        'simulations under rich and sparse sampling. The original PKPy fitting components and a Gaussian-likelihood two-stage estimator served as '
        'development comparators, and the established open-source nonlinear mixed-effects programs nlmixr2 and saemix were applied to the same '
        'simulated datasets. Smaller simulation experiments assessed three additional compartmental structures. We validated PKPy2 with real clinical '
        'data by applying it to the public theophylline, warfarin, and tobramycin datasets and comparing the estimates with published expert NONMEM '
        'analyses and with nlmixr2 and saemix fits of the same models. All evaluations used prespecified pharmacokinetic models.')

    # ---------------------------------------------------------------- Methods
    replace_in(find(d, 'where fᵢⱼ is the model-predicted concentration.'),
               'Fixed components were excluded from the optimization coordinates and the count of free parameters.',
               'Fixed components were excluded from the optimization coordinates and the count of free parameters, and estimated components '
               'could be restricted by lower and upper bounds on the reporting scale, as with parameter bounds in NONMEM.')
    replace_in(find(d, 'PKPy2 used a staged estimation procedure'),
               'PKPy2 used a staged estimation procedure consisting of initial marginal-likelihood refinement, stochastic approximation '
               'expectation-maximization when required, and subsequent importance-sampling refinement.',
               'PKPy2 used a staged estimation procedure. A deterministic Laplace approximation of the marginal likelihood was first minimized '
               'from the starting values and from quasi-random perturbations of the structural parameters; the best Laplace optimum served '
               'only as the starting point for the subsequent stages, and distinct optima were recorded as a multimodality diagnostic. '
               'Initial marginal-likelihood refinement followed, with stochastic approximation expectation-maximization when required and '
               'subsequent importance-sampling refinement.')
    replace_in(find(d, 'Final convergence required agreement between two independently generated integration banks.'),
               'and the projected score accounted for the lower bound on interindividual variance.',
               'and the projected score accounted for declared bounds and the lower bound on interindividual variance.')
    replace_in(find(d, 'Intervals were reported when the information matrix was positive definite'),
               'Interval availability and empirical coverage were evaluated separately.',
               'Interval availability and empirical coverage were evaluated separately. Intervals were not calculated for estimates on a '
               'declared bound.')
    replace_in(find(d, 'Each dataset was analyzed with PKPy2, the unchanged model'),
               'from the archived PKPy 0.1.0 release', 'of PKPy')

    head = find(d, 'Clinical applications and expert NONMEM reference analyses')
    set_text(head, 'Real-world clinical data and expert NONMEM reference analyses')
    replace_in(find(d, 'PKPy2 was applied to the public theophylline and warfarin datasets'),
               'PKPy2 was applied to the public theophylline and warfarin datasets [10,11].',
               'PKPy2 was applied to the public theophylline, warfarin, and tobramycin datasets [10,11].')
    last = find(d, 'PKPy2 parameter estimates were compared with expert NONMEM results')
    replace_in(last, 'The theophylline specification was equivalent to the archived NONMEM model, and the warfarin specification followed '
               'the published model description.', 'The PKPy2 models matched the expert NONMEM models.')
    body_proto = last
    tob = insert_after(last,
        f"The tobramycin dataset of the PKGPT study contains {N['tob_obs']} concentrations from {N['tob_subjects']} patients who received "
        f"{N['tob_doses']} intravenous bolus doses [11]. The expert model is a two-compartment model with power effects of creatinine clearance "
        'on CL and of body weight on the central volume (V1), interindividual variability on CL, and a combined residual error with a fixed '
        'additive term (Supplementary Section S10). PKPy2 fitted this model from the documented initial values of the expert analysis and from '
        'the expert estimates. Because these fits reached a different optimum (Results), the model was refitted with the modeling judgment '
        'underlying the expert solution expressed as fixed values and bounds: V1 proportional to body weight (exponent fixed at 1), a small '
        'central compartment (V1 ≤ 10 L), a peripheral volume consistent with the extracellular distribution of aminoglycosides (V2 ≤ 30 L), '
        'and the remaining parameter bounds of the expert analysis. These expert-judgment constraints were used for the comparison with the '
        'other programs.', like=body_proto)
    h = insert_after(blank_after(tob), 'Comparison with established open-source NLME software', like=head)
    p1 = insert_after(h,
        'To compare PKPy2 with existing tools, the 200 primary simulation datasets and the three clinical datasets were also analyzed with nlmixr2 [15] '
        'using first-order conditional estimation with interaction (FOCEi) and stochastic approximation expectation-maximization (SAEM) [21], and with '
        'the saemix package [5]. All programs received the same observation records, exported from the PKPy2 analysis data, together with the same '
        'structural and statistical models, fixed terms, bounds, and starting values. nlmixr2 used analytical linear-compartment solutions, and '
        'ordinary differential equations for the warfarin model with absorption lag. Default estimation and covariance settings were used: the '
        'sandwich covariance for FOCEi, the stochastic-approximation covariance for SAEM, and the linearized Fisher information for saemix. The '
        'combined residual-error variance was defined as in PKPy2. nlmixr2 SAEM does not apply parameter bounds. saemix was applied to the simulated '
        'and theophylline data; it does not support the fixed variance terms of the warfarin model or the bounds of the tobramycin model.',
        like=body_proto)
    insert_after(p1,
        'Simulation estimates were summarized with the metrics used in the primary evaluation. Wald 95% intervals for CL and V were formed on the '
        'logarithmic scale from the standard errors reported by each program. Paired differences in relative RMSE between PKPy2 and each program were '
        'evaluated with 10,000 bootstrap resamples of datasets [20]. Because the objective functions reported by the programs are defined differently, '
        'the clinical estimates of every program were also evaluated with a single exact marginal OFV, computed by adaptive Gauss-Hermite quadrature '
        '[3,18] in a separate implementation that shares no code with PKPy2 (Supplementary Section S8). The comparison used R 4.5.3 [22], nlmixr2 7.0.1 '
        '(nlmixr2est 7.1.0, rxode2 5.1.7), and saemix 3.5 on the computer described below, with one thread per fit and five concurrent processes.',
        like=body_proto)

    # ---------------------------------------------------------------- Results
    set_text(find(d, 'Analytical predictions were evaluated under 48 conditions'),
        'Analytical predictions were evaluated under 48 conditions spanning the four supported model structures. The maximum scaled discrepancy from '
        f"the independent calculations was {sci(N['pred_max'])} for concentration predictions and {sci(N['sens_max'])} for parameter sensitivities. "
        'The four comparisons with the independent Gaussian-quadrature implementation also showed close agreement. The maximum objective function '
        f"difference at identical parameter values was {sci(N['quad_same_point_ofv'])}, the maximum difference between optimized coordinates was "
        f"{sci(N['quad_coord_gap'])}, and the maximum relative difference in coordinate standard errors was {N['quad_se_gap_pct']}% (Table 2).")
    replace_in(find(d, 'Table 1 summarizes the main changes from PKPy to PKPy2.'), 'adds explicit fixed-parameter handling',
               'adds explicit fixed-parameter and bound handling')
    replace_in(find(d, 'PKPy2 converged for all 100 rich-sampling and 100 sparse-sampling datasets'),
               'and every primary fit converged during the initial marginal-likelihood refinement.',
               f"and {'every primary fit' if N['primary_no_saem'] == 200 else str(N['primary_no_saem']) + ' of the 200 fits'} converged without "
               'the SAEM stage.')
    set_text(find(d, 'Under sparse sampling, the relative RMSEs for CL and V were'),
        f"Under sparse sampling, the relative RMSEs for CL and V were {N['sparse_rmse_pkpy_theta_CL']}% and {N['sparse_rmse_pkpy_theta_V']}% with "
        f"PKPy and {N['sparse_rmse_pkpy2_theta_CL']}% and {N['sparse_rmse_pkpy2_theta_V']}% with PKPy2. Larger differences were observed for the "
        f"interindividual variance components. The relative RMSE of ω²(CL) decreased from {N['sparse_rmse_pkpy_omega_CL']}% with PKPy to "
        f"{N['sparse_rmse_pkpy2_omega_CL']}% with PKPy2, and the relative RMSE of ω²(V) decreased from {N['sparse_rmse_pkpy_omega_V']}% to "
        f"{N['sparse_rmse_pkpy2_omega_V']}%. The relative bias of ω²(V) was {N['sparse_bias_pkpy_omega_V']}% with PKPy and "
        f"{N['sparse_bias_pkpy2_omega_V']}% with PKPy2. The paired difference in the relative RMSE of ω²(V), calculated as PKPy2 minus PKPy, was "
        f"{N['paired_pkpy_omegaV']} percentage points, with a bootstrap 95% interval of {N['paired_pkpy_omegaV_lo']} to "
        f"{N['paired_pkpy_omegaV_hi']} percentage points (Figure 2).")
    lo, hi = float(N['paired_gaussian_tst_omegaV_lo']), float(N['paired_gaussian_tst_omegaV_hi'])
    set_text(find(d, 'For the Gaussian two-stage control, the sparse-sampling relative RMSEs were'),
        f"For the Gaussian two-stage control, the sparse-sampling relative RMSEs were {N['sparse_rmse_gaussian_tst_theta_CL']}% for CL, "
        f"{N['sparse_rmse_gaussian_tst_theta_V']}% for V, {N['sparse_rmse_gaussian_tst_omega_CL']}% for ω²(CL), and "
        f"{N['sparse_rmse_gaussian_tst_omega_V']}% for ω²(V). The paired PKPy2-minus-control difference for ω²(V) was "
        f"{N['paired_gaussian_tst_omegaV']} percentage points, with a bootstrap 95% interval of {N['paired_gaussian_tst_omegaV_lo']} to "
        f"{N['paired_gaussian_tst_omegaV_hi']} percentage points.")
    set_text(find(d, 'Local 95% confidence intervals were obtained for all 200 primary fits.'),
        f"Local 95% confidence intervals were obtained for all 200 primary fits. Empirical coverage ranged from {N['coverage_min']}% to "
        f"{N['coverage_max']}% across parameters and sampling designs. Under sparse sampling, coverage was {N['coverage_sparse_theta_CL']}% for CL, "
        f"{N['coverage_sparse_theta_V']}% for V, {N['coverage_sparse_omega_CL']}% for ω²(CL), {N['coverage_sparse_omega_V']}% for ω²(V), and "
        f"{N['coverage_sparse_sigma_prop']}% for the proportional residual standard deviation. Interindividual variance intervals were wider than "
        'the CL and V intervals, particularly under sparse sampling (Figure 3).')
    c1, c2, c3 = (N['secondary_converged_1cmt_oral'], N['secondary_converged_2cmt_iv'], N['secondary_converged_2cmt_oral'])
    sec = (f"{converged_phrase(c1, 10, 'one-compartment oral datasets')} and {converged_phrase(c2, 10, 'two-compartment intravenous datasets').lower()} "
           f"converged. {converged_phrase(c3, 10, 'two-compartment oral datasets')} converged. Among these {WORDS[c3].lower()} fits, the relative "
           f"RMSEs were {N['secondary_rmse_2cmt_oral_theta_V1']}% for V1, {N['secondary_rmse_2cmt_oral_theta_Ka']}% for Ka, and "
           f"{N['secondary_rmse_2cmt_oral_omega_V1']}% for ω²(V1). Structural parameter RMSEs were at most {N['sec_1cmt_oral_struct_max']}% in the "
           f"one-compartment oral model and at most {N['sec_2cmt_iv_struct_max']}% in the two-compartment intravenous model.")
    set_text(find(d, 'All ten one-compartment oral datasets and all ten two-compartment intravenous datasets converged.'), sec)

    rhead = find(d, 'Clinical applications and comparison with NONMEM estimates')
    set_text(rhead, 'Real-world clinical data: comparison with NONMEM, nlmixr2, and saemix')
    set_text(find(d, 'In the theophylline proportional-error model, PKPy2 estimated'),
        f"In the theophylline proportional-error model, PKPy2 estimated CL70/F as {N['theo_CL']} L/h, V70/F as {N['theo_V']} L, and Ka as "
        f"{N['theo_Ka']} h⁻¹. The three structural parameter estimates and three interindividual variance estimates differed by no more than "
        f"{N['theo_max_diff']}% from the expert NONMEM estimates. Structural-parameter RSEs were also similar, and larger RSE differences were "
        f"observed for some variance components. The RSE of ω²(V), for example, was {N['theo_rse_omegaV_pkpy2']}% with PKPy2 and "
        f"{N['theo_rse_omegaV_nonmem']}% with NONMEM (Table 3).")
    big = [x.split(' (')[0].replace('ALAG', 'absorption lag time') for x in N['warf_rse_largest']]
    wf = find(d, 'Using the published warfarin model specification')
    set_text(wf,
        f"For warfarin, the five freely estimated parameters differed by no more than {N['warf_max_diff']}% from the expert NONMEM estimates. The "
        f"largest difference was observed for {N['warf_max_param'].split(' (')[0]}. Point estimates agreed more closely than RSEs, with the largest "
        f"RSE differences observed for {big[0]} and {big[1]} (Table 4 and Figure 4).")
    ofv_warf_saem = float(N['warf_ofv_saem_minus_pkpy2'])
    p = insert_after(wf,
        f"The same clinical models were fitted with nlmixr2 and saemix (Table 5 and Figure 4). For theophylline, all {N['theo_tools_n']} point "
        f"estimates from the four open-source programs were within {N['theo_tools_max_diff']}% of the expert NONMEM estimates, "
        f"and the exact marginal OFVs at the five sets of estimates, including NONMEM, were within {N['theo_ofv_spread']} units of one another "
        f"(PKPy2, {N['theo_ofv_pkpy2']}; NONMEM, {N['theo_ofv_nonmem']}). For warfarin, the nlmixr2 FOCEi estimates were within "
        f"{N['warf_focei_max_diff']}% of the expert estimates, and the exact OFV at these estimates was {N['warf_ofv_focei_minus_pkpy2']} units "
        f"higher than at the PKPy2 estimates. nlmixr2 SAEM estimated Ka and absorption lag time {N['warf_saem_ka_diff']}% and "
        f"{N['warf_saem_alag_diff']}% lower than the expert estimates, and the exact OFV at its estimates was {N['warf_ofv_saem_minus_pkpy2']} units "
        'higher than at the PKPy2 estimates, indicating that the SAEM run stopped at an inferior solution of the same model. Standard errors '
        f"differed more than point estimates across programs (Supplementary Table S13).")
    assert ofv_warf_saem > 3
    tf = N['tob_fits']
    t1 = insert_after(p,
        f"For tobramycin, PKPy2 fits of the published model from the documented initial values and from the expert estimates converged to the same "
        f"maximum-likelihood estimate (OFV {N['tob_unc_ofv']}), which differed from the expert solution: the body-weight exponent of V1 was "
        f"{N['tob_unc_bwt']}, and V2 was {N['tob_unc_V2']} L (Supplementary Table S15). The exact OFV at the expert estimates was "
        f"{N['tob_expert_minus_unc']} units higher. An independent quadrature calculation reproduced the PKPy2 OFV, and nlmixr2 FOCEi and saemix "
        'fits of the same model also gave negative weight exponents (Supplementary Table S17). The data carry little information on distribution: '
        f"{N['tob_obs_at_2p5']} of {N['tob_obs']} concentrations were measured 2.5 h after a dose, body weight ranged only from "
        f"{N['tob_wt_range'][0]} to {N['tob_wt_range'][1]} kg and was correlated with creatinine clearance (r = {N['tob_corr_wt_clcr']} on the "
        f"log scale), and the profile OFV varied by {N['tob_profile_range']} units across V2 values from {N['tob_profile_v2'][0]} to "
        f"{N['tob_profile_v2'][1]} L (Supplementary Table S16). Within the parameter bounds of the expert NONMEM analysis, the best solution lay "
        f"on the bounds (weight exponent {float(N['tob_bounded_1']['b_wt']):g}, V2 {N['tob_bounded_1']['V2']} L), and the fit started from the expert "
        f"estimates remained near them with an OFV {N['tob_bounded_gap']} units higher; the nlmixr2 FOCEi objective ranked the two solutions in "
        'the same order.', like=wf)
    saem_rows = N['tob_saem_diffs']
    insert_after(t1,
        f"With the expert-judgment constraints, PKPy2 converged from both starting points to the same estimate, with no constraint active, and "
        f"reproduced the expert estimates within {N['tob_ej_max']}% (Table 5 and Figure 4C). The OFV of this estimate was {N['tob_ej_minus_unc']} "
        f"units above the unconstrained maximum. nlmixr2 FOCEi with the same constraints gave estimates within {N['tob_nlmixr2_focei_max']}% of the "
        f"expert estimates, and the exact OFV at its estimates was {N['tob_ofv_focei_minus_pkpy2']} units higher than at the PKPy2 estimates. "
        f"nlmixr2 SAEM, which does not apply the bounds, estimated CL and Q {saem_rows['CL']}% and {saem_rows['Q']}% higher than the expert "
        f"estimates, with an exact OFV {N['tob_ofv_saem_minus_pkpy2']} units higher than at the PKPy2 estimates.", like=wf)

    h2 = insert_after(blank_after(find(d, 'With the expert-judgment constraints, PKPy2 converged')),
                      'Comparison with established software in the primary simulation', like=rhead)
    p2 = insert_after(h2, sim_comparison_text(), like=wf)
    ms = N['median_fit_seconds']

    def span(m):
        a, b = sorted([ms[f'rich_{m}'], ms[f'sparse_{m}']])
        return f'{a}-{b} s'
    insert_after(p2,
        f"Coverage of nominal 95% intervals for CL and V was {cov['pkpy2'][0]}% to {cov['pkpy2'][1]}% for PKPy2, {cov['nlmixr2_focei'][0]}% to "
        f"{cov['nlmixr2_focei'][1]}% for nlmixr2 FOCEi, {cov['nlmixr2_saem'][0]}% to {cov['nlmixr2_saem'][1]}% for nlmixr2 SAEM, and "
        f"{cov['saemix'][0]}% to {cov['saemix'][1]}% for saemix. Across the two sampling designs, median wall-clock times per fit were "
        f"{span('saemix')} for saemix, {span('nlmixr2_focei')} for nlmixr2 FOCEi, {span('nlmixr2_saem')} for nlmixr2 SAEM, and "
        f"{span('pkpy2')} for PKPy2, which computed intervals in a separate step (medians {N['median_uncertainty_seconds']['sparse']}-"
        f"{N['median_uncertainty_seconds']['rich']} s).", like=wf)
    replace_in(find(d, 'Prediction-call times after compilation were similar'), '(Figure 5)', '(Figure 6)')

    # ---------------------------------------------------------------- Discussion
    disc = find(d, 'Discussion\nPKPy2 advances PKPy by replacing population summaries')
    assert disc.runs[1].text == '\n' and len(disc.runs) == 3
    disc.runs[2].text = (
        'PKPy2 advances PKPy by replacing population summaries of individual fits with joint marginal-likelihood inference. The advantage was most '
        f"evident for interindividual variability under sparse sampling. The relative RMSE of ω²(V) decreased from {N['sparse_rmse_pkpy_omega_V']}% "
        f"with PKPy to {N['sparse_rmse_pkpy2_omega_V']}% with PKPy2, and relative bias changed from {N['sparse_bias_pkpy_omega_V']}% to "
        f"{N['sparse_bias_pkpy2_omega_V']}%. CL and V were already recovered with low error by both workflows, so differences in their typical values "
        'were smaller. Joint estimation uses all subjects simultaneously and estimates residual error together with population variability, reducing '
        'the contribution of uncertainty from separate individual fits to the reported variance components.')
    includes_zero = lo < 0 < hi
    set_text(find(d, 'The Gaussian-likelihood two-stage control provided an informative reference'),
        'The Gaussian-likelihood two-stage control provided an informative reference because it used the generating residual standard deviation. '
        f"Its sparse-sampling RMSE for ω²(V) was {N['sparse_rmse_gaussian_tst_omega_V']}%, compared with {N['sparse_rmse_pkpy2_omega_V']}% for PKPy2, "
        f"and the paired bootstrap interval for the difference {'included' if includes_zero else 'excluded'} zero. PKPy2 estimated residual error "
        'jointly with the population parameters and improved both interindividual variance estimates relative to the original PKPy implementation. '
        'The comparison shows how population estimation and residual-error specification jointly shape variance recovery.')
    set_text(find(d, 'Independent calculations verified the numerical accuracy of the implementation.'),
        'Independent calculations verified the numerical accuracy of the implementation. All 48 prediction conditions agreed with the independent '
        'matrix-exponential and numerical-integration calculations, and the independent quadrature comparisons closely reproduced objective values, '
        f"optimized coordinates, and coordinate standard errors. The maximum same-point OFV difference was {sci(N['quad_same_point_ofv'])}, and the "
        f"maximum relative difference in coordinate standard errors was {N['quad_se_gap_pct']}%. All 200 primary fits converged and produced local "
        'confidence intervals, demonstrating stable marginal-likelihood estimation and information-matrix calculation across both sampling designs.')
    set_text(find(d, 'Empirical coverage of the nominal 95% intervals ranged from'),
        f"Empirical coverage of the nominal 95% intervals ranged from {N['coverage_min']}% to {N['coverage_max']}%. Sparse-sampling coverage was "
        f"{N['coverage_sparse_theta_CL']}% for CL and {N['coverage_sparse_omega_V']}% for ω²(V), showing parameter-specific differences in "
        'finite-sample calibration. Variance-component intervals were wider than structural-parameter intervals, particularly under sparse sampling. '
        'The combination of coverage and interval width provides a direct view of the precision available for each parameter and sampling design.')
    set_text(find(d, 'Performance extended beyond the primary one-compartment intravenous model.'),
        f"Performance extended beyond the primary one-compartment intravenous model. {sec_total} of the 30 additional-structure datasets converged. "
        f"Structural parameter RMSEs were at most {N['sec_1cmt_oral_struct_max']}% in the one-compartment oral model and at most "
        f"{N['sec_2cmt_iv_struct_max']}% in the two-compartment intravenous model. The two-compartment oral model was more demanding, with relative "
        f"RMSEs of {N['secondary_rmse_2cmt_oral_theta_V1']}% for V1, {N['secondary_rmse_2cmt_oral_theta_Ka']}% for Ka, and "
        f"{N['secondary_rmse_2cmt_oral_omega_V1']}% for ω²(V1). These results identify absorption and distribution models as the next priority for "
        'precision improvements.')
    clin = find(d, 'The clinical analyses showed close agreement with established NONMEM results.')
    set_text(clin,
        'The clinical analyses showed close agreement with established NONMEM results. In the theophylline analysis, all six structural and '
        f"interindividual variance estimates were within {N['theo_max_diff']}% of the expert NONMEM estimates. In the warfarin analysis, five freely "
        f"estimated parameters differed by no more than {N['warf_max_diff']}% from the expert estimates. RSEs varied more across implementations, "
        'consistent with their distinct likelihood approximations and covariance calculations. The close point-estimate agreement across both '
        'datasets supports the use of PKPy2 for declared oral pharmacokinetic models.')
    c2p = insert_after(clin,
        'With identical datasets, models, and starting values, PKPy2, nlmixr2, and saemix recovered structural parameters and variance components '
        f"with similar accuracy under rich sampling. Under sparse sampling, the FOCEi approximation overestimated clearance by "
        f"{N['sparse_bias_CL_focei']}% on average, and the SAEM-based programs estimated the variance of volume less precisely than PKPy2. In the "
        'clinical datasets the programs agreed closely, except for the nlmixr2 SAEM fits of warfarin and tobramycin, whose exact objective function '
        'values were well above those at the PKPy2 estimates. PKPy2 reports convergence after projected marginal scores are small in two '
        'independent integration banks, a criterion designed to detect such non-stationary solutions. nlmixr2 and saemix support broader model '
        'classes, and saemix and nlmixr2 FOCEi had shorter median fit times.')
    insert_after(c2p,
        'The tobramycin analysis shows how PKPy2 responds to pharmacological judgment. The published model had a maximum-likelihood solution with a '
        'negative weight exponent and a large peripheral volume, because concentrations were measured mostly at a single time after dose and body '
        'weight varied little and was correlated with creatinine clearance; nlmixr2 and saemix reached similar solutions. The expert solution reflects '
        'additional knowledge of how tobramycin distributes. Entered as fixed values and bounds, this knowledge led PKPy2 to the best solution '
        'within the constraints from either starting point, and that solution reproduced the expert estimates. Declared constraints therefore let a '
        'clinical pharmacologist steer an analysis toward pharmacologically plausible solutions while the likelihood remains the criterion within '
        'them, and the exact OFV shows how much fit each constraint costs.')
    replace_in(find(d, 'The evaluation used 30-subject datasets, diagonal random effects'),
               'The evaluation used 30-subject datasets, diagonal random effects, specified linear compartmental models, and ten replicates for each additional structure.',
               'The evaluation used 30-subject datasets, diagonal random effects, specified linear compartmental models, and ten replicates for each additional '
               'structure. The NONMEM comparisons used the expert estimates as reported, and nlmixr2 and saemix were run with default settings. The '
               'tobramycin constraints were formulated with knowledge of the expert solution; the data alone supported a different solution.')
    replace_in(find(d, 'Overall, PKPy2 provides a unified Python workflow'),
               'close agreement with independent calculations and expert NONMEM estimates,',
               'close agreement with independent calculations and expert NONMEM estimates, performance comparable to nlmixr2 and saemix on identical data,')
    replace_in(find(d, 'Overall, PKPy2 provides a unified Python workflow'),
               'for explicit PopPK model specification, joint estimation,',
               'for explicit PopPK model specification with fixed values and bounds, joint estimation,')

    # ---------------------------------------------------------------- Conclusions
    set_text(find(d, 'PKPy2 extends PKPy from separate individual fits to joint nonlinear'),
        'PKPy2 extends PKPy from separate individual fits to joint nonlinear mixed-effects population estimation. In the primary simulation, all 200 '
        'fits converged, and the sparse-sampling relative RMSE of the interindividual variance in volume decreased from '
        f"{N['sparse_rmse_pkpy_omega_V']}% with PKPy to {N['sparse_rmse_pkpy2_omega_V']}% with PKPy2. Independent calculations closely matched PKPy2 "
        'predictions, likelihoods, optimized parameters, and standard errors. On the same simulated and clinical datasets, PKPy2 performed comparably '
        'to nlmixr2 and saemix. Theophylline and warfarin estimates agreed closely with published expert NONMEM estimates, tobramycin estimates '
        'reproduced the expert estimates once the expert\'s pharmacological judgment was declared as fixed values and bounds, and analytical '
        'dose-state propagation accelerated prediction for long dose histories.')

    # ---------------------------------------------------------------- Data availability
    da = find(d, 'PKPy2, installation instructions, runnable examples, public benchmark data')
    p = insert_after(da,
        'Raw data underlying all figures and tables are provided as Supplemental Files: PKPy2_raw_data.xlsx (with the same sheets as CSV files) contains '
        'the simulated and clinical analysis datasets, per-dataset estimates from all programs, numerical checks, and timings, and PKPy2_codebook.xlsx '
        'defines every variable and categorical code. These files, together with the nlmixr2 and saemix comparison scripts and their saved fit '
        'records, are archived at Zenodo (https://doi.org/10.5281/zenodo.XXXXXXX).')
    for r in p.runs:
        if 'XXXXXXX' in r.text:
            r.font.highlight_color = WD_COLOR_INDEX.YELLOW

    # ---------------------------------------------------------------- References
    ref20 = find(d, '[20] Efron B.')
    r21 = insert_after(ref20, '[21] Schoemaker R, Fidler M, Laveille C, Wilkins JJ, Hooijmaijers R, Post TM, Trame MN, Xiong Y, Wang W. Performance of '
                       'the SAEM and FOCEI algorithms in the open-source, nonlinear mixed effect modeling tool nlmixr. CPT: Pharmacometrics & Systems '
                       'Pharmacology. 2019;8(12):923-930. doi:10.1002/psp4.12471.')
    insert_after(r21, '[22] R Core Team. R: A language and environment for statistical computing, version 4.5.3. Vienna, Austria: R Foundation for '
                      'Statistical Computing; 2026. https://www.R-project.org/.')
    d.save(DST / 'PKPy2_PeerJ_manuscript_revised.docx')


LEGENDS = {
    1: 'Figure 1. Population inference in PKPy and PKPy2. (A) PKPy fits each subject separately and summarizes the individual log parameters. '
       '(B) PKPy2 fits the declared population model jointly by marginal likelihood and reports independent numerical checks and local intervals.',
    2: 'Figure 2. Parameter recovery in the primary simulation by PKPy, the Gaussian two-stage control, and PKPy2. Left, relative bias with '
       '±1.96 Monte Carlo standard errors; right, relative RMSE. Each point summarizes 100 datasets (98 for the rich-sampling Gaussian control).',
    3: 'Figure 3. Coverage (A) and width (B) of PKPy2 95% confidence intervals in the primary simulation. Error bars in (A) are Wilson 95% '
       'intervals, and the dashed line marks 95%. Widths are medians relative to the true value, with interquartile ranges.',
    4: 'Figure 4. Differences of the theophylline (A), warfarin (B), and tobramycin (C) estimates from the published expert NONMEM estimates. '
       'Tobramycin estimates were obtained with the expert-judgment constraints.',
    5: 'Figure 5. Relative errors of PKPy2, nlmixr2 FOCEi, nlmixr2 SAEM, and saemix estimates in the 200 primary simulation datasets. Boxes show '
       'medians and interquartile ranges, whiskers extend to 1.5 times the interquartile range, and diamonds mark means.',
    6: 'Figure 6. Prediction-call speedup of dose-state recurrence over direct summation.',
}


def legends():
    d = docx.Document(SRC / 'PKPy2_PeerJ_figure_legends.docx')
    paras = [p for p in d.paragraphs if p.text.strip()]
    assert [p.runs[0].text for p in paras] == [f'Figure {i}. ' for i in range(1, 6)]
    assert paras[4].text.startswith('Figure 5. Prediction-call speedup')
    sixth = copy.deepcopy(paras[4]._p)
    paras[4]._p.addnext(sixth)
    paras.append(docx.text.paragraph.Paragraph(sixth, paras[4]._parent))
    for i, p in enumerate(paras, start=1):
        label, body = LEGENDS[i].split('. ', 1)
        p.runs[0].text = label + '. '
        p.runs[1].text = body
        for r in p.runs[2:]:
            r._r.getparent().remove(r._r)
    d.save(DST / 'PKPy2_PeerJ_figure_legends_revised.docx')


if __name__ == '__main__':
    DST.mkdir(parents=True, exist_ok=True)
    manuscript()
    legends()
    print('written to', DST)
