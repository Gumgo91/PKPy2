"""Write the codebook for PKPy2_raw_data.xlsx (variables, units and categorical codes)."""
from pathlib import Path
import sys

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from pkpy2_extended_raw_data import CODEBOOK as EXTENDED_CODEBOOK   # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
OUT = Path(sys.argv[1]) if len(sys.argv) > 1 else ROOT / 'output/pkpy2_peerj_raw_data'

EVENT = [
    ('ID', 'Subject identifier within the dataset', 'integer', '', 'Subjects are numbered 1..N within each dataset; identifiers are not shared across datasets'),
    ('TIME', 'Time after the first dose', 'numeric', 'h', 'Single dose in all datasets except tobramycin (repeated IV bolus doses)'),
    ('AMT', 'Dose amount; empty on observation records', 'numeric', 'mg', ''),
    ('DV', 'Observed plasma concentration; empty on dose records', 'numeric', 'mg/L', 'Simulated data, theophylline, warfarin and tobramycin: mg/L'),
    ('EVID', 'Event identifier (NONMEM convention)', 'categorical (numeric code)', '', '0 = observation record; 1 = dose record'),
    ('MDV', 'Missing dependent variable flag (NONMEM convention)', 'categorical (numeric code)', '', '0 = DV is an observation used in estimation; 1 = no observation on this record (dose record)'),
    ('CMT', 'Compartment receiving the dose or observed', 'categorical (numeric code)', '',
     'IV models: 1 = central compartment (dose and observations). Oral models: 1 = depot (absorption) compartment receiving the dose; 2 = central compartment where concentrations are observed'),
]
SIM = [
    ('dataset', 'Dataset identifier: <model>__<sampling>__<replicate> (primary) or <model>__<replicate> (additional)', 'text', '', ''),
    ('model', 'Structural model used to simulate the data', 'categorical (text)', '',
     '1cmt_iv = one-compartment IV bolus; 1cmt_oral = one-compartment first-order oral absorption; 2cmt_iv = two-compartment IV bolus; 2cmt_oral = two-compartment first-order oral absorption'),
    ('sampling', 'Sampling design of the primary simulation', 'categorical (text)', '',
     'rich = 0.25, 0.5, 1, 2, 4, 6, 8, 12, 18, 24 h; sparse = 1, 8, 24 h'),
    ('replicate', 'Replicate index within a model and sampling design', 'integer', '', '0-99 (primary), 0-9 (additional)'),
    ('seed', 'Random-number seed used to generate the dataset', 'integer', '', ''),
]
ETA = [
    ('ETA_CL_TRUE', 'True individual random effect on log CL used for simulation', 'numeric', 'log scale', ''),
    ('ETA_V_TRUE', 'True individual random effect on log V (one-compartment models)', 'numeric', 'log scale', ''),
    ('ETA_V1_TRUE', 'True individual random effect on log V1 (two-compartment models)', 'numeric', 'log scale', 'S3 only'),
    ('ETA_Ka_TRUE', 'True individual random effect on log Ka (oral models)', 'numeric', 'log scale', 'S3 only'),
]
PARAM = ('parameter', 'Population parameter', 'categorical (text)', '',
         'theta_CL = typical clearance (L/h); theta_V = typical volume (L); theta_V1, theta_Q, theta_V2, theta_Ka = two-compartment/absorption parameters (L, L/h, L, 1/h); '
         'omega_<P> = interindividual variance of log <P> (unitless); sigma_prop = proportional residual SD (unitless)')
METHOD = ('method', 'Estimation program or method', 'categorical (text)', '',
          'PKPy2 = joint marginal-likelihood estimation (this work); PKPy = original two-stage PKPy fitting components; '
          'Gaussian two-stage control = two-stage estimator with Gaussian individual likelihood and residual SD fixed at the generating value; '
          'nlmixr2 FOCEi = nlmixr2 first-order conditional estimation with interaction; nlmixr2 SAEM = nlmixr2 stochastic approximation EM; saemix SAEM = saemix R package')
EST_COMMON = [
    METHOD,
    ('fit_status', 'Status reported by the program', 'categorical (text)', '',
     'PKPy2: converged / not converged (independent numerical audit). PKPy and Gaussian control: number of successful individual fits. nlmixr2 and saemix: returned = estimates returned without error; error: <message>'),
    ('accepted', 'Fit included in the numerically accepted population', 'categorical (numeric code)', '',
     '1 = accepted; 0 = not accepted. PKPy2: passed the independent convergence audit. PKPy / Gaussian control: all individual fits successful. nlmixr2 / saemix: finite positive estimates returned'),
    PARAM,
    ('estimate', 'Estimated value on the reporting scale', 'numeric', 'as parameter', ''),
    ('true_value', 'Generating (true) value used for simulation', 'numeric', 'as parameter', ''),
    ('relative_error', '(estimate - true_value) / true_value', 'numeric', 'fraction', 'Multiply by 100 for percent'),
    ('fit_seconds', 'Wall-clock time of the estimation call', 'numeric', 's', 'Descriptive only; programs were run with different parallel settings'),
]
CI = [
    ('ci95_lower', 'Lower limit of the 95% local (Wald) confidence interval', 'numeric', 'as parameter',
     'PKPy2: all parameters, from the full marginal-likelihood information. nlmixr2 / saemix: CL and V only, Wald interval on the log scale from the program-reported standard error. Empty when unavailable'),
    ('ci95_upper', 'Upper limit of the 95% local (Wald) confidence interval', 'numeric', 'as parameter', 'See ci95_lower'),
    ('ci95_covers_truth', 'Whether the 95% interval contains the true value', 'categorical (numeric code)', '',
     '1 = interval contains true_value; 0 = does not; empty = no interval available'),
    ('uncertainty_seconds', 'Wall-clock time of the PKPy2 uncertainty calculation', 'numeric', 's', 'PKPy2 only'),
    ('note', 'Additional information (warnings, fixed terms)', 'text', '', ''),
]
CLIN = [
    ('dataset', 'Clinical dataset', 'categorical (text)', '', 'theophylline = R datasets::Theoph (12 subjects); warfarin = warfarin PK dataset (32 subjects); tobramycin = tobramycin dataset of the PKGPT repository (97 subjects)'),
    ('WT', 'Body weight', 'numeric', 'kg', 'Theophylline and warfarin: covariate on CL and V (reference 70 kg). Tobramycin: covariate on V1 (reference 62 kg)'),
    ('CLCR', 'Creatinine clearance (tobramycin only)', 'numeric', 'mL/min', 'Covariate on CL (reference 58 mL/min); empty for other datasets'),
]
CLIN_EST = [
    ('dataset', 'Clinical dataset', 'categorical (text)', '', 'theophylline; warfarin; tobramycin'),
    ('parameter', 'Parameter label as used in Tables 3-5', 'categorical (text)', '',
     'CL/F (L/h) and V/F (L) standardised to 70 kg; K_a (1/h); ALAG = absorption lag time (h); tobramycin: CL, V1, Q, V2 at CLCR 58 mL/min and WT 62 kg, CLCR exponent = power coefficient of CLCR on CL; ω²(P) = interindividual variance of log P; σ_prop = proportional residual SD'),
    ('software', 'Program providing the estimate', 'categorical (text)', '',
     'NONMEM reference = published expert NONMEM estimates (PKGPT study); other levels as in method'),
    ('estimate', 'Parameter estimate', 'numeric', 'as parameter', ''),
    ('rse_pct', 'Relative standard error reported by, or derived from, the program', 'numeric', '%', 'Empty when the program does not report it'),
    ('relative_difference_from_nonmem_pct', '100 × (estimate - NONMEM) / NONMEM', 'numeric', '%', ''),
    ('note', 'Reason an estimate is unavailable', 'text', '', ''),
]
NUM = [
    ('model', 'Structural model', 'categorical (text)', '', 'as in S1'),
    ('case', 'Parameter/dosing condition index', 'integer', '', '0-11 within each model; conditions include single and repeated doses and equal or nearly equal rate constants'),
    ('prediction_relative_scaled_error', 'Maximum scaled discrepancy of predicted concentrations vs an independent matrix-exponential calculation', 'numeric', 'unitless', ''),
    ('sensitivity_relative_scaled_error', 'Maximum scaled discrepancy of analytical parameter sensitivities vs finite differences of independent predictions', 'numeric', 'unitless', ''),
    ('score_relative_scaled_error', 'Maximum scaled discrepancy of the score', 'numeric', 'unitless', ''),
]
QUAD = [
    ('case', 'Quadrature comparison index', 'integer', '', '0-3'),
    ('sampling', 'Sampling design', 'categorical (text)', '', 'rich; sparse'),
    ('wt_effect_on_cl', 'Whether a body-weight effect on CL was estimated', 'categorical (text)', '', 'estimated; absent'),
    ('reference_ofv', 'OFV at the optimum of the independent quadrature implementation', 'numeric', '-2 log L', ''),
    ('pkpy2_ofv', 'OFV at the PKPy2 optimum', 'numeric', '-2 log L', ''),
    ('node_refinement_ofv_gap', 'OFV change when quadrature nodes increased from 4,097 to 8,193', 'numeric', '-2 log L', ''),
    ('same_point_ofv_gap', 'OFV difference between PKPy2 and the reference at identical parameter values', 'numeric', '-2 log L', ''),
    ('maximum_coordinate_gap', 'Maximum absolute difference between optimized coordinates', 'numeric', 'coordinate units', ''),
    ('hessian_step_relative_gap', 'Relative Hessian change between finite-difference steps', 'numeric', 'fraction', ''),
    ('maximum_relative_coordinate_se_gap', 'Maximum relative difference in coordinate standard errors', 'numeric', 'fraction', ''),
    ('passed', 'All predefined agreement criteria met', 'categorical (numeric code)', '', '1 = passed; 0 = failed'),
]
TIMING = [
    ('model', 'Structural model', 'categorical (text)', '', 'as in S1'),
    ('dose_events', 'Number of dose events in the history', 'integer', '', '1, 10, 100, 1000'),
    ('observations', 'Number of prediction times', 'integer', '', ''),
    ('repetitions_per_batch', 'Prediction calls averaged within a batch', 'integer', '', ''),
    ('method', 'Prediction method', 'categorical (text)', '', 'direct = direct summation over all preceding doses; dispatch = recurrence-based dose-state propagation'),
    ('batch', 'Batch number (method order alternated between batches)', 'integer', '', '1-7'),
    ('seconds_per_call', 'Mean wall-clock time per prediction call after compilation', 'numeric', 's', ''),
]


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    rows = []
    for sheet, variables in [
        ('S1_primary_sim_data', SIM[:1] + [SIM[1], SIM[2], SIM[3], SIM[4]] + EVENT + ETA[:2]),
        ('S2_primary_estimates', [SIM[0], SIM[2], SIM[3], SIM[4]] + EST_COMMON + CI),
        ('S3_additional_sim_data', [SIM[0], SIM[1], SIM[3], SIM[4]] + EVENT + ETA),
        ('S4_additional_estimates', [SIM[0], SIM[1], SIM[3], SIM[4]] + EST_COMMON),
        ('S5_clinical_data', CLIN[:1] + EVENT + CLIN[1:]),
        ('S6_clinical_estimates', CLIN_EST),
        ('S7_prediction_checks', NUM),
        ('S8_quadrature_checks', QUAD),
        ('S9_timing', TIMING),
    ] + list(EXTENDED_CODEBOOK.items()):
        for name, desc, kind, unit, codes in variables:
            rows.append(dict(sheet=sheet, variable=name, description=desc, type=kind, units=unit,
                             codes_or_allowed_values=codes))
    codes = pd.DataFrame(rows)
    numeric_codes = codes[codes['type'] == 'categorical (numeric code)'][['sheet', 'variable', 'codes_or_allowed_values']]
    with pd.ExcelWriter(OUT / 'PKPy2_codebook.xlsx', engine='openpyxl') as xw:
        pd.DataFrame([
            ('Purpose', 'Codebook for the raw-data workbook (Online Resource 2) and its CSV files (Online Resource 3).'),
            ('Article', 'PKPy2: A Python framework for joint population pharmacokinetic estimation and uncertainty assessment'),
            ('Journal', 'Journal of Pharmacokinetics and Pharmacodynamics'),
            ('Authors', 'Hyunseung Kong, Inyoung Kim'),
            ('Corresponding author', 'Inyoung Kim, Department of Defense Science, Korea National Defense University, Nonsan, Republic of Korea; inyoungkim@korea.kr'),
            ('Numerically coded categorical variables', 'EVID, MDV and CMT follow NONMEM event-record conventions; accepted, ci95_covers_truth and passed are 0/1 indicators. Their codes are listed on the sheet "numeric_codes" and in full on "variables".'),
            ('Text-coded categorical variables', 'model, sampling, method, software, fit_status, parameter and dataset are stored as text labels; their levels are defined on "variables".'),
            ('Missing values', 'Empty cells denote values that do not apply (e.g. DV on dose records) or are not reported by a program; reasons are given in the note columns.'),
        ], columns=['Item', 'Description']).to_excel(xw, sheet_name='README', index=False)
        numeric_codes.to_excel(xw, sheet_name='numeric_codes', index=False)
        codes.to_excel(xw, sheet_name='variables', index=False)
    codes.to_csv(OUT / 'PKPy2_codebook.csv', index=False, encoding='utf-8')
    print(len(codes), 'variables;', len(numeric_codes), 'numerically coded categorical variables')


if __name__ == '__main__':
    main()
