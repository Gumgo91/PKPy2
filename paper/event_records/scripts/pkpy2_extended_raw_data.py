"""Raw-data sheets and codebook entries for the event-record interface (S10-S18).

Used by build_pkpy2_peerj_raw_data.py and build_pkpy2_peerj_codebook.py.
"""
from pathlib import Path
import csv
import json
import math

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
V = ROOT / 'output/pkpy2_extended_validation'
ODE = {'mm_iv_multiple', 'mm_oral_ss', 'idr1', 'idr2', 'idr3', 'idr4', 'tmdd_full', 'tmdd_qss'}


def read(path):
    return json.loads((V / path).read_text(encoding='utf-8'))


def flatten(d):
    out = {}
    for group in ('theta', 'coefficients', 'omega', 'omega_covariance', 'iov', 'sigma'):
        for k, v in d.get(group, {}).items():
            out[f'{group}:{k}'] = v
    return out


def warfarin_nlmixr2(method):
    """nlmixr2 warfarin PK/PD estimates on the PKPy2 reporting scale (mapping of validate_pkpy2_warfarin_pkpd.py)."""
    r = read(f'warfarin_pkpd/nlmixr2_{method}.json')
    th, om = r['theta'], r['omega']
    names = dict(KTR='tktr', Ka='tka', CL='tcl', V='tv', R0='te0', KOUT='tkout', IMAX='temax', IC50='tec50')
    etas = dict(KTR='eta.ktr', Ka='eta.ka', CL='eta.cl', V='eta.v', R0='eta.e0', KOUT='eta.kout', IMAX='eta.emax',
                IC50='eta.ec50')
    out = {f'theta:{k}': 1 / (1 + math.exp(-th[v])) if k == 'IMAX' else math.exp(th[v]) for k, v in names.items()}
    out.update({f'omega:{k}': om[v] for k, v in etas.items()})
    out.update({'sigma:CP:proportional': th['prop.err'], 'sigma:CP:additive': th['pkadd.err'],
                'sigma:R:additive': th['pdadd.err']})
    return out


def sheets():
    s = {}
    pv = pd.read_csv(V / 'prediction_values.csv')
    pv.insert(1, 'model_type', ['ODE' if x in ODE else 'linear' for x in pv['scenario']])
    s['S10_event_prediction_checks'] = pv

    lik = pd.DataFrame([dict(check='quadrature', item=r['model'], subjects=r['subjects'], observations=r['observations'],
                             censored=r['censored'], pkpy2_ofv_bank1=r['pkpy2_ofv'][0], pkpy2_ofv_bank2=r['pkpy2_ofv'][1],
                             reference_ofv=r['reference_ofv'], ofv_difference=r['difference'])
                        for r in read('likelihood_agreement.json')]
                       + [dict(check='compact_interface', item=r['dataset'], status=r['general_status'],
                               reference_ofv=r['classic_ofv'], pkpy2_ofv_bank1=r['general_ofv'], ofv_difference=r['ofv_difference'],
                               max_relative_estimate_difference_pct=r['max_abs_relative_difference_pct'])
                          for r in read('general_vs_classic.json')])
    s['S11_event_likelihood_checks'] = lik

    paired = pd.read_csv(V / 'diagnostics/paired.csv')
    s['S12_diagnostic_checks'] = paired

    rows = []
    for r in read('vs_nlmixr2.json'):
        a, b, t = flatten(r['pkpy2']), flatten(r.get('nlmixr2', {})), flatten(r['truth'])
        for k in a:
            rows.append(dict(analysis=r['scenario'], parameter=k, true_value=t.get(k), pkpy2=a[k], nlmixr2_focei=b.get(k)))
        rows.append(dict(analysis=r['scenario'], parameter='exact_ofv', pkpy2=r['pkpy2_ofv'], nlmixr2_focei=r['exact_ofv_at_nlmixr2']))
    wfit = read('warfarin_pkpd/pkpy2_fit.json')
    focei, saem = warfarin_nlmixr2('focei'), warfarin_nlmixr2('saem')
    for k, v in flatten(wfit).items():
        rows.append(dict(analysis='warfarin_pkpd', parameter=k, pkpy2=v, nlmixr2_focei=focei.get(k), nlmixr2_saem=saem.get(k)))
    exact = {r['source']: r['ofv'] for r in read('warfarin_pkpd/exact_ofv.json')}
    rows.append(dict(analysis='warfarin_pkpd', parameter='exact_ofv', pkpy2=exact['PKPy2'],
                     nlmixr2_focei=exact['nlmixr2 focei'], nlmixr2_saem=exact['nlmixr2 saem']))
    s['S13_event_comparisons'] = pd.DataFrame(rows)

    rows = []
    for design in ('complex_linear', 'pkpd_idr'):
        for p in sorted((V / 'recovery' / design).glob('rep_[0-9][0-9][0-9].json')):
            r = json.loads(p.read_text())
            for k, v in flatten(r.get('estimates', {})).items():
                rows.append(dict(design=design, replicate=r['replicate'], status=r['status'], converged=int(r['converged']),
                                 ofv=r.get('ofv'), seconds=r['seconds'], parameter=k, estimate=v))
    s['S14_event_recovery'] = pd.DataFrame(rows)

    rows = []
    for design in ('complex_linear', 'pkpd_idr'):
        for p in sorted((V / 'calibration' / design).glob('rep_*.json')):
            r = json.loads(p.read_text())
            row = dict(design=design, replicate=r['replicate'], censored=r['censored'], n_npde=r['n_npde'],
                       npde_mean=r['npde_mean'], npde_variance=r['npde_variance'], npde_ks_p=r['npde_ks_p'],
                       cwres_mean=r['cwres_mean'], cwres_variance=r['cwres_variance'])
            row.update({f'vpc_within_{o}': v for o, v in r['vpc_within'].items()})
            rows.append(row)
    s['S15_event_calibration'] = pd.DataFrame(rows)

    th = json.loads((V / 'tools/theophylline_intervals.json').read_text())
    rows = []
    for key, w in th['wald'].items():
        for method in ('wald', 'sandwich', 'profile', 'bootstrap', 'sir'):
            if method == 'profile':
                iv = th['profile'].get(key, {}).get('interval')
            elif method == 'bootstrap':
                iv = th.get('bootstrap', {}).get('summary', {}).get(key, {}).get('interval')
            elif method == 'sir':
                iv = th['sir']['summary'].get(key, {}).get('interval')
            else:
                iv = th[method].get(key, {}).get('interval')
            rows.append(dict(analysis='theophylline_intervals', parameter=key, method=method, estimate=w['estimate'],
                             lower=None if not iv else iv[0], upper=None if not iv else iv[1]))
    for p in sorted((V / 'tools/scm').glob('rep_*.json')):
        r = json.loads(p.read_text())
        rows.append(dict(analysis='stepwise_covariates', replicate=r['replicate'],
                         selected=';'.join(r['selected']) if r.get('selected') is not None else None,
                         seconds=r['seconds']))
    s['S16_interval_and_scm'] = pd.DataFrame(rows)

    diag = read('warfarin_pkpd/pkpy2_diagnostics.json')
    rows = []
    for out, e in diag['vpc'].items():
        for b, t in enumerate(e['bin_time']):
            for j, q in enumerate(e['quantiles']):
                rows.append(dict(output=out, record='percentile', bin=b + 1, bin_time=t, quantile=q,
                                 observed=e['observed'][b][j], simulated_lower=e['lower'][b][j],
                                 simulated_median=e['median'][b][j], simulated_upper=e['upper'][b][j]))
        for t, y in zip(diag['observations'][out]['time'], diag['observations'][out]['dv']):
            rows.append(dict(output=out, record='observation', bin_time=t, observed=y))
    s['S17_warfarin_pkpd_vpc'] = pd.DataFrame(rows)

    frames = []
    for name in ('infusion_block_covariates', 'oral_blq_m3', 'lognormal_residual', 'michaelis_menten'):
        df = pd.read_csv(V / 'vs_nlmixr2' / f'{name}.csv', na_values='.')
        df.insert(0, 'dataset', name)
        frames.append(df)
    df = pd.read_csv(V / 'warfarin_pkpd/warfarin_pkpd.csv', na_values='.')
    df.insert(0, 'dataset', 'warfarin_pkpd')
    frames.append(df)
    s['S18_event_datasets'] = pd.concat(frames, ignore_index=True)
    s['S19_example_diagnostics'] = pd.read_csv(V / 'example_theophylline/diagnostics.csv')
    return s


README = [
    ('S10_event_prediction_checks', 'PKPy2 predictions and rxode2 values for the 24 event-record scenarios (Figure 6a, Table S18).'),
    ('S11_event_likelihood_checks', 'Marginal OFV against independent quadrature and refits through the event-record interface against the compact interface (Figure 6b, Table S19).'),
    ('S12_diagnostic_checks', 'Paired PKPy2 and nlmixr2 diagnostics of the theophylline model at identical parameters (Figure 6c, Table S19).'),
    ('S13_event_comparisons', 'Estimates and exact OFVs of PKPy2 and nlmixr2 for the simulated comparison datasets and the warfarin PK/PD model (Figure 6d, Tables S20-S21).'),
    ('S14_event_recovery', 'Per-replicate PKPy2 estimates in the recovery simulations (Table S22).'),
    ('S15_event_calibration', 'Per-replicate NPDE, CWRES and VPC summaries at the true parameters (Table S23).'),
    ('S16_interval_and_scm', 'Theophylline 95% intervals by method and stepwise covariate selections (Figure 7c, Tables S24-S25).'),
    ('S17_warfarin_pkpd_vpc', 'Visual predictive check of the warfarin PK/PD fit (Figure 7a, b).'),
    ('S18_event_datasets', 'Simulated comparison datasets and the warfarin PK/PD dataset in NONMEM event format.'),
    ('S19_example_diagnostics', 'Diagnostics of the theophylline example analysis plotted in Online Resource 1, Fig. S1 (Listing 2).'),
]

CODEBOOK = {
    'S10_event_prediction_checks': [
        ('scenario', 'Prediction scenario', 'categorical (text)', '', 'See Online Resource 1, Table S18'),
        ('model_type', 'System type', 'categorical (text)', '', 'linear; ODE'),
        ('ID', 'Subject', 'integer', '', ''), ('TIME', 'Time', 'numeric', 'h', ''),
        ('output', 'Model output', 'categorical (numeric code)', '', '1 = first output (concentration); 2 = second output (response or metabolite)'),
        ('pkpy2', 'PKPy2 prediction', 'numeric', 'output units', ''), ('rxode2', 'rxode2 prediction', 'numeric', 'output units', ''),
        ('scaled_difference', '|PKPy2 - rxode2| divided by max(|rxode2|, 1e-6 × largest value of the subject)', 'numeric', 'unitless', '')],
    'S11_event_likelihood_checks': [
        ('check', 'Comparison', 'categorical (text)', '', 'quadrature = independent Gauss-Hermite quadrature at fixed parameters; compact_interface = refit through the event-record interface vs compact-interface fit'),
        ('item', 'Model or dataset', 'text', '', ''), ('subjects', 'Subjects', 'integer', '', ''),
        ('observations', 'Observations', 'integer', '', ''), ('censored', 'Censored observations', 'integer', '', ''),
        ('status', 'PKPy2 convergence status', 'categorical (text)', '', 'converged; partial'),
        ('pkpy2_ofv_bank1', 'PKPy2 OFV (bank 1 or refit)', 'numeric', '-2 log L', ''),
        ('pkpy2_ofv_bank2', 'PKPy2 OFV (bank 2)', 'numeric', '-2 log L', ''),
        ('reference_ofv', 'Reference OFV (quadrature or compact interface)', 'numeric', '-2 log L', ''),
        ('ofv_difference', 'PKPy2 minus reference', 'numeric', '-2 log L', ''),
        ('max_relative_estimate_difference_pct', 'Maximum relative difference of the estimates', 'numeric', '%', '')],
    'S12_diagnostic_checks': [
        ('quantity', 'Diagnostic', 'categorical (text)', '', 'PRED; IPRED; IWRES; CWRES (standardized by its own variance); NPDE (upper Cholesky factor)'),
        ('pkpy2', 'PKPy2 value', 'numeric', '', ''), ('nlmixr2', 'nlmixr2 value', 'numeric', '', '')],
    'S13_event_comparisons': [
        ('analysis', 'Dataset', 'categorical (text)', '', 'infusion_block_covariates; oral_blq_m3; lognormal_residual; michaelis_menten; warfarin_pkpd'),
        ('parameter', 'Parameter (group:name) or exact OFV', 'text', '', 'theta = typical value; coefficients = covariate coefficient; omega = variance; omega_covariance = covariance; iov = interoccasion variance; sigma = residual SD'),
        ('true_value', 'Simulation value', 'numeric', 'as parameter', ''), ('pkpy2', 'PKPy2 estimate or OFV', 'numeric', 'as parameter', ''),
        ('nlmixr2_focei', 'nlmixr2 FOCEi estimate or exact OFV at its estimates', 'numeric', 'as parameter', ''),
        ('nlmixr2_saem', 'nlmixr2 SAEM estimate or exact OFV at its estimates', 'numeric', 'as parameter', 'warfarin_pkpd only')],
    'S14_event_recovery': [
        ('design', 'Simulation design', 'categorical (text)', '', 'complex_linear = two-compartment steady-state infusion model; pkpd_idr = indirect-response PK/PD model'),
        ('replicate', 'Replicate', 'integer', '', ''), ('status', 'PKPy2 status', 'categorical (text)', '', 'converged; partial; error'),
        ('converged', 'Convergence', 'categorical (numeric code)', '', '1 = converged; 0 = not converged'),
        ('ofv', 'OFV', 'numeric', '-2 log L', ''), ('seconds', 'Fit time', 'numeric', 's', ''),
        ('parameter', 'Parameter (group:name)', 'text', '', 'as in S13'), ('estimate', 'Estimate', 'numeric', 'as parameter', '')],
    'S15_event_calibration': [
        ('design', 'Simulation design', 'categorical (text)', '', 'as in S14'), ('replicate', 'Replicate', 'integer', '', ''),
        ('censored', 'Censored observations', 'integer', '', ''), ('n_npde', 'NPDE values', 'integer', '', ''),
        ('npde_mean', 'Mean NPDE', 'numeric', '', ''), ('npde_variance', 'NPDE variance', 'numeric', '', ''),
        ('npde_ks_p', 'Kolmogorov-Smirnov p value of NPDE vs N(0, 1)', 'numeric', '', ''),
        ('cwres_mean', 'Mean CWRES', 'numeric', '', ''), ('cwres_variance', 'CWRES variance', 'numeric', '', ''),
        ('vpc_within_CP', 'Fraction of observed VPC percentiles within the 95% intervals, concentration', 'numeric', 'fraction', ''),
        ('vpc_within_R', 'Same for the PD response', 'numeric', 'fraction', 'pkpd_idr only')],
    'S16_interval_and_scm': [
        ('analysis', 'Analysis', 'categorical (text)', '', 'theophylline_intervals; stepwise_covariates'),
        ('parameter', 'Reporting quantity', 'text', '', 'theta:CL, theta:V, theta:Ka, omega:<P> (variance), sigma:CP:proportional'),
        ('method', 'Interval method', 'categorical (text)', '', 'wald; sandwich; profile; bootstrap; sir'),
        ('estimate', 'Estimate', 'numeric', 'as parameter', ''), ('lower', '95% lower limit', 'numeric', 'as parameter', ''),
        ('upper', '95% upper limit', 'numeric', 'as parameter', ''), ('replicate', 'SCM replicate', 'integer', '', ''),
        ('selected', 'Selected covariate effects', 'text', '', 'semicolon-separated labels parameter~covariate[form]'),
        ('seconds', 'SCM run time', 'numeric', 's', '')],
    'S17_warfarin_pkpd_vpc': [
        ('output', 'Output', 'categorical (text)', '', 'CP = warfarin concentration (mg/L); R = prothrombin complex activity (%)'),
        ('record', 'Row type', 'categorical (text)', '', 'percentile; observation'),
        ('bin', 'Time bin', 'integer', '', ''), ('bin_time', 'Median time of the bin, or observation time', 'numeric', 'h', ''),
        ('quantile', 'Percentile', 'numeric', 'fraction', '0.05; 0.5; 0.95'),
        ('observed', 'Observed percentile or observation', 'numeric', 'output units', ''),
        ('simulated_lower', '2.5th percentile of the simulated percentile', 'numeric', 'output units', ''),
        ('simulated_median', 'Median of the simulated percentile', 'numeric', 'output units', ''),
        ('simulated_upper', '97.5th percentile of the simulated percentile', 'numeric', 'output units', '')],
    'S18_event_datasets': [
        ('dataset', 'Dataset', 'categorical (text)', '', 'as in S13'),
        ('ID', 'Subject', 'integer', '', ''), ('TIME', 'Time', 'numeric', 'h', ''),
        ('EVID', 'Event identifier (NONMEM convention)', 'categorical (numeric code)', '', '0 = observation record; 1 = dose record'),
        ('AMT', 'Dose amount', 'numeric', 'mg', '0 on observation records'),
        ('CMT', 'Compartment receiving the dose or observed', 'categorical (numeric code)', '',
         '1 = central compartment; in oral_blq_m3, 1 = depot (doses) and 2 = central (observations)'),
        ('RATE', 'Infusion rate', 'numeric', 'mg/h', '0 = bolus or first-order input'),
        ('II', 'Dosing interval of a steady-state dose', 'numeric', 'h', '0 = no steady-state dose'),
        ('SS', 'Steady-state flag', 'categorical (numeric code)', '', '1 = dose given at steady state; 0 = other records'),
        ('DV', 'Observation', 'numeric', 'mg/L; % for DVID 2', 'empty on dose records; the quantification limit when CENS = 1'),
        ('CENS', 'Censoring flag', 'categorical (numeric code)', '', '1 = below the quantification limit; 0 = quantified observation'),
        ('WT', 'Body weight', 'numeric', 'kg', ''),
        ('SEX', 'Sex', 'categorical (numeric code)', '',
         'warfarin_pkpd: 1 = male, 0 = female; infusion_block_covariates: simulated binary covariate with reference category 0'),
        ('DVID', 'Observed output', 'categorical (numeric code)', '', '1 = warfarin concentration (mg/L); 2 = prothrombin complex activity (%)'),
        ('AGE', 'Age', 'numeric', 'years', '')],
    'S19_example_diagnostics': [
        ('ID', 'Subject', 'integer', '', ''), ('TIME', 'Time after dose', 'numeric', 'h', ''),
        ('DV', 'Observed theophylline concentration', 'numeric', 'mg/L', ''),
        ('CENS', 'Censoring flag', 'categorical (numeric code)', '', '1 = below the quantification limit; 0 = quantified observation'),
        ('PRED', 'Population prediction (random effects zero)', 'numeric', 'mg/L', ''),
        ('IPRED', 'Individual prediction at the conditional modes', 'numeric', 'mg/L', ''),
        ('IWRES', 'Individual weighted residual', 'numeric', 'unitless', ''),
        ('CWRES', 'Conditional weighted residual', 'numeric', 'unitless', ''),
        ('NPDE', 'Normalized prediction distribution error', 'numeric', 'unitless', '')],
}
