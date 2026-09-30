"""Collect the results of the event-record (general engine) validation for the manuscript.

Reads output/pkpy2_extended_validation/* and writes docs/pkpy2_paper/extended_numbers.json.
Every number quoted in the manuscript sections on the event-record interface comes from here.
"""
from pathlib import Path
import csv
import json
import math

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
V = ROOT / 'output/pkpy2_extended_validation'
OUT = ROOT / 'docs/pkpy2_paper/extended_numbers.json'

ODE = {'mm_iv_multiple', 'mm_oral_ss', 'idr1', 'idr2', 'idr3', 'idr4', 'tmdd_full', 'tmdd_qss'}


def load(path):
    path = V / path
    return json.loads(path.read_text(encoding='utf-8')) if path.exists() else None


def sci(x, digits=1):
    m, e = f'{x:.{digits}e}'.split('e')
    return f'{m}×10{int(e)}'


def main():
    n = {}
    pred = load('prediction_agreement.json')
    if pred:
        lin = [r for r in pred if r['scenario'] not in ODE]
        ode = [r for r in pred if r['scenario'] in ODE]
        n.update(pred_scenarios=len(pred), pred_values=sum(r['values'] for r in pred), pred_linear_n=len(lin),
                 pred_ode_n=len(ode), pred_linear_max=max(r['max_relative_difference'] for r in lin),
                 pred_ode_max=max(r['max_relative_difference'] for r in ode))
    lik = load('likelihood_agreement.json')
    if lik:
        n.update(lik_models=len(lik), lik_max_abs_diff=max(abs(r['difference']) for r in lik),
                 lik_max_bank_range=max(r['bank_range'] for r in lik),
                 lik_censored=next(r['censored'] for r in lik if r['model'] == 'iov_blq_lognormal'))
    eng = load('general_vs_classic.json')
    if eng:
        n.update(engine_datasets=len(eng), engine_converged=sum(r['general_status'] == 'converged' for r in eng),
                 engine_max_abs_ofv_diff=max(abs(r['ofv_difference']) for r in eng),
                 engine_max_param_pct=max(r['max_abs_relative_difference_pct'] for r in eng))
    classic = load('classic_unchanged.json')
    if classic:
        n.update(classic_identical=all(r['ofv_difference'] == 0 and r['max_relative_estimate_difference'] == 0
                                       for r in classic), classic_checked=len(classic))
    diag = load('diagnostics_agreement.json')
    if diag:
        n.update(diag_pred_max=diag['PRED']['max_abs_difference'], diag_ipred_max=diag['IPRED']['max_abs_difference'],
                 diag_iwres_max=diag['IWRES']['max_abs_difference'], diag_cwres_max=diag['CWRES']['max_abs_difference'],
                 diag_ebe_max=diag['EBE']['max_abs_difference'], diag_npde_corr=diag['NPDE']['correlation'],
                 diag_npde_pkg_max=diag['NPDE_vs_npde_package_same_replicates']['max_abs_difference'],
                 diag_n=diag['CWRES']['n'])
    cmp = load('vs_nlmixr2.json')
    if cmp:
        rows = []
        for r in cmp:
            rel = []
            for group in ('theta', 'coefficients', 'omega', 'sigma'):
                for k, v in r['pkpy2'][group].items():
                    ref = r.get('nlmixr2', {}).get(group, {}).get(k)
                    if ref is not None and ref != 0 and v != 0:
                        rel.append(abs(v / ref - 1) * 100)
            data = list(csv.DictReader((V / 'vs_nlmixr2' / f"{r['scenario']}.csv").open()))
            obs = [q for q in data if q['EVID'] == '0']
            rows.append(dict(scenario=r['scenario'], status=r['pkpy2_status'], ofv=r['pkpy2_ofv'],
                             observations=len(obs), censored=sum(q.get('CENS', '0') in ('1', '1.0') for q in obs),
                             exact_ofv_difference=r.get('exact_ofv_difference'),
                             max_relative_difference_pct=max(rel) if rel else None,
                             pkpy2_seconds=r['pkpy2_seconds'], nlmixr2_seconds=r.get('nlmixr2_seconds')))
        n['vs_nlmixr2'] = rows
    wpd = load('warfarin_pkpd/exact_ofv.json')
    if wpd:
        pk = next(r for r in wpd if r['source'] == 'PKPy2')
        fitw = load('warfarin_pkpd/pkpy2_fit.json')
        saem = load('warfarin_pkpd/nlmixr2_saem.json')
        focei = load('warfarin_pkpd/nlmixr2_focei.json')
        expit = lambda v: 1. / (1. + math.exp(-v))
        n['warfarin_pkpd_estimates'] = dict(imax=fitw['theta']['IMAX'], imax_saem=expit(saem['theta']['temax']),
                                            imax_focei=expit(focei['theta']['temax']),
                                            cl=fitw['theta']['CL'], cl_saem=math.exp(saem['theta']['tcl']),
                                            v=fitw['theta']['V'], v_saem=math.exp(saem['theta']['tv']))
        n['warfarin_pkpd'] = dict(pkpy2_ofv=pk['ofv'], status=pk['status'],
                                  focei_minus_pkpy2=next((r['ofv'] - pk['ofv'] for r in wpd if r['source'] == 'nlmixr2 focei'), None),
                                  saem_minus_pkpy2=next((r['ofv'] - pk['ofv'] for r in wpd if r['source'] == 'nlmixr2 saem'), None))
    cal = load('calibration_summary.json')
    if cal:
        n['calibration'] = {k: dict(replicates=v['replicates'], npde_mean=v['npde_mean'], npde_variance=v['npde_variance'],
                                    ks_rejections=v['npde_ks_rejections_5pct'],
                                    vpc_coverage={o: c['overall'] for o, c in v['vpc_coverage'].items()})
                            for k, v in cal.items()}
    rec = load('recovery_summary.json')
    if rec:
        out = {}
        for k, v in rec.items():
            groups = {g: [e for e in v['estimates'] if e['quantity'].startswith(prefix)]
                      for g, prefix in (('theta', 'theta:'), ('coefficients', 'coefficients:'), ('variance', ('omega', 'iov')),
                                        ('sigma', 'sigma:'))}
            row = dict(replicates=v['replicates'], converged=v['converged'], median_seconds=v['median_seconds'])
            for g, rows in groups.items():
                if rows:
                    bias = [e['relative_bias_pct'] for e in rows]
                    row[g] = dict(bias_min=min(bias), bias_max=max(bias), max_abs_bias=max(abs(b) for b in bias),
                                  rmse_max=max(e['relative_rmse_pct'] for e in rows),
                                  worst=max(rows, key=lambda e: abs(e['relative_bias_pct']))['quantity'])
            out[k] = row
        n['recovery'] = out
    tools = load('tools_summary.json')
    if tools:
        n['tools'] = tools
    OUT.write_text(json.dumps(n, indent=1, default=float), encoding='utf-8')
    print(json.dumps(n, indent=1, default=float))


if __name__ == '__main__':
    main()
