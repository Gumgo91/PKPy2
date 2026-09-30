"""Runnable PKPy2 workflow on synthetic data; no clinical reference files needed.

Run with Python 3.13 after installing the supplied PKPy2 wheel or source.
This example illustrates fixed/free terms and result handling, not a validation
experiment or a recommendation to fix volume in clinical analyses.
"""
import argparse
import json
from pathlib import Path
import time
import numpy as np
from pkpy2 import Subject, ModelSpec, Parameter as P, Covariate, fit, load_fit


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output-dir', type=Path, default=Path('example_output'))
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    destination = args.output_dir / 'fit.json'
    if destination.exists():
        raise FileExistsError(f'Refusing to overwrite {destination}')

    rng = np.random.default_rng(20260914)
    times = np.array([.25, .5, 1., 2., 4., 6., 8., 12., 18., 24.])
    weights = np.linspace(50., 90., 30)
    subjects = []
    for sid, weight in enumerate(weights, start=1):
        clearance = 4. * (weight / 70.)**.75 * np.exp(rng.normal(0., .3))
        mean = 100. / 40. * np.exp(-clearance / 40. * times)
        observations = mean * (1. + rng.normal(0., .15, len(times)))
        subjects.append(Subject(sid, times.copy(), observations, dose=100.,
                                covariates={'WT': float(weight)}))

    specification = ModelSpec(
        '1cmt_iv',
        theta={'CL': P(3.), 'V': P(40., fixed=True)},
        omega={'CL': P(.08)},
        sigma_prop=P(.2), sigma_add=P(0., fixed=True),
        covariates=(Covariate('CL', 'WT', 70., P(.75, fixed=True)),),
    )
    started = time.perf_counter()
    result = fit(subjects, specification, seed=20260915, workers=2,
        saem_options={'cpu_budget_seconds': 60.},
        refinement_options={'cpu_budget_seconds': 300., 'max_stages': 90,
                            'analytic_non_eta': True})
    print('Fit status:', result.status, 'audit passed:', result.audit.get('passed'))
    print('Estimates:', result.theta, 'IIV variances:', result.omega)
    if result.converged:
        uncertainty = result.uncertainty(seed=20260916, workers=2, power=16, step=.002)
        print('Uncertainty status:', uncertainty['status'])
        if uncertainty['status'] == 'computed':
            for row in uncertainty['information']['data']['intervals']:
                print(row['coordinate'], row['estimate'], row['interval'])
    else:
        print('Unresolved:', result.estimation['refinement_message'])

    # Save even an unresolved fit; never replace its status with success.
    result.save(destination)
    restored = load_fit(destination, subjects)
    assert restored.theta == result.theta and restored.omega == result.omega
    assert restored.audit == result.audit
    assert restored.uncertainty_report == result.uncertainty_report
    assert restored.theta['V'] == 40. and restored.coefficients == [.75]
    assert restored.sigma['sigma_add'] == 0.
    summary = dict(fit_status=result.status, audit_passed=result.audit.get('passed'),
        uncertainty_status=(result.uncertainty_report or {}).get('status'),
        fixed_terms_preserved=True, save_restore_identical=True,
        n_subjects=len(subjects), observations=sum(len(s.obs) for s in subjects),
        total_seconds=time.perf_counter()-started,
        purpose='Executable usage demonstration, excluded from study performance summaries')
    (args.output_dir / 'summary.json').write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary))


if __name__ == '__main__':
    main()
