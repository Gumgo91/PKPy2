"""Compare PKPy2 general-engine predictions with rxode2 for dosing and model features.

Usage:
  python validate_pkpy2_extended_predictions.py write    # scenario data -> output/pkpy2_extended_validation/predictions
  Rscript validate_pkpy2_extended_predictions.R          # rxode2 predictions
  python validate_pkpy2_extended_predictions.py compare  # agreement table (JSON)

Every scenario is a NONMEM-style CSV read identically by both programs (rxode2
expands ADDL itself). rxode2 solves each model as ODEs with tolerances 1e-12.
"""
from pathlib import Path
import csv
import json
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'packages/pkpy2/src'))
from pkpy2 import structures as S          # noqa: E402
from pkpy2.data import read_nonmem          # noqa: E402
from pkpy2._general.packing import predict  # noqa: E402

OUT = ROOT / 'output/pkpy2_extended_validation/predictions'
COLUMNS = ['ID', 'TIME', 'EVID', 'AMT', 'CMT', 'RATE', 'II', 'SS', 'ADDL', 'DV', 'DVID', 'WT']


def obs(times, cmt=1):
    return [dict(TIME=t, EVID=0, CMT=cmt) for t in times]


def dose(t, amt, cmt=1, **kw):
    return dict(TIME=t, EVID=1, AMT=amt, CMT=cmt, **kw)


GRID = [.1, .25, .5, 1, 1.5, 2, 3, 4, 6, 8, 10, 12, 16, 20, 24, 30, 36, 48, 60, 72]


def scenarios():
    s = []
    base1 = dict(CL=3.1, V=42.)
    s.append(dict(name='iv1_infusion', r_model='pk1iv', build=lambda: S.pk(1), params=base1,
                  subjects=[[dose(0, 500, RATE=250), dose(24, 500, RATE=100)] + obs(GRID)]))
    s.append(dict(name='iv2_infusion_ss', r_model='pk2iv', build=lambda: S.pk(2),
                  params=dict(CL=4.2, V1=18., Q=6.3, V2=37.),
                  subjects=[[dose(0, 400, RATE=200, SS=1, II=12)] + obs([.5, 1, 2, 3, 5, 8, 11.9, 13, 16, 24, 30])]))
    s.append(dict(name='oral1_ss_lag_f', r_model='pk1oral',
                  build=lambda: S.pk(1, 'first_order', lag=True, bioavailability=True),
                  params=dict(CL=2.4, V=35., Ka=1.3, ALAG=.6, F=.7),
                  subjects=[[dose(0, 200, SS=1, II=24)] + obs([.3, .8, 1, 2, 4, 8, 12, 20, 23.9, 24.5, 26, 30, 36, 47])]))
    s.append(dict(name='oral2_addl_lag', r_model='pk2oral', build=lambda: S.pk(2, 'first_order', lag=True),
                  params=dict(CL=5.3, V1=25., Q=8.1, V2=60., Ka=.8, ALAG=.4),
                  subjects=[[dose(0, 100, ADDL=6, II=12)] + obs(GRID + [84, 96])]))
    s.append(dict(name='iv3_bolus_infusion', r_model='pk3iv', build=lambda: S.pk(3),
                  params=dict(CL=6.2, V1=12., Q2=9.5, V2=30., Q3=1.2, V3=85.),
                  subjects=[[dose(0, 1000), dose(12, 300, RATE=150)] + obs(GRID + [96, 120])]))
    s.append(dict(name='zero_order_d1', r_model='pk1iv', build=lambda: S.pk(1, 'zero_order'),
                  params=dict(CL=3.1, V=42., D1=3.5),
                  subjects=[[dose(0, 250, RATE=-2), dose(24, 250, RATE=-2)] + obs(GRID)]))
    s.append(dict(name='modeled_rate_r1', r_model='pk1iv', build=lambda: S.pk(1, infusion_parameter='rate'),
                  params=dict(CL=3.1, V=42., R1=120.),
                  subjects=[[dose(0, 300, RATE=-1)] + obs(GRID)]))
    s.append(dict(name='transit3', r_model='transit3', build=lambda: S.pk(1, 'transit', transit=3),
                  params=dict(CL=2.9, V=30., Ka=1.1, MTT=2.4),
                  subjects=[[dose(0, 100)] + obs(GRID)]))
    s.append(dict(name='reset_evid3_evid4', r_model='pk1oral', build=lambda: S.pk(1, 'first_order', lag=True, bioavailability=True),
                  params=dict(CL=2.4, V=35., Ka=1.3, ALAG=0., F=1.),
                  subjects=[[dose(0, 100, cmt=1)] + obs([1, 2, 6, 11]) + [dict(TIME=12, EVID=3)] + obs([13, 14]) +
                            [dict(TIME=20, EVID=4, AMT=50, CMT=1)] + obs([21, 24, 30])]))
    s.append(dict(name='timevarying_wt', r_model='pk1iv_tv', build=lambda: S.pk(1), params=None,
                  subjects=[[dict(dose(0, 500), WT=60)] + [dict(o, WT=60) for o in obs([1, 4, 8, 12, 23])] +
                            [dict(TIME=24, EVID=2, WT=90)] + [dict(o, WT=90) for o in obs([26, 30, 40])] +
                            [dict(dose(48, 500), WT=75)] + [dict(o, WT=75) for o in obs([49, 54, 60, 72])]]))
    s.append(dict(name='effect_sigmoid_emax', r_model='effect_sig',
                  build=lambda: S.pk(1, 'first_order', pd='sigmoid_emax', effect='compartment'),
                  params=dict(CL=2.2, V=28., Ka=1.5, E0=5., EMAX=40., EC50=1.2, GAMMA=2.3, KE0=.35),
                  outputs=2, subjects=[[dose(0, 150)] + obs(GRID)]))
    s.append(dict(name='parent_metabolite', r_model='parent_met', build=lambda: S.parent_metabolite(1),
                  params=dict(CL=5., V=40., FM=.6, CLM=2.5, VM=25.), outputs=2,
                  subjects=[[dose(0, 400)] + obs(GRID)]))
    s.append(dict(name='ss2_superposition', r_model='pk2iv', build=lambda: S.pk(2),
                  params=dict(CL=4.2, V1=18., Q=6.3, V2=37.),
                  subjects=[[dose(0, 300), dose(6, 200, SS=2, II=12)] + obs([1, 5, 6.5, 8, 12, 17.9, 20, 30])]))
    s.append(dict(name='continuous_infusion_ss', r_model='pk2iv', build=lambda: S.pk(2),
                  params=dict(CL=4.2, V1=18., Q=6.3, V2=37.),
                  subjects=[[dict(TIME=0, EVID=1, AMT=0, CMT=1, RATE=20, SS=1, II=0)] + obs([.5, 2, 10, 24])]))
    s.append(dict(name='infusion_bioavailability', r_model='pk1iv', build=lambda: S.pk(1, bioavailability=True),
                  params=dict(CL=3.1, V=42., F=.6),
                  subjects=[[dose(0, 600, RATE=100)] + obs([1, 3, 3.5, 4, 5, 6, 8, 12])]))
    s.append(dict(name='ss_infusion_lag_tail', r_model='pk1iv', build=lambda: S.pk(1, lag=True),
                  params=dict(CL=3.1, V=42., ALAG=2.),
                  subjects=[[dose(0, 400, RATE=40, SS=1, II=12)] + obs([.5, 1, 1.9, 2.5, 6, 11, 12.5, 14, 20])]))
    s.append(dict(name='mm_iv_multiple', r_model='mm1iv', build=lambda: S.michaelis_menten(1),
                  params=dict(V=20., VMAX=40., KM=3.), subjects=[[dose(0, 500), dose(12, 300)] + obs(GRID)]))
    s.append(dict(name='mm_oral_ss', r_model='mm1oral', build=lambda: S.michaelis_menten(1, 'first_order'),
                  params=dict(V=20., Ka=1.2, VMAX=40., KM=3.),
                  subjects=[[dose(0, 200, SS=1, II=12)] + obs([.5, 1, 2, 4, 8, 11.9, 13, 18, 24])]))
    for kind in (1, 2, 3, 4):
        extra = dict(IMAX=.8, IC50=1.5) if kind in (1, 2) else dict(EMAX=2.5, EC50=1.5)
        s.append(dict(name=f'idr{kind}', r_model=f'idr{kind}', build=(lambda k=kind: S.indirect_response(k)),
                      params=dict(CL=2.2, V=28., Ka=1.5, R0=100., KOUT=.12, **extra), outputs=2,
                      subjects=[[dose(0, 150, cmt=1), dose(24, 150, cmt=1)] + obs(GRID + [96])]))
    s.append(dict(name='tmdd_full', r_model='tmdd_full', build=lambda: S.tmdd('full'),
                  params=dict(CL=.2, V=3., KON=.8, KOFF=.05, KINT=.3, KDEG=.1, R0=2.), outputs=2,
                  subjects=[[dose(0, 30)] + obs(GRID + [96, 120, 168])]))
    s.append(dict(name='tmdd_qss', r_model='tmdd_qss', build=lambda: S.tmdd('qss'),
                  params=dict(CL=.2, V=3., KSS=.4375, KINT=.3, KDEG=.1, R0=2.), outputs=2,
                  subjects=[[dose(0, 30)] + obs(GRID + [96, 120, 168])]))
    return s


def write():
    OUT.mkdir(parents=True, exist_ok=True)
    manifest = []
    for sc in scenarios():
        rows = []
        for sid, recs in enumerate(sc['subjects'], start=1):
            for r in recs:
                row = dict(ID=sid, TIME=r['TIME'], EVID=r['EVID'], AMT=r.get('AMT', 0), CMT=r.get('CMT', 1),
                           RATE=r.get('RATE', 0), II=r.get('II', 0), SS=r.get('SS', 0), ADDL=r.get('ADDL', 0),
                           DV='.' if r['EVID'] != 0 else 0, DVID=1, WT=r.get('WT', 70))
                rows.append(row)
        with (OUT / f"{sc['name']}.csv").open('w', newline='') as f:
            w = csv.DictWriter(f, fieldnames=COLUMNS)
            w.writeheader()
            w.writerows(rows)
        params = sc['params'] if sc['params'] is not None else dict(TCL=3.1, V=42.)
        manifest.append(dict(name=sc['name'], r_model=sc['r_model'], params=params, outputs=sc.get('outputs', 1)))
    (OUT / 'manifest.json').write_text(json.dumps(manifest, indent=1))
    print(len(manifest), 'scenarios written to', OUT)


def pkpy2_predictions(sc):
    structure = sc['build']()
    data = read_nonmem(OUT / f"{sc['name']}.csv", covariates=['WT'])
    results = []
    for ind in data:
        if sc['params'] is None:     # CL = TCL*(WT/70)^0.75 per record
            params = dict(CL=3.1 * (ind.covariates['WT'] / 70.) ** .75, V=42.)
        else:
            params = sc['params']
        rows = {}
        for out in range(sc.get('outputs', 1)):
            ind.dvid[ind.evid == 0] = out + 1
            times, _, pred = predict(structure, ind, params)
            rows[out] = pred
        results.append((ind.id, times, rows))
    return results


def compare():
    report = []
    values = []
    for sc in scenarios():
        ref = list(csv.DictReader((OUT / f"{sc['name']}_rxode2.csv").open()))
        ours = pkpy2_predictions(sc)
        worst = 0.
        count = 0
        for sid, times, rows in ours:
            subset = [r for r in ref if int(float(r['id'])) == sid]
            if len(subset) != len(times):
                raise RuntimeError(f"{sc['name']}: {len(subset)} rxode2 rows vs {len(times)} observations")
            for j, t in enumerate(times):
                assert abs(float(subset[j]['time']) - t) < 1e-9, (sc['name'], t, subset[j]['time'])
                for out, name in enumerate(['cp', 'out2'][:sc.get('outputs', 1)]):
                    a, b = rows[out][j], float(subset[j][name])
                    scale = max(abs(b), 1e-6 * max(abs(float(r[name])) for r in subset))
                    worst = max(worst, abs(a - b) / scale)
                    count += 1
                    values.append(dict(scenario=sc['name'], ID=sid, TIME=t, output=out + 1, pkpy2=a, rxode2=b,
                                       scaled_difference=abs(a - b) / scale))
        report.append(dict(scenario=sc['name'], structure=sc['build']().name, values=count,
                           max_relative_difference=worst))
        print(f"{sc['name']:26s} {count:4d} values  max relative difference {worst:.2e}")
    with (OUT.parent / 'prediction_values.csv').open('w', newline='') as fh:
        w = csv.DictWriter(fh, fieldnames=list(values[0]))
        w.writeheader()
        w.writerows(values)
    path = OUT.parent / 'prediction_agreement.json'
    path.write_text(json.dumps(report, indent=1))
    print('written', path)


if __name__ == '__main__':
    {'write': write, 'compare': compare}[sys.argv[1]]()
