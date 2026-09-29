"""Write the PKPy2 benchmark inputs as NONMEM-style CSV files for external NLME software.

The exported rows are exactly the observations analysed by PKPy2: the 200 primary
simulation datasets (confirmatory_v2), the theophylline analysis dataset and the
warfarin analysis dataset. Nothing is re-simulated.
"""
from pathlib import Path
import csv
import hashlib
import json

ROOT = Path(__file__).resolve().parents[1]
DEV = ROOT / 'output/pkpy2_development'
REF = ROOT / 'output/pkpy2_nonmem_reference'
OUT = ROOT / 'output/pkpy2_software_comparison/data'
FIELDS = ['ID', 'TIME', 'AMT', 'DV', 'EVID', 'MDV', 'CMT', 'WT']


def rows_for(subjects, oral):
    """Yield dose and observation records; oral doses enter the depot (CMT 1)."""
    for s in subjects:
        wt = s.get('covariates', {}).get('WT', '')
        yield dict(ID=s['sid'], TIME=0.0, AMT=s['dose'], DV='.', EVID=1, MDV=1, CMT=1, WT=wt)
        for t, y in zip(s['time'], s['obs']):
            yield dict(ID=s['sid'], TIME=t, AMT='.', DV=y, EVID=0, MDV=0, CMT=2 if oral else 1, WT=wt)


def write(path, rows):
    with path.open('w', encoding='utf-8', newline='') as f:
        w = csv.DictWriter(f, fieldnames=FIELDS)
        w.writeheader()
        w.writerows(rows)
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    (OUT / 'simulation').mkdir(parents=True, exist_ok=True)
    manifest = []
    for fit in sorted((DEV / 'confirmatory_v2').glob('1cmt_iv__*__[0-9][0-9][0-9].json')):
        record = json.loads(fit.read_text(encoding='utf-8'))
        data = json.loads((fit.parent / record['data_file']).read_text(encoding='utf-8'))
        subjects = [dict(sid=i + 1, time=t, obs=y, dose=data['dose'])
                    for i, (t, y) in enumerate(zip(data['time'], data['observations']))]
        path = OUT / 'simulation' / f"{fit.stem}.csv"
        manifest.append(dict(dataset=fit.stem, sampling=record['sampling'], replicate=record['replicate'],
                             seed=record['seed'], file=f'simulation/{path.name}', sha256=write(path, rows_for(subjects, False)),
                             source_data_sha256=record['pkpy2']['data_sha256']))
    for name, source in [('theophylline', DEV / 'theophylline_v2/data.json'), ('warfarin', REF / 'warfarin_data.json')]:
        subjects = json.loads(source.read_text(encoding='utf-8'))
        path = OUT / f'{name}.csv'
        manifest.append(dict(dataset=name, file=path.name, sha256=write(path, rows_for(subjects, True)),
                             subjects=len(subjects), observations=sum(len(s['obs']) for s in subjects)))
    (OUT / 'manifest.json').write_text(json.dumps(manifest, indent=1), encoding='utf-8')
    print(len(manifest), 'datasets written to', OUT)


if __name__ == '__main__':
    main()
