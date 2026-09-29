# PKPy2 paper materials: comparison with nlmixr2 and saemix, tobramycin analyses, and raw data

Materials for the PKPy2 manuscript submitted to the Journal of Pharmacokinetics and Pharmacodynamics. This
folder contains the analysis data, scripts and saved fit records for the comparison of PKPy2 with nlmixr2
(FOCEi, SAEM) and saemix, the PKPy2 tobramycin analyses, the raw data underlying all tables and figures,
and the scripts that assemble the figures, tables and supplementary material.

## Contents

| Path | Description |
| --- | --- |
| `data/simulation/*.csv` | The 200 primary simulation datasets in NONMEM-style event format, as analysed by PKPy2 |
| `data/theophylline.csv`, `data/warfarin.csv`, `data/tobramycin.csv` | Clinical analysis datasets (event format with covariates) |
| `data/manifest.json` | SHA-256 of every exported file and of the corresponding PKPy2 input |
| `results/simulation/*.json` | One record per dataset and program (nlmixr2 FOCEi, nlmixr2 SAEM, saemix): estimates, standard errors, run time, messages, environment |
| `results/clinical/*.json` | Clinical fits of the same programs; `tobramycin_expert__*` use the expert-judgment constraints, `tobramycin__*` the published model without constraints |
| `tobramycin/*.json` | PKPy2 tobramycin fits (without constraints, with the expert NONMEM bounds, with partial and full expert-judgment constraints), their protocols, the independent V2 profile likelihood, and the nlmixr2 FOCEi objective check |
| `comparison_summary.json` | Bias, RMSE, coverage, paired-bootstrap RMSE differences and clinical comparisons |
| `comparison_replicates.csv` | Per-dataset estimates of PKPy2 and the three external programs in long format |
| `clinical_likelihood_check.json` | Exact marginal OFV at each program's clinical estimates (independent adaptive Gauss-Hermite quadrature) |
| `logs/` | Run logs of the simulation shards and tobramycin fits |
| `install_packages.R` | Package installation used for the comparison |
| `scripts/` | All scripts listed below |
| `raw_data/` | Raw-data workbook, its CSV files and the codebook (Online Resources 2-4 of the article) |

The tobramycin data are the public dataset of the PKGPT repository (https://github.com/Gumgo91/PKGPT, `dataset/`).

## Software

R 4.5.3, nlmixr2 7.0.1 (nlmixr2est 7.1.0, rxode2 5.1.7), saemix 3.5, Rtools 4.5 (Windows);
Python 3.13.5 with PKPy2, NumPy, SciPy, pandas, matplotlib, python-docx and PyMuPDF.

## Reproducing

The scripts were run in this order from the root of the study workspace, in which the saved PKPy2
simulation and clinical records reside under `output/`; the paths in the scripts refer to that layout.

```
python scripts/export_pkpy2_comparison_data.py              # write data/ (NONMEM-style CSV)
Rscript scripts/pkpy2_software_comparison.R clinical
Rscript scripts/pkpy2_software_comparison.R simulation 0 5  # repeat for shards 0..4 (parallel)
Rscript scripts/pkpy2_software_comparison.R tobramycin
Rscript scripts/pkpy2_software_comparison.R tobramycin_expert
python scripts/pkpy2_tobramycin_application.py              # PKPy2, published model without constraints
python scripts/pkpy2_tobramycin_bounded.py                  # expert NONMEM bounds
python scripts/pkpy2_tobramycin_judgment.py                 # WT exponent fixed, V2 <= 30 L
python scripts/pkpy2_tobramycin_expert_judgment.py          # expert-judgment constraints
python scripts/pkpy2_tobramycin_profile.py                  # independent V2 profile likelihood
Rscript scripts/pkpy2_tobramycin_focei_check.R
python scripts/analyze_pkpy2_software_comparison.py
python scripts/pkpy2_clinical_likelihood_check.py
python scripts/pkpy2_manuscript_numbers.py
python scripts/build_pkpy2_peerj_figures.py --tobramycin
python scripts/build_pkpy2_peerj_tables.py <Table_3.docx template> <output dir> --tobramycin
python scripts/build_pkpy2_peerj_raw_data.py <output dir>
python scripts/build_pkpy2_peerj_codebook.py <output dir>
python scripts/build_pkpy2_peerj_supplement.py <output.pdf> --tobramycin
```

The R script skips records that already exist, so an interrupted run can be resumed. Models, starting
values, fixed terms and bounds are identical to the PKPy2 protocols. nlmixr2 SAEM does not apply
parameter bounds; saemix was not applied to warfarin, whose model fixes nonzero variance components,
or to the bounded tobramycin model.

## Raw data

`raw_data/PKPy2_raw_data.xlsx` (and `raw_data/PKPy2_raw_data_csv.zip`, one CSV per sheet) contains the raw
data underlying all figures and tables; `raw_data/PKPy2_codebook.xlsx` defines every variable, unit and
categorical code.
