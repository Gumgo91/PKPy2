#!/usr/bin/env bash
# Rebuild every manuscript product of the JPKPD submission from the saved results.
# Usage: bash scripts/build_pkpy2_submission.sh "<submitted PeerJ folder>" "<JPKPD output folder>"
set -euo pipefail
cd "$(dirname "$0")/.."
SRC=${1:?usage: build_pkpy2_submission.sh <submitted PeerJ folder> <JPKPD output folder>}
DST=${2:?usage: build_pkpy2_submission.sh <submitted PeerJ folder> <JPKPD output folder>}
BUILD=output/pkpy2_peerj_build
python scripts/pkpy2_manuscript_numbers.py
python scripts/pkpy2_extended_numbers.py > /dev/null
python scripts/build_pkpy2_peerj_figures.py --tobramycin               # Figures 2-5
python scripts/build_pkpy2_extended_figures.py                         # Figures 1, 6 and 7
python scripts/build_pkpy2_peerj_tables.py "$SRC/Tables/Table_3.docx" "$BUILD/Tables" --tobramycin
python scripts/build_pkpy2_peerj_raw_data.py "$BUILD/Supplemental_Files"
python scripts/build_pkpy2_peerj_codebook.py "$BUILD/Supplemental_Files"
python scripts/revise_pkpy2_peerj_manuscript.py "$SRC" "$BUILD"
python scripts/build_pkpy2_peerj_supplement.py "$BUILD/PKPy2_supplement.pdf" --tobramycin --springer-references
python scripts/build_pkpy2_jpkpd_submission.py "$BUILD" "$DST"
python scripts/build_pkpy2_paper_materials.py "$BUILD/Supplemental_Files" "$DST/GitHub_upload/paper"
