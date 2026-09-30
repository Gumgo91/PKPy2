#!/usr/bin/env bash
# Sequential run of the extended-interface validation (general engine).
# Usage: bash scripts/run_pkpy2_extended_validation.sh [step ...]
# Steps: classic general_vs_classic likelihood vs_nlmixr2 warfarin_pkpd calibration tools example scm recovery
#        (also calibration_complex calibration_pkpd recovery_complex recovery_pkpd summaries)
# Parallel lanes: NUMBA_NUM_THREADS=4 bash scripts/run_pkpy2_extended_validation.sh <steps> &
set -u
cd "$(dirname "$0")/.."
LOG=output/pkpy2_extended_validation/logs
mkdir -p "$LOG"
steps=("$@")
[ ${#steps[@]} -eq 0 ] && steps=(classic general_vs_classic likelihood vs_nlmixr2 warfarin_pkpd calibration tools example scm recovery)
run() {
  name=$1; shift
  echo "$(date '+%F %T') start $name" | tee -a "$LOG/queue.log"
  python -u "$@" > "$LOG/$name.log" 2>&1
  echo "$(date '+%F %T') end $name (exit $?)" | tee -a "$LOG/queue.log"
}
for s in "${steps[@]}"; do
  case $s in
    classic) run classic_unchanged scripts/validate_pkpy2_classic_unchanged.py ;;
    predictions) run predictions scripts/validate_pkpy2_extended_predictions.py compare ;;
    diagnostics) run diagnostics scripts/validate_pkpy2_diagnostics.py compare ;;
    general_vs_classic) run general_vs_classic scripts/validate_pkpy2_general_vs_classic.py ;;
    likelihood) run likelihood scripts/validate_pkpy2_extended_likelihood.py ;;
    vs_nlmixr2) run vs_nlmixr2 scripts/validate_pkpy2_extended_vs_nlmixr2.py fit ;;
    warfarin_pkpd) run warfarin_pkpd scripts/validate_pkpy2_warfarin_pkpd.py ;;
    calibration) run calibration_complex scripts/validate_pkpy2_calibration.py complex_linear 50
                 run calibration_pkpd scripts/validate_pkpy2_calibration.py pkpd_idr 30
                 run calibration_summary scripts/validate_pkpy2_calibration.py summarize ;;
    calibration_complex) run calibration_complex scripts/validate_pkpy2_calibration.py complex_linear 50 ;;
    calibration_pkpd) run calibration_pkpd scripts/validate_pkpy2_calibration.py pkpd_idr 30 ;;
    recovery_complex) run recovery_complex scripts/validate_pkpy2_extended_recovery.py complex_linear 20 ;;
    recovery_pkpd) run recovery_pkpd scripts/validate_pkpy2_extended_recovery.py pkpd_idr 20 ;;
    summaries) run calibration_summary scripts/validate_pkpy2_calibration.py summarize
               run recovery_summary scripts/validate_pkpy2_extended_recovery.py summarize
               run tools_summary scripts/validate_pkpy2_tools.py summarize ;;
    tools) run tools_theophylline scripts/validate_pkpy2_tools.py theophylline 200 ;;
    example) run example_theophylline scripts/validate_pkpy2_theophylline_example.py ;;
    scm) run tools_scm scripts/validate_pkpy2_tools.py scm 20
         run tools_summary scripts/validate_pkpy2_tools.py summarize ;;
    recovery) run recovery_complex scripts/validate_pkpy2_extended_recovery.py complex_linear 20
              run recovery_pkpd scripts/validate_pkpy2_extended_recovery.py pkpd_idr 20
              run recovery_summary scripts/validate_pkpy2_extended_recovery.py summarize ;;
    *) echo "unknown step $s" ;;
  esac
done
