#!/usr/bin/env bash
set -euo pipefail

ROOT="${ROOT:-/Data3/it_FA0571/vlm}"
cd "$ROOT"
RUN_DIR="${RUN_DIR:-$ROOT/logs/control_analysis/$(date +%Y%m%d_%H%M%S)}"
mkdir -p "$RUN_DIR" "$ROOT/logs/control_analysis"
echo "$RUN_DIR" > "$ROOT/logs/control_analysis/latest_run.txt"

vars="CONTROL_ANALYSIS_DIR=${CONTROL_ANALYSIS_DIR:-$ROOT/results/control_analysis}"
log="$RUN_DIR/control_analysis.pbs.log"
if [[ -n "${AFTER_JOB:-}" ]]; then
  job_id="$(qsub -v "$vars" -W "depend=${DEPEND_TYPE:-afterany}:$AFTER_JOB" -o "$log" scripts/pbs_control_analysis.pbs)"
else
  job_id="$(qsub -v "$vars" -o "$log" scripts/pbs_control_analysis.pbs)"
fi
echo "$job_id" > "$RUN_DIR/last_job.txt"
echo "Submitted control analysis: $job_id"
echo "Log: $log"
qstat -u "$USER" || true
