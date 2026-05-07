#!/usr/bin/env bash
set -euo pipefail
ROOT="${ROOT:-/Data3/it_FA0571/vlm}"
cd "$ROOT"
RUN_DIR="${RUN_DIR:-$ROOT/logs/results_bundle/$(date +%Y%m%d_%H%M%S)}"
mkdir -p "$RUN_DIR" "$ROOT/logs/results_bundle"
echo "$RUN_DIR" > "$ROOT/logs/results_bundle/latest_run.txt"
log="$RUN_DIR/results_bundle.pbs.log"
if [[ -n "${AFTER_JOB:-}" ]]; then
  job_id="$(qsub -W "depend=${DEPEND_TYPE:-afterany}:$AFTER_JOB" -o "$log" scripts/pbs_results_bundle.pbs)"
else
  job_id="$(qsub -o "$log" scripts/pbs_results_bundle.pbs)"
fi
echo "$job_id" > "$RUN_DIR/last_job.txt"
echo "Submitted results bundle: $job_id"
echo "Log: $log"
qstat -u "$USER" || true
