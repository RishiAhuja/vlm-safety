#!/usr/bin/env bash
set -euo pipefail

ROOT="${ROOT:-/Data3/it_FA0571/vlm}"
cd "$ROOT"

if ! command -v qsub >/dev/null 2>&1; then
  echo "ERROR: qsub not found. Submit this only from the HPC login node." >&2
  exit 1
fi

RUN_DIR="${RUN_DIR:-$ROOT/logs/model_smoke/env_setup_$(date +%Y%m%d_%H%M%S)}"
mkdir -p "$RUN_DIR"
echo "$RUN_DIR" > "$ROOT/logs/model_smoke/latest_env_setup_run.txt"

job_id="$(qsub -o "$RUN_DIR/env_setup.pbs.log" scripts/pbs_model_env_setup.pbs)"
echo "$job_id" | tee "$RUN_DIR/env_setup_job.txt"
echo "Monitor with:"
echo "  qstat -u $USER"
echo "  tail -f $RUN_DIR/env_setup.pbs.log"
