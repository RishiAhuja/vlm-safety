#!/usr/bin/env bash
set -euo pipefail

ROOT="${ROOT:-/Data3/it_FA0571/vlm}"
cd "$ROOT"

if ! command -v qsub >/dev/null 2>&1; then
  echo "ERROR: qsub not found. Submit this only from the HPC login node." >&2
  exit 1
fi

if pgrep -u "$USER" -af "model_smoke_hf.py|ollama serve|ollama pull" >/dev/null; then
  echo "ERROR: login-node model processes are still running. Stop them before submitting PBS jobs:" >&2
  pgrep -u "$USER" -af "model_smoke_hf.py|ollama serve|ollama pull" >&2 || true
  exit 1
fi

RUN_DIR="${RUN_DIR:-$ROOT/logs/model_smoke/pbs_$(date +%Y%m%d_%H%M%S)}"
mkdir -p "$RUN_DIR"
echo "$RUN_DIR" > "$ROOT/logs/model_smoke/latest_pbs_run.txt"
echo "$RUN_DIR" > "$ROOT/logs/model_smoke/latest_hf_run.txt"

MODELS=("$@")
if [[ "${#MODELS[@]}" -eq 0 ]]; then
  MODELS=(phi glm internvl molmo cogvlm2 deepseek)
fi

QSUB_VARS="RUN_DIR=$RUN_DIR,HF_HOME=/Data3/it_FA0571/hf_cache"
if [[ -n "${MIG_UUID:-}" ]]; then
  QSUB_VARS="$QSUB_VARS,MIG_UUID=$MIG_UUID"
fi

prev_job=""
echo "Submitting PBS smoke jobs sequentially to workq."
echo "Run dir: $RUN_DIR"

for model in "${MODELS[@]}"; do
  vars="$QSUB_VARS,MODEL=$model"
  output="$RUN_DIR/${model}.pbs.log"
  if [[ -n "$prev_job" ]]; then
    job_id="$(qsub -v "$vars" -W "depend=afterany:$prev_job" -o "$output" scripts/pbs_model_smoke.pbs)"
  else
    job_id="$(qsub -v "$vars" -o "$output" scripts/pbs_model_smoke.pbs)"
  fi
  echo "$model $job_id" | tee -a "$RUN_DIR/submitted_jobs.tsv"
  prev_job="$job_id"
done

echo
echo "Monitor with:"
echo "  cd $ROOT && scripts/model_smoke_status.sh"
echo
echo "Current PBS queue:"
qstat -u "$USER" || true
