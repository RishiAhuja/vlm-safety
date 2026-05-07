#!/usr/bin/env bash
set -euo pipefail

ROOT="${ROOT:-/Data3/it_FA0571/vlm}"
cd "$ROOT"

if ! command -v qsub >/dev/null 2>&1; then
  echo "ERROR: qsub not found. Submit this only from the HPC login node." >&2
  exit 1
fi

matching_processes="$(pgrep -u "$USER" -af "run_hf_matrix.py|model_smoke_hf.py|ollama serve|ollama pull" || true)"
running_pbs_jobs="$(qstat -u "$USER" 2>/dev/null | awk '$6 == "R" || $10 == "R" {print}' || true)"
if [[ -n "$matching_processes" && -z "$running_pbs_jobs" ]]; then
  echo "ERROR: matching model processes exist but no running PBS job was found:" >&2
  echo "$matching_processes" >&2
  exit 1
elif [[ -n "$matching_processes" ]]; then
  echo "Found model process owned by running PBS job; continuing with queued/dependent submission."
fi

RUN_DIR="${RUN_DIR:-$ROOT/logs/hf_matrix/pbs_$(date +%Y%m%d_%H%M%S)}"
OUTPUT_FILE="${OUTPUT_FILE:-$RUN_DIR/hf_inference_results.json}"
mkdir -p "$RUN_DIR"
echo "$RUN_DIR" > "$ROOT/logs/hf_matrix/latest_run.txt"
echo "$OUTPUT_FILE" > "$ROOT/logs/hf_matrix/latest_results.txt"

MODELS=("$@")
if [[ "${#MODELS[@]}" -eq 0 ]]; then
  MODELS=(phi molmo internvl glm cogvlm2 deepseek)
fi

PERSONAS_CSV="${PERSONAS:-none,west,east}"
IFS=',' read -r -a PERSONA_LIST <<< "$PERSONAS_CSV"

QSUB_VARS="RUN_DIR=$RUN_DIR,OUTPUT_FILE=$OUTPUT_FILE,HF_HOME=/Data3/it_FA0571/hf_cache,LIMIT=${LIMIT:-0},CONTROL_MODE=${CONTROL_MODE:-image_task}"
if [[ -n "${GPU_SELECTOR:-}" ]]; then
  QSUB_VARS="$QSUB_VARS,GPU_SELECTOR=$GPU_SELECTOR"
elif [[ -n "${MIG_UUID:-}" ]]; then
  QSUB_VARS="$QSUB_VARS,MIG_UUID=$MIG_UUID"
fi

prev_job="${AFTER_JOB:-}"
echo "Submitting HF matrix jobs sequentially to workq."
echo "Run dir: $RUN_DIR"
echo "Results: $OUTPUT_FILE"

for model in "${MODELS[@]}"; do
  for persona in "${PERSONA_LIST[@]}"; do
    vars="$QSUB_VARS,MODEL=$model,PERSONA=$persona"
    output="$RUN_DIR/${model}_${persona}.pbs.log"
    if [[ -n "$prev_job" ]]; then
      job_id="$(qsub -v "$vars" -W "depend=afterany:$prev_job" -o "$output" scripts/pbs_hf_matrix.pbs)"
    else
      job_id="$(qsub -v "$vars" -o "$output" scripts/pbs_hf_matrix.pbs)"
    fi
    echo "$model $persona $job_id" | tee -a "$RUN_DIR/submitted_jobs.tsv"
    prev_job="$job_id"
  done
done

echo "$prev_job" > "$RUN_DIR/last_job.txt"

echo
echo "Monitor compactly with:"
echo "  cd $ROOT && scripts/progress.sh"
echo "  cd $ROOT && scripts/watch_progress.sh 30"
echo
qstat -u "$USER" || true
