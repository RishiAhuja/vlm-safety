#!/usr/bin/env bash
set -euo pipefail

ROOT="${ROOT:-/Data3/it_FA0571/vlm}"
cd "$ROOT"

if ! command -v qsub >/dev/null 2>&1; then
  echo "ERROR: qsub not found. Submit this only from the HPC login node." >&2
  exit 1
fi

RUN_DIR="${RUN_DIR:-$ROOT/logs/ollama_pull/pbs_$(date +%Y%m%d_%H%M%S)}"
mkdir -p "$RUN_DIR" "$ROOT/logs/ollama_pull"
echo "$RUN_DIR" > "$ROOT/logs/ollama_pull/latest_run.txt"

TAGS=("$@")
if [[ "${#TAGS[@]}" -eq 0 ]]; then
  TAGS=(llava:7b llama3.2-vision:11b gemma3:12b granite3.2-vision qwen2.5vl:3b minicpm-v)
fi

QSUB_VARS="RUN_DIR=$RUN_DIR,OLLAMA_MODELS=${OLLAMA_MODELS:-/Data3/it_FA0571/ollama_models}"
prev_job="${AFTER_JOB:-}"
echo "Submitting Ollama pull/check jobs sequentially to cpuq."
echo "Run dir: $RUN_DIR"

for tag in "${TAGS[@]}"; do
  safe="${tag//[:\/]/_}"
  vars="$QSUB_VARS,MODEL_TAG=$tag"
  output="$RUN_DIR/${safe}.pbs.log"
  if [[ -n "$prev_job" ]]; then
    job_id="$(qsub -v "$vars" -W "depend=afterany:$prev_job" -o "$output" scripts/pbs_ollama_pull.pbs)"
  else
    job_id="$(qsub -v "$vars" -o "$output" scripts/pbs_ollama_pull.pbs)"
  fi
  echo "$tag $job_id" | tee -a "$RUN_DIR/submitted_jobs.tsv"
  prev_job="$job_id"
done

echo "$prev_job" > "$RUN_DIR/last_job.txt"
echo "Last pull job: $prev_job"
echo "Monitor with: qstat -u $USER"
