#!/usr/bin/env bash
set -euo pipefail

ROOT="${ROOT:-/Data3/it_FA0571/vlm}"
cd "$ROOT"

if ! command -v qsub >/dev/null 2>&1; then
  echo "ERROR: qsub not found. Submit this only from the HPC login node." >&2
  exit 1
fi

matching_processes="$(pgrep -u "$USER" -af "run_hf_matrix.py|run_ollama_matrix.py|model_smoke_hf.py|ollama serve|ollama pull" || true)"
running_pbs_jobs="$(qstat -u "$USER" 2>/dev/null | awk '$6 == "R" || $10 == "R" {print}' || true)"
if [[ -n "$matching_processes" && -z "$running_pbs_jobs" ]]; then
  echo "ERROR: matching model processes exist but no running PBS job was found:" >&2
  echo "$matching_processes" >&2
  exit 1
elif [[ -n "$matching_processes" ]]; then
  echo "Found model process owned by running PBS job; continuing with queued/dependent submission."
fi

RUN_DIR="${RUN_DIR:-$ROOT/logs/ollama_matrix/pbs_$(date +%Y%m%d_%H%M%S)}"
OUTPUT_FILE="${OUTPUT_FILE:-$RUN_DIR/ollama_inference_results.json}"
mkdir -p "$RUN_DIR" "$ROOT/logs/ollama_matrix"
echo "$RUN_DIR" > "$ROOT/logs/ollama_matrix/latest_run.txt"
echo "$OUTPUT_FILE" > "$ROOT/logs/ollama_matrix/latest_results.txt"

MODEL_KEYS=("$@")
if [[ "${#MODEL_KEYS[@]}" -eq 0 ]]; then
  MODEL_KEYS=(llava_7b llama32_vision gemma3_12b granite_vision qwen2.5vl_3b minicpm_v)
fi

PERSONAS_CSV="${PERSONAS:-none,west,east}"
IFS=',' read -r -a PERSONA_LIST <<< "$PERSONAS_CSV"

declare -A TAGS ORIGINS
TAGS[llava_7b]="llava:7b"
ORIGINS[llava_7b]="Western"
TAGS[llama32_vision]="llama3.2-vision:11b"
ORIGINS[llama32_vision]="Western"
TAGS[gemma3_12b]="gemma3:12b"
ORIGINS[gemma3_12b]="Western"
TAGS[granite_vision]="granite3.2-vision"
ORIGINS[granite_vision]="Western"
TAGS[qwen2.5vl_3b]="qwen2.5vl:3b"
ORIGINS[qwen2.5vl_3b]="Eastern"
TAGS[minicpm_v]="minicpm-v"
ORIGINS[minicpm_v]="Eastern"

QSUB_VARS="RUN_DIR=$RUN_DIR,OUTPUT_FILE=$OUTPUT_FILE,LIMIT=${LIMIT:-0},OLLAMA_MODELS=${OLLAMA_MODELS:-/Data3/it_FA0571/ollama_models}"
if [[ -n "${MIG_UUID:-}" ]]; then
  QSUB_VARS="$QSUB_VARS,MIG_UUID=$MIG_UUID"
fi

prev_job="${AFTER_JOB:-}"
echo "Submitting Ollama matrix jobs sequentially to workq."
echo "Run dir: $RUN_DIR"
echo "Results: $OUTPUT_FILE"
if [[ -n "$prev_job" ]]; then
  echo "First job waits on: $prev_job"
fi

for model_key in "${MODEL_KEYS[@]}"; do
  if [[ -z "${TAGS[$model_key]:-}" || -z "${ORIGINS[$model_key]:-}" ]]; then
    echo "ERROR: unknown model key: $model_key" >&2
    exit 2
  fi
  model_tag="${TAGS[$model_key]}"
  origin="${ORIGINS[$model_key]}"
  for persona in "${PERSONA_LIST[@]}"; do
    vars="$QSUB_VARS,MODEL_KEY=$model_key,MODEL_TAG=$model_tag,ORIGIN=$origin,PERSONA=$persona"
    output="$RUN_DIR/${model_key}_${persona}.pbs.log"
    if [[ -n "$prev_job" ]]; then
      job_id="$(qsub -v "$vars" -W "depend=afterany:$prev_job" -o "$output" scripts/pbs_ollama_matrix.pbs)"
    else
      job_id="$(qsub -v "$vars" -o "$output" scripts/pbs_ollama_matrix.pbs)"
    fi
    echo "$model_key $persona $model_tag $origin $job_id" | tee -a "$RUN_DIR/submitted_jobs.tsv"
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
