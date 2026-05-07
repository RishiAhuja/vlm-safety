#!/usr/bin/env bash
set -euo pipefail

ROOT="${ROOT:-/Data3/it_FA0571/vlm}"
cd "$ROOT"

if ! command -v qsub >/dev/null 2>&1; then
  echo "ERROR: qsub not found. Submit this only from the HPC login node." >&2
  exit 1
fi

RUN_DIR="${RUN_DIR:-$ROOT/logs/ollama_smoke/pbs_$(date +%Y%m%d_%H%M%S)}"
OUTPUT_FILE="${OUTPUT_FILE:-$RUN_DIR/ollama_smoke_results.json}"
mkdir -p "$RUN_DIR" "$ROOT/logs/ollama_smoke"
echo "$RUN_DIR" > "$ROOT/logs/ollama_smoke/latest_run.txt"
echo "$OUTPUT_FILE" > "$ROOT/logs/ollama_smoke/latest_results.txt"

MODEL_KEYS=("$@")
if [[ "${#MODEL_KEYS[@]}" -eq 0 ]]; then
  MODEL_KEYS=(llava_7b llama32_vision gemma3_12b granite_vision qwen2.5vl_3b minicpm_v)
fi

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

QSUB_VARS="RUN_DIR=$RUN_DIR,OUTPUT_FILE=$OUTPUT_FILE,LIMIT=${LIMIT:-1},OLLAMA_MODELS=${OLLAMA_MODELS:-/Data3/it_FA0571/ollama_models},PERSONA=${PERSONA:-none},GPU_SELECTOR=${GPU_SELECTOR:-auto}"
if [[ -n "${OLLAMA_NUM_CTX:-}" ]]; then
  QSUB_VARS="$QSUB_VARS,OLLAMA_NUM_CTX=$OLLAMA_NUM_CTX"
fi
prev_job="${AFTER_JOB:-}"
echo "Submitting Ollama smoke jobs sequentially to workq."
echo "Run dir: $RUN_DIR"
echo "Results: $OUTPUT_FILE"

for model_key in "${MODEL_KEYS[@]}"; do
  if [[ -z "${TAGS[$model_key]:-}" || -z "${ORIGINS[$model_key]:-}" ]]; then
    echo "ERROR: unknown model key: $model_key" >&2
    exit 2
  fi
  vars="$QSUB_VARS,MODEL_KEY=$model_key,MODEL_TAG=${TAGS[$model_key]},ORIGIN=${ORIGINS[$model_key]}"
  output="$RUN_DIR/${model_key}_${PERSONA:-none}.pbs.log"
  if [[ -n "$prev_job" ]]; then
    job_id="$(qsub -v "$vars" -W "depend=afterany:$prev_job" -o "$output" scripts/pbs_ollama_matrix.pbs)"
  else
    job_id="$(qsub -v "$vars" -o "$output" scripts/pbs_ollama_matrix.pbs)"
  fi
  echo "$model_key ${PERSONA:-none} ${TAGS[$model_key]} ${ORIGINS[$model_key]} $job_id" | tee -a "$RUN_DIR/submitted_jobs.tsv"
  prev_job="$job_id"
done

echo "$prev_job" > "$RUN_DIR/last_job.txt"
echo "Monitor with: qstat -u $USER; tail -f $RUN_DIR/*.pbs.log"
