#!/usr/bin/env bash
set -euo pipefail

ROOT="${ROOT:-/Data3/it_FA0571/vlm}"
PYTHON="${PYTHON:-/Data3/it_FA0571/hf_vlm_env/bin/python}"
LOG_DIR="${LOG_DIR:-$ROOT/logs/model_smoke/hf_$(date +%Y%m%d_%H%M%S)}"
IMAGE="${IMAGE:-$ROOT/stimuli/neutral_01.png}"
HF_HOME="${HF_HOME:-/Data3/it_FA0571/hf_cache}"

mkdir -p "$LOG_DIR"
cd "$ROOT"

export HF_HOME
export HUGGINGFACE_HUB_CACHE="$HF_HOME/hub"
export TRANSFORMERS_CACHE="$HF_HOME/transformers"
export TOKENIZERS_PARALLELISM=false
export PYTHONUNBUFFERED=1

if [[ "${1:-}" == "--queue" ]]; then
  gpu="$2"
  shift 2
  echo "[$(date --iso-8601=seconds)] QUEUE_START gpu=$gpu models=$*"
  for model in "$@"; do
    echo "[$(date --iso-8601=seconds)] MODEL_START gpu=$gpu model=$model"
    set +e
    CUDA_VISIBLE_DEVICES="$gpu" "$PYTHON" scripts/model_smoke_hf.py \
      --model "$model" \
      --image "$IMAGE" \
      --cache-dir "$HF_HOME" \
      --result-file "$LOG_DIR/${model}.json" \
      > "$LOG_DIR/${model}.log" 2>&1
    status="$?"
    set -e
    echo "[$(date --iso-8601=seconds)] MODEL_DONE gpu=$gpu model=$model status=$status"
  done
  echo "[$(date --iso-8601=seconds)] QUEUE_DONE gpu=$gpu"
  exit 0
fi

launch_queue() {
  local gpu="$1"
  shift
  local queue_log="$LOG_DIR/queue_gpu${gpu}.log"
  nohup bash "$0" --queue "$gpu" "$@" > "$queue_log" 2>&1 &
  echo "$!" > "$LOG_DIR/queue_gpu${gpu}.pid"
  echo "started gpu=$gpu pid=$(cat "$LOG_DIR/queue_gpu${gpu}.pid") models=$* log=$queue_log"
}

echo "$LOG_DIR" > "$ROOT/logs/model_smoke/latest_hf_run.txt"
echo "Log dir: $LOG_DIR"

# One queue per currently-free GPU. Each queue runs models sequentially, so no
# GPU gets two heavyweight model loads at once.
launch_queue 2 phi glm
launch_queue 3 molmo deepseek
launch_queue 4 internvl cogvlm2

echo
echo "Monitor with:"
echo "  cd $ROOT && scripts/model_smoke_status.sh"
echo "Tail all queue logs with:"
echo "  tail -f $LOG_DIR/queue_gpu2.log $LOG_DIR/queue_gpu3.log $LOG_DIR/queue_gpu4.log"
