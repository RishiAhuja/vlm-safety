#!/usr/bin/env bash
set -euo pipefail

ROOT="${ROOT:-/Data3/it_FA0571/vlm}"
cd "$ROOT"

if [[ -f logs/model_smoke/latest_hf_run.txt ]]; then
  LOG_DIR="$(cat logs/model_smoke/latest_hf_run.txt)"
else
  LOG_DIR="${1:-}"
fi

echo "===== time ====="
date
echo

echo "===== GPUs ====="
nvidia-smi --query-gpu=index,name,memory.used,memory.total,utilization.gpu --format=csv,noheader
echo

echo "===== relevant processes ====="
pgrep -af "model_smoke_hf.py|queue_gpu|ollama serve|ollama pull|pip install|from_pretrained|generate" || true
echo

echo "===== Ollama models ====="
if command -v ollama >/dev/null 2>&1; then
  env TERM=dumb OLLAMA_HOST=127.0.0.1:11435 OLLAMA_MODELS=/Data3/it_FA0571/ollama_models ollama list || true
else
  env TERM=dumb PATH=/Data3/it_FA0571/.local/bin:$PATH OLLAMA_HOST=127.0.0.1:11435 OLLAMA_MODELS=/Data3/it_FA0571/ollama_models ollama list || true
fi
echo

if [[ -z "${LOG_DIR:-}" || ! -d "$LOG_DIR" ]]; then
  echo "No HF smoke log directory found yet."
  exit 0
fi

echo "===== HF smoke log dir ====="
echo "$LOG_DIR"
echo

echo "===== queue logs ====="
for f in "$LOG_DIR"/queue_gpu*.log; do
  [[ -f "$f" ]] || continue
  echo "--- $f"
  tail -20 "$f"
done
echo

echo "===== model phases/results ====="
for f in "$LOG_DIR"/*.log; do
  [[ -f "$f" ]] || continue
  case "$(basename "$f")" in
    queue_*) continue ;;
  esac
  echo "--- $f"
  grep -E "START |PHASE |RESULT_JSON|Traceback|Error|ERROR|Killed|No space|OutOfMemory|CUDA out" "$f" | tail -30 || tail -20 "$f"
done
echo

echo "===== JSON results ====="
for f in "$LOG_DIR"/*.json; do
  [[ -f "$f" ]] || continue
  "$ROOT"/../hf_vlm_env/bin/python - <<PY
import json
from pathlib import Path
p = Path("$f")
d = json.loads(p.read_text())
print(f"{p.name}: status={d.get('status')} elapsed_s={d.get('elapsed_s')} gpu={d.get('cuda_visible_devices')} error={d.get('error', '')[:160]}")
resp = d.get("response", "")
if resp:
    print("  response=" + resp[:220].replace("\\n", " "))
PY
done
