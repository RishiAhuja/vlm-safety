#!/usr/bin/env bash
set -euo pipefail

ROOT="${ROOT:-/Data3/it_FA0571/vlm}"
cd "$ROOT"

if [[ -f logs/model_smoke/latest_pbs_run.txt ]]; then
  RUN_DIR="$(cat logs/model_smoke/latest_pbs_run.txt)"
else
  RUN_DIR="${1:-}"
fi

echo "===== time ====="
date
echo

echo "===== PBS jobs ====="
qstat -u "$USER" || true
echo

echo "===== login-node compute guard ====="
pgrep -u "$USER" -af "model_smoke_hf.py|ollama serve|ollama pull|python.*from_pretrained" || echo "No matching login-node compute processes."
echo

if [[ -z "${RUN_DIR:-}" || ! -d "$RUN_DIR" ]]; then
  echo "No PBS smoke run directory found."
  exit 0
fi

echo "===== run dir ====="
echo "$RUN_DIR"
echo

if [[ -f "$RUN_DIR/submitted_jobs.tsv" ]]; then
  echo "===== submitted jobs ====="
  cat "$RUN_DIR/submitted_jobs.tsv"
  echo
fi

echo "===== PBS logs ====="
for f in "$RUN_DIR"/*.pbs.log; do
  [[ -f "$f" ]] || continue
  echo "--- $f"
  tail -40 "$f"
done
echo

echo "===== model logs ====="
for f in "$RUN_DIR"/*.log; do
  [[ -f "$f" ]] || continue
  case "$(basename "$f")" in
    *.pbs.log) continue ;;
  esac
  echo "--- $f"
  grep -E "START |PHASE |RESULT_JSON|Traceback|Error|ERROR|Killed|OutOfMemory|CUDA out" "$f" | tail -40 || tail -40 "$f"
done
echo

echo "===== JSON results ====="
for f in "$RUN_DIR"/*.json; do
  [[ -f "$f" ]] || continue
  /Data3/it_FA0571/hf_vlm_env/bin/python - <<PY
import json
from pathlib import Path
p = Path("$f")
d = json.loads(p.read_text())
print(f"{p.name}: status={d.get('status')} elapsed_s={d.get('elapsed_s')} gpu={d.get('cuda_visible_devices')} error={d.get('error', '')[:180]}")
resp = d.get("response", "")
if resp:
    print("  response=" + resp[:240].replace("\\n", " "))
PY
done
