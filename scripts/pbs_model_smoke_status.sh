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

echo "===== scheduler/process check ====="
matching_processes="$(pgrep -u "$USER" -af "model_smoke_hf.py|ollama serve|ollama pull|python.*from_pretrained" || true)"
running_pbs_jobs="$(qstat -u "$USER" 2>/dev/null | awk '$6 == "R" || $10 == "R" {print}' || true)"
if [[ -z "$matching_processes" ]]; then
  echo "No matching model processes on this host."
elif [[ -n "$running_pbs_jobs" ]]; then
  echo "Matching model process is present while PBS reports a running job; expected for scheduler-owned execution."
  echo "$matching_processes"
else
  echo "WARNING: matching model process exists but no running PBS job was found:"
  echo "$matching_processes"
fi
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

echo "===== ETA ====="
/Data3/it_FA0571/hf_vlm_env/bin/python scripts/model_smoke_eta.py "$RUN_DIR" || true
echo

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
