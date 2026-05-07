#!/usr/bin/env bash
set -euo pipefail

ROOT="${ROOT:-/Data3/it_FA0571/vlm}"
INTERVAL="${1:-30}"
cd "$ROOT"

while true; do
  if command -v tput >/dev/null 2>&1 && tput clear >/dev/null 2>&1; then
    tput clear
  else
    printf '\n\n'
  fi
  scripts/model_smoke_status.sh
  echo
  /Data3/it_FA0571/hf_vlm_env/bin/python scripts/model_smoke_eta.py || true
  echo
  echo "Refreshing every ${INTERVAL}s. Ctrl-C stops this watcher only; it does not affect PBS jobs."
  sleep "$INTERVAL"
done
