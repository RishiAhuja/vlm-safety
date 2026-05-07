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
  scripts/progress.sh
  echo
  echo "Refreshing every ${INTERVAL}s. Press Ctrl-C to stop watching; PBS jobs keep running."
  sleep "$INTERVAL"
done
