#!/usr/bin/env bash
set -euo pipefail

ROOT="${ROOT:-/Data3/it_FA0571/vlm}"
cd "$ROOT"

/Data3/it_FA0571/hf_vlm_env/bin/python scripts/progress_status.py
