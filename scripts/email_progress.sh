#!/usr/bin/env bash
set -euo pipefail

ROOT="${ROOT:-/Data3/it_FA0571/vlm}"
ENV_FILE="${VLM_PROGRESS_EMAIL_ENV:-$HOME/.config/vlm/resend.env}"
LOG_DIR="$ROOT/logs/progress_email"
mkdir -p "$LOG_DIR"

if [[ ! -f "$ENV_FILE" ]]; then
  echo "Missing env file: $ENV_FILE" >&2
  exit 2
fi

set -a
# shellcheck disable=SC1090
source "$ENV_FILE"
set +a

cd "$ROOT"
python3 scripts/email_progress.py "$@"
