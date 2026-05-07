#!/usr/bin/env bash
set -euo pipefail

exec "$(dirname "$0")/pbs_model_smoke_status.sh" "$@"
