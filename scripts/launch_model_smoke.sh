#!/usr/bin/env bash
set -euo pipefail

cat >&2 <<'MSG'
Direct login-node model execution is disabled.

The NITJ HPC policy requires GPU work to be submitted through PBS workq,
with one GPU instance per user. Forwarding to the PBS submitter.
MSG

exec "$(dirname "$0")/submit_model_smoke_pbs.sh" "$@"
