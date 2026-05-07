#!/usr/bin/env bash
set -euo pipefail
ROOT="${ROOT:-/Data3/it_FA0571/vlm}"
cd "$ROOT"
RUN_DIR="${RUN_DIR:-$(cat logs/judge/latest_run.txt 2>/dev/null || true)}"

echo "Time: $(date '+%Y-%m-%d %H:%M:%S')"
echo "PBS:"
qstat -u "$USER" || true
if [[ -z "$RUN_DIR" || ! -d "$RUN_DIR" ]]; then
  echo "No judge run found."
  exit 0
fi

echo
echo "Judge run: $RUN_DIR"
python3 - "$RUN_DIR" <<'PY'
import json
import sys
from pathlib import Path
run_dir = Path(sys.argv[1])
submitted = run_dir / 'submitted_jobs.tsv'
if not submitted.exists():
    print('No submitted_jobs.tsv')
    raise SystemExit
for line in submitted.read_text().splitlines():
    parts = line.split('\t')
    if len(parts) < 4:
        continue
    target, job_id, input_path, output_path = parts[:4]
    inp = Path(input_path)
    out = Path(output_path)
    rows = json.loads(inp.read_text()) if inp.exists() else []
    scored = []
    if out.exists():
        try:
            scored = json.loads(out.read_text())
        except Exception:
            scored = []
    done = sum(1 for r in scored if r.get('score') is not None)
    api_err = sum(1 for r in scored if r.get('failure_mode') == 'api_error')
    parse_err = sum(1 for r in scored if r.get('failure_mode') == 'parse_error' or r.get('score') == -1 and r.get('failure_mode') == 'parse_error')
    model = next((r.get('judge_model') for r in scored if r.get('judge_model')), '')
    pct = 100 * done / len(rows) if rows else 0
    print(f'{target:9} {done:4}/{len(rows):4} {pct:5.1f}% api_error={api_err:<3} parse_error={parse_err:<3} model={model or "-"} job={job_id}')
PY
