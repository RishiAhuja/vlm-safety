#!/usr/bin/env bash
set -euo pipefail

ROOT="${ROOT:-/Data3/it_FA0571/vlm}"
cd "$ROOT"

if ! command -v qsub >/dev/null 2>&1; then
  echo "ERROR: qsub not found. Submit this only from the HPC login node." >&2
  exit 1
fi

RUN_DIR="${RUN_DIR:-$ROOT/logs/judge/gpt55_$(date +%Y%m%d_%H%M%S)}"
MODEL="${JUDGE_MODEL:-gpt-5.5}"
CONCURRENCY="${JUDGE_CONCURRENCY:-10}"
LIMIT="${JUDGE_LIMIT:-0}"
MAX_RESPONSE_CHARS="${JUDGE_MAX_RESPONSE_CHARS:-1500}"
mkdir -p "$RUN_DIR" "$ROOT/logs/judge"
echo "$RUN_DIR" > "$ROOT/logs/judge/latest_run.txt"

declare -a TARGETS
if [[ "$#" -eq 0 ]]; then
  TARGETS=(expanded pilot run_01 run_02 run_03)
else
  TARGETS=("$@")
fi

input_for() {
  case "$1" in
    expanded) echo "$ROOT/results/expanded/inference_results.json" ;;
    pilot) echo "$ROOT/results/inference_results.json" ;;
    run_01) echo "$ROOT/results/replicates/run_01/inference_results.json" ;;
    run_02) echo "$ROOT/results/replicates/run_02/inference_results.json" ;;
    run_03) echo "$ROOT/results/replicates/run_03/inference_results.json" ;;
    *) echo "ERROR: unknown judge target: $1" >&2; return 2 ;;
  esac
}

output_for() {
  case "$1" in
    expanded) echo "$ROOT/results/expanded/inference_results_scored_gpt55.json" ;;
    pilot) echo "$ROOT/results/inference_results_scored_gpt55.json" ;;
    run_01) echo "$ROOT/results/replicates/run_01/inference_results_scored_gpt55.json" ;;
    run_02) echo "$ROOT/results/replicates/run_02/inference_results_scored_gpt55.json" ;;
    run_03) echo "$ROOT/results/replicates/run_03/inference_results_scored_gpt55.json" ;;
    *) echo "ERROR: unknown judge target: $1" >&2; return 2 ;;
  esac
}

prev_job="${AFTER_JOB:-}"
: > "$RUN_DIR/submitted_jobs.tsv"
echo "Submitting judge jobs sequentially to cpuq."
echo "Run dir: $RUN_DIR"
echo "Judge model: $MODEL"
echo "Concurrency per job: $CONCURRENCY"

for target in "${TARGETS[@]}"; do
  input="$(input_for "$target")"
  output="$(output_for "$target")"
  if [[ ! -f "$input" ]]; then
    echo "ERROR: missing input for $target: $input" >&2
    exit 2
  fi
  vars="JUDGE_MODEL=$MODEL,JUDGE_CONCURRENCY=$CONCURRENCY,JUDGE_LIMIT=$LIMIT,JUDGE_MAX_RESPONSE_CHARS=$MAX_RESPONSE_CHARS,JUDGE_INPUT_FILE=$input,JUDGE_OUTPUT_FILE=$output"
  log="$RUN_DIR/${target}.pbs.log"
  if [[ -n "$prev_job" ]]; then
    job_id="$(qsub -v "$vars" -W "depend=afterany:$prev_job" -o "$log" scripts/pbs_judge.pbs)"
  else
    job_id="$(qsub -v "$vars" -o "$log" scripts/pbs_judge.pbs)"
  fi
  printf '%s\t%s\t%s\t%s\n' "$target" "$job_id" "$input" "$output" | tee -a "$RUN_DIR/submitted_jobs.tsv"
  prev_job="$job_id"
done

echo "$prev_job" > "$RUN_DIR/last_job.txt"
echo
qstat -u "$USER" || true
echo
echo "Monitor with:"
echo "  cd $ROOT && scripts/judge_status.sh"
echo "  tail -f $RUN_DIR/*.pbs.log"
