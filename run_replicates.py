#!/usr/bin/env python3
"""Run independent full-pipeline replicates without overwriting pilot results."""
import argparse
import os
import subprocess
import sys
from pathlib import Path


def run_step(step: str, env: dict[str, str], log_file: Path) -> None:
    cmd = [sys.executable, "run_all.py", "--step", step]
    with log_file.open("a", encoding="utf-8") as log:
        log.write(f"\n===== {step.upper()} =====\n")
        log.flush()
        proc = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            env=env,
        )
        assert proc.stdout is not None
        for line in proc.stdout:
            print(line, end="", flush=True)
            log.write(line)
            log.flush()
        rc = proc.wait()
        if rc != 0:
            raise SystemExit(f"step {step} failed with exit code {rc}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run repeated VLM safety experiments")
    parser.add_argument("--runs", type=int, default=3)
    parser.add_argument("--start", type=int, default=1)
    parser.add_argument("--results-root", default="results/replicates")
    parser.add_argument("--skip-infer", action="store_true")
    parser.add_argument("--skip-judge", action="store_true")
    parser.add_argument("--skip-analyze", action="store_true")
    args = parser.parse_args()

    root = Path(args.results_root)
    root.mkdir(parents=True, exist_ok=True)

    for run_num in range(args.start, args.start + args.runs):
        run_id = f"run_{run_num:02d}"
        run_dir = root / run_id
        run_dir.mkdir(parents=True, exist_ok=True)
        log_file = run_dir / "pipeline.log"

        env = os.environ.copy()
        env["RUN_ID"] = run_id
        env["INFERENCE_RESULTS_FILE"] = str(run_dir / "inference_results.json")
        env["JUDGE_INPUT_FILE"] = str(run_dir / "inference_results.json")
        env["JUDGE_OUTPUT_FILE"] = str(run_dir / "inference_results_scored.json")
        env["ANALYSIS_INPUT_FILE"] = str(run_dir / "inference_results_scored.json")
        env["SUMMARY_CSV_FILE"] = str(run_dir / "summary.csv")

        print(f"\n########## {run_id} ##########", flush=True)
        if not args.skip_infer:
            run_step("infer", env, log_file)
        if not args.skip_judge:
            run_step("judge", env, log_file)
        if not args.skip_analyze:
            run_step("analyze", env, log_file)

    print("\nAll requested replicates complete.")


if __name__ == "__main__":
    main()
