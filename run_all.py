#!/usr/bin/env python3
"""
VLM Cultural Safety Pilot — Full Pipeline
Run all steps: generate stimuli → inference → judge → analyze

Usage:
    python run_all.py              # run everything
    python run_all.py --step gen   # only generate stimuli
    python run_all.py --step infer # only run inference
    python run_all.py --step judge # only run judge scoring
    python run_all.py --step analyze  # only analyze results
"""
import argparse
import sys
import time


def step_generate():
    print("\n" + "=" * 60)
    print("  STEP 1: Generating typographic stimulus images")
    print("=" * 60 + "\n")
    from generate_stimuli import generate_all
    generate_all()


def step_inference():
    print("\n" + "=" * 60)
    print("  STEP 2: Running VLM inference via Ollama")
    print("=" * 60)
    from run_inference import run_inference
    run_inference()


def step_judge():
    print("\n" + "=" * 60)
    print("  STEP 3: Scoring responses with LLM judge")
    print("=" * 60)
    from run_judge import run_judge
    run_judge()


def step_analyze():
    print("\n" + "=" * 60)
    print("  STEP 4: Analyzing results")
    print("=" * 60)
    from analyze import analyze
    analyze()


STEPS = {
    "gen": step_generate,
    "infer": step_inference,
    "judge": step_judge,
    "analyze": step_analyze,
}


def main():
    parser = argparse.ArgumentParser(description="VLM Cultural Safety Pilot")
    parser.add_argument(
        "--step",
        choices=list(STEPS.keys()),
        help="Run a single step instead of the full pipeline",
    )
    args = parser.parse_args()

    if args.step:
        STEPS[args.step]()
        return

    # Full pipeline
    print("\n" + "#" * 60)
    print("  VLM CULTURAL SAFETY PILOT — FULL PIPELINE")
    print("#" * 60)

    t0 = time.time()
    try:
        step_generate()
        step_inference()
        step_judge()
        step_analyze()
    except KeyboardInterrupt:
        print("\n\nInterrupted. Partial results are saved.")
        sys.exit(1)

    elapsed = time.time() - t0
    mins = int(elapsed // 60)
    secs = int(elapsed % 60)
    print(f"\n{'#' * 60}")
    print(f"  PIPELINE COMPLETE — {mins}m {secs}s total")
    print(f"{'#' * 60}\n")


if __name__ == "__main__":
    main()
