#!/usr/bin/env python3
"""Compact cluster progress summary.

This is intentionally terse: use model_smoke_status.sh when debugging logs.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any


ROOT = Path(os.environ.get("ROOT", "/Data3/it_FA0571/vlm"))
STATE_NAMES = {
    "R": "running",
    "Q": "queued",
    "H": "held",
    "E": "exiting",
    "C": "complete",
    "T": "transit",
    "S": "suspended",
    "W": "waiting",
}


def pct(done: int, total: int) -> str:
    if total <= 0:
        return "n/a"
    return f"{(100 * done / total):5.1f}%"


def short_path(path: Path) -> str:
    try:
        return str(path.relative_to(ROOT))
    except ValueError:
        return str(path)


def qstat_jobs() -> list[dict[str, str]]:
    try:
        out = subprocess.check_output(
            ["qstat", "-u", os.environ.get("USER", "")],
            text=True,
            stderr=subprocess.STDOUT,
        )
    except Exception:
        return []

    jobs = []
    for line in out.splitlines():
        fields = line.split()
        if not fields or fields[0].startswith("-") or "." not in fields[0]:
            continue
        state = next((f for f in fields[1:] if len(f) == 1 and f in STATE_NAMES), "?")
        jobs.append(
            {
                "id": fields[0],
                "queue": fields[2] if len(fields) > 2 else "?",
                "name": fields[3] if len(fields) > 3 else "?",
                "state": state,
            }
        )
    return jobs


def pgrep_model_processes() -> list[str]:
    try:
        out = subprocess.check_output(
            [
                "pgrep",
                "-u",
                os.environ.get("USER", ""),
                "-af",
                "run_hf_matrix.py|model_smoke_hf.py|ollama serve|ollama pull|python.*from_pretrained",
            ],
            text=True,
            stderr=subprocess.DEVNULL,
        )
    except Exception:
        return []
    return [line for line in out.splitlines() if line.strip()]


def load_json(path: Path, default: Any) -> Any:
    try:
        return json.loads(path.read_text())
    except Exception:
        return default


def latest_smoke_run() -> Path | None:
    marker = ROOT / "logs/model_smoke/latest_pbs_run.txt"
    if marker.exists():
        path = Path(marker.read_text().strip())
        if path.exists():
            return path
    runs = sorted((ROOT / "logs/model_smoke").glob("pbs_*"))
    return runs[-1] if runs else None


def submitted_jobs(run_dir: Path) -> list[tuple[str, str]]:
    path = run_dir / "submitted_jobs.tsv"
    if not path.exists():
        return []
    jobs = []
    for line in path.read_text().splitlines():
        parts = line.split()
        if len(parts) >= 2:
            jobs.append((parts[0], parts[1]))
    return jobs


def smoke_result(run_dir: Path, model: str) -> dict[str, Any] | None:
    path = run_dir / f"{model}.json"
    if not path.exists():
        return None
    return load_json(path, None)


def print_smoke_summary(jobs: list[dict[str, str]]) -> None:
    run_dir = latest_smoke_run()
    if not run_dir:
        print("Smoke latest: no PBS smoke run found")
        return

    state_by_job = {job["id"]: job["state"] for job in jobs}
    state_by_job.update({job["id"].split(".")[0]: job["state"] for job in jobs})
    submitted = submitted_jobs(run_dir)
    if not submitted:
        print(f"Smoke latest: no submitted jobs in {short_path(run_dir)}")
        return

    rows = []
    counts = Counter()
    for model, job_id in submitted:
        result = smoke_result(run_dir, model)
        state = state_by_job.get(job_id, state_by_job.get(job_id.split(".")[0], "done"))
        if result:
            status = str(result.get("status", "done"))
            done = 1
        elif state == "R":
            status = "running"
            done = 0
        elif state in {"Q", "H", "W"}:
            status = STATE_NAMES.get(state, state)
            done = 0
        else:
            status = "pending"
            done = 0
        counts[status] += 1
        rows.append((model, done, status, job_id, state))

    completed = sum(done for _, done, _, _, _ in rows)
    total = len(rows)
    ok = counts["ok"]
    err = counts["error"]
    active = counts["running"] + counts["queued"] + counts["held"] + counts["waiting"]
    print(
        f"Smoke latest: {completed}/{total} done ({pct(completed, total)}) | "
        f"ok={ok} error={err} active={active} | {short_path(run_dir)}"
    )
    for model, done, status, job_id, state in rows:
        state_label = STATE_NAMES.get(state, state)
        print(f"  {model:10} {done}/1 {pct(done, 1)} {status:8} {job_id:13} {state_label}")


def print_smoke_readiness() -> None:
    latest: dict[str, tuple[float, dict[str, Any], Path]] = {}
    for path in (ROOT / "logs/model_smoke").glob("pbs_*/*.json"):
        result = load_json(path, None)
        if not isinstance(result, dict):
            continue
        model = result.get("model") or path.stem
        mtime = path.stat().st_mtime
        if model not in latest or mtime > latest[model][0]:
            latest[model] = (mtime, result, path)

    if not latest:
        return

    ok = sum(1 for _, result, _ in latest.values() if result.get("status") == "ok")
    total = len(latest)
    print(f"Smoke readiness: {ok}/{total} models ok ({pct(ok, total)})")
    for model in sorted(latest):
        _, result, _ = latest[model]
        status = result.get("status", "?")
        elapsed = result.get("elapsed_s", "?")
        print(f"  {model:10} {status:8} {elapsed}s")


def expected_matrix() -> tuple[list[str], list[str]]:
    sys.path.insert(0, str(ROOT))
    try:
        from config import STIMULI, VLM_MODELS  # type: ignore

        stimuli = [row[0] for row in STIMULI]
        models = list(VLM_MODELS)
        return stimuli, models
    except Exception:
        return [], []


def print_inference_summary() -> None:
    results_path = ROOT / "results/inference_results.json"
    scored_path = ROOT / "results/inference_results_scored.json"
    if not results_path.exists():
        print("Inference matrix: no results/inference_results.json yet")
        return

    results = load_json(results_path, [])
    scored = load_json(scored_path, []) if scored_path.exists() else []
    if not isinstance(results, list):
        print("Inference matrix: results file is not a list")
        return

    stimuli, configured_models = expected_matrix()
    result_models = sorted({str(row.get("model")) for row in results if row.get("model")})
    models = configured_models or result_models
    per_model_total = len(stimuli) if stimuli else 0
    total = len(models) * per_model_total if per_model_total else len(results)

    done_pairs = {
        (row.get("model"), row.get("stimulus"))
        for row in results
        if row.get("model")
        and row.get("stimulus")
        and not str(row.get("response", "")).startswith("ERROR:")
    }
    judged_pairs = {
        (row.get("model"), row.get("stimulus"))
        for row in scored
        if row.get("model") and row.get("stimulus") and row.get("score") is not None
    }

    done = len(done_pairs)
    judged = len(judged_pairs)
    print(
        f"Inference matrix: {done}/{total} responses ({pct(done, total)}) | "
        f"judged={judged}/{total} ({pct(judged, total)})"
    )

    if not models:
        return
    per_model_done = defaultdict(int)
    per_model_judged = defaultdict(int)
    for model, _stimulus in done_pairs:
        per_model_done[str(model)] += 1
    for model, _stimulus in judged_pairs:
        per_model_judged[str(model)] += 1

    for model in models:
        model_total = per_model_total or max(per_model_done.get(model, 0), per_model_judged.get(model, 0))
        print(
            f"  {model:20} "
            f"responses {per_model_done.get(model, 0):>3}/{model_total:<3} {pct(per_model_done.get(model, 0), model_total)} "
            f"judged {per_model_judged.get(model, 0):>3}/{model_total:<3}"
        )



def latest_hf_matrix_run() -> tuple[Path, Path] | None:
    run_marker = ROOT / "logs/hf_matrix/latest_run.txt"
    results_marker = ROOT / "logs/hf_matrix/latest_results.txt"
    if not run_marker.exists() or not results_marker.exists():
        return None
    run_dir = Path(run_marker.read_text().strip())
    results_file = Path(results_marker.read_text().strip())
    if not run_dir.exists():
        return None
    return run_dir, results_file


def print_hf_matrix_summary(jobs: list[dict[str, str]]) -> None:
    latest = latest_hf_matrix_run()
    if not latest:
        print("HF expanded matrix: no run submitted yet")
        return

    run_dir, results_file = latest
    submitted_path = run_dir / "submitted_jobs.tsv"
    submitted = []
    if submitted_path.exists():
        for line in submitted_path.read_text().splitlines():
            parts = line.split()
            if len(parts) >= 3:
                submitted.append((parts[0], parts[1], parts[2]))

    results = load_json(results_file, [])
    if not isinstance(results, list):
        results = []

    stimuli, _models = expected_matrix()
    per_job_total = len(stimuli) if stimuli else 48
    expected_total = len(submitted) * per_job_total
    done_pairs = {
        (row.get("model"), row.get("stimulus"))
        for row in results
        if row.get("model")
        and row.get("stimulus")
        and not str(row.get("response", "")).startswith("ERROR:")
    }
    error_pairs = {
        (row.get("model"), row.get("stimulus"))
        for row in results
        if row.get("model")
        and row.get("stimulus")
        and str(row.get("response", "")).startswith("ERROR:")
    }

    state_by_job = {job["id"]: job["state"] for job in jobs}
    state_by_job.update({job["id"].split(".")[0]: job["state"] for job in jobs})
    print(
        f"HF expanded matrix: {len(done_pairs)}/{expected_total} responses "
        f"({pct(len(done_pairs), expected_total)}) | errors={len(error_pairs)} | {short_path(run_dir)}"
    )
    for model, persona, job_id in submitted:
        key_prefix = f"{model}_{persona}"
        done = sum(1 for item in done_pairs if item[0] == key_prefix)
        errors = sum(1 for item in error_pairs if item[0] == key_prefix)
        state_key = state_by_job.get(job_id, state_by_job.get(job_id.split(".")[0], "done"))
        state = STATE_NAMES.get(state_key, state_key)
        print(
            f"  {key_prefix:20} responses {done:>3}/{per_job_total:<3} {pct(done, per_job_total)} "
            f"errors {errors:<3} {state}"
        )


def main() -> int:
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    jobs = qstat_jobs()
    processes = pgrep_model_processes()
    job_counts = Counter(job["state"] for job in jobs)

    print(f"Time: {now}")
    print(
        "PBS: "
        f"running={job_counts.get('R', 0)} "
        f"queued={job_counts.get('Q', 0)} "
        f"held={job_counts.get('H', 0)} "
        f"total={len(jobs)}"
    )
    if processes:
        print(f"Processes: {len(processes)} matching model process(es)")
    else:
        print("Processes: none")

    print()
    print_smoke_summary(jobs)
    print()
    print_smoke_readiness()
    print()
    print_inference_summary()
    print()
    print_hf_matrix_summary(jobs)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
