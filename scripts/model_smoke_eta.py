#!/usr/bin/env python3
"""Summarize PBS smoke-test progress with a rough compute ETA."""

from __future__ import annotations

import json
import os
import re
import subprocess
import sys
from datetime import datetime
from pathlib import Path


# Conservative rough estimates for one cached single-image smoke run.
# Queue wait is not included because PBS scheduling is external.
ESTIMATE_S = {
    "phi": 120,
    "glm": 420,
    "internvl": 180,
    "molmo": 900,
    "cogvlm2": 1200,
    "deepseek": 1200,
}


def fmt_s(seconds: float | int | None) -> str:
    if seconds is None:
        return "unknown"
    seconds = max(0, int(seconds))
    h, r = divmod(seconds, 3600)
    m, s = divmod(r, 60)
    if h:
        return f"{h}h{m:02d}m"
    if m:
        return f"{m}m{s:02d}s"
    return f"{s}s"


def latest_run_dir(root: Path, explicit: str | None) -> Path | None:
    if explicit:
        return Path(explicit)
    marker = root / "logs/model_smoke/latest_pbs_run.txt"
    if marker.exists():
        return Path(marker.read_text().strip())
    return None


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


def qstat_states(user: str) -> dict[str, str]:
    try:
        out = subprocess.check_output(["qstat", "-u", user], text=True, stderr=subprocess.STDOUT)
    except Exception:
        return {}
    states = {}
    for line in out.splitlines():
        fields = line.split()
        if not fields or fields[0].startswith("-") or fields[0].lower() == "job":
            continue
        job_id = fields[0]
        state = next((f for f in fields[1:] if len(f) == 1 and f in "QREHSTCW"), "")
        if state:
            states[job_id] = state
            states[job_id.split(".")[0]] = state
    return states


def log_text(run_dir: Path, model: str) -> str:
    chunks = []
    for name in (f"{model}.log", f"{model}.pbs.log"):
        path = run_dir / name
        if path.exists():
            chunks.append(path.read_text(errors="ignore"))
    return "\n".join(chunks)


def last_phase(text: str) -> str:
    phases = re.findall(r"PHASE\s+([^\n\r]+)", text)
    if phases:
        return phases[-1].strip()
    if "START " in text:
        return "started"
    return "-"


def start_time(text: str) -> datetime | None:
    match = re.search(r"\[(\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2})\]\s+START", text)
    if not match:
        return None
    try:
        return datetime.fromisoformat(match.group(1))
    except ValueError:
        return None


def visible_gpu(text: str, result: dict | None) -> str:
    if result and result.get("cuda_visible_devices"):
        return str(result["cuda_visible_devices"])
    match = re.search(r"CUDA_VISIBLE_DEVICES=([^\s]+)", text)
    return match.group(1) if match else "-"


def read_result(run_dir: Path, model: str) -> dict | None:
    path = run_dir / f"{model}.json"
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text())
    except Exception:
        return None


def main() -> int:
    root = Path(os.environ.get("ROOT", "/Data3/it_FA0571/vlm"))
    run_dir = latest_run_dir(root, sys.argv[1] if len(sys.argv) > 1 else None)
    if not run_dir or not run_dir.exists():
        print("ETA: no PBS smoke run directory found.")
        return 0

    jobs = submitted_jobs(run_dir)
    if not jobs:
        print(f"ETA: no submitted_jobs.tsv in {run_dir}")
        return 0

    states = qstat_states(os.environ.get("USER", ""))
    now = datetime.now()
    remaining_total = 0
    unfinished = 0

    print("===== ETA summary =====")
    print("Queue wait is not included; ETA is rough remaining compute time after PBS starts jobs.")
    print(f"Run dir: {run_dir}")
    print(f"{'model':10} {'job':18} {'state':7} {'status':8} {'gpu':18} {'phase':28} {'elapsed':>9} {'eta':>9}")

    for model, job_id in jobs:
        result = read_result(run_dir, model)
        text = log_text(run_dir, model)
        phase = last_phase(text)[:28]
        state = states.get(job_id, states.get(job_id.split(".")[0], "done/none"))
        status = result.get("status", "done") if result else ("running" if state == "R" else "queued")
        gpu = visible_gpu(text, result)[:18]

        elapsed = None
        eta = None
        if result:
            elapsed = float(result.get("elapsed_s") or 0)
            eta = 0
        else:
            st = start_time(text)
            if st:
                elapsed = (now - st).total_seconds()
            estimate = ESTIMATE_S.get(model, 600)
            if state == "R" and elapsed is not None:
                eta = max(estimate - elapsed, 0)
            else:
                eta = estimate
            remaining_total += eta
            unfinished += 1

        print(
            f"{model:10} {job_id[:18]:18} {state:7} {status:8} "
            f"{gpu:18} {phase:28} {fmt_s(elapsed):>9} {fmt_s(eta):>9}"
        )

    print()
    if unfinished:
        print(f"Rough remaining compute ETA: {fmt_s(remaining_total)} plus PBS queue wait.")
    else:
        print("All submitted smoke jobs have produced JSON results.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
