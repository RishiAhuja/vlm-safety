#!/usr/bin/env python3
"""Create a structured, timestamped snapshot of experiment results.

This script never moves or deletes live result files. It copies whatever exists
into results/structured/snapshots/<timestamp>/ and writes manifests so the
snapshot is auditable even while long-running jobs are still producing outputs.
"""

from __future__ import annotations

import csv
import hashlib
import json
import os
import shutil
import subprocess
from datetime import datetime
from pathlib import Path
from typing import Any

ROOT = Path(os.environ.get("ROOT", "/Data3/it_FA0571/vlm"))
RESULTS = ROOT / "results"
STRUCTURED = RESULTS / "structured"
SNAPSHOT_NAME = os.environ.get("RESULTS_SNAPSHOT_NAME") or datetime.now().strftime("%Y%m%d_%H%M%S")
SNAPSHOT = STRUCTURED / "snapshots" / SNAPSHOT_NAME

EXPECTED_ROWS = {
    "pilot_raw": 432,
    "pilot_scored": 432,
    "replicate_run": 432,
    "expanded_raw": 1728,
    "expanded_scored": 1728,
    "ollama_control": 864,
    "hf_control": 864,
    "manual_validation_sample": 100,
}

COPY_ITEMS: list[tuple[str, str, str, str]] = [
    # group, source relative path, destination relative path, note
    ("pilot", "results/inference_results.json", "01_pilot/inference_results_raw.json", "Original 432-row pilot responses."),
    ("pilot", "results/inference_results_scored.json", "01_pilot/inference_results_scored_original_judge.json", "Original pilot judged file, kept for provenance."),
    ("pilot", "results/inference_results_scored_gpt55.json", "01_pilot/inference_results_scored_gpt55.json", "Pilot rejudged with GPT-5.5."),
    ("pilot", "results/summary.csv", "01_pilot/summary.csv", "Original pilot summary."),
    ("replicates", "results/replicates", "02_replicates", "Three independent replicate runs and judged outputs."),
    ("aggregate", "results/aggregate", "03_replicate_aggregate", "Aggregate replicate summaries and regression output."),
    ("expanded", "results/expanded", "04_expanded_matrix", "Expanded 12-model matrix raw/scored outputs and manifests."),
    ("controls", "results/controls/ollama_text_only_results.json", "05_controls/ollama_text_only_results.json", "Plain-text control responses for Ollama models."),
    ("controls", "results/controls/ollama_text_only_scored_gpt55.json", "05_controls/ollama_text_only_scored_gpt55.json", "GPT-5.5 judged plain-text control responses."),
    ("controls", "results/controls/ollama_ocr_results.json", "05_controls/ollama_ocr_results.json", "Ollama OCR-only control responses."),
    ("controls", "results/controls/hf_ocr_results.json", "05_controls/hf_ocr_results.json", "HF OCR-only control responses."),
    ("controls", "results/controls/ollama_ocr_then_answer_results.json", "05_controls/ollama_ocr_then_answer_results.json", "Prompt-wrapper robustness responses: transcribe first, then answer."),
    ("controls", "results/controls/ollama_ocr_then_answer_scored_gpt55.json", "05_controls/ollama_ocr_then_answer_scored_gpt55.json", "GPT-5.5 judged prompt-wrapper robustness responses."),
    ("control_smoke", "results/controls/hf_ocr_smoke.json", "05_controls/smoke/hf_ocr_smoke.json", "HF OCR smoke test."),
    ("control_smoke", "results/controls/ollama_ocr_smoke.json", "05_controls/smoke/ollama_ocr_smoke.json", "Ollama OCR smoke test."),
    ("control_smoke", "results/controls/ollama_text_smoke.json", "05_controls/smoke/ollama_text_smoke.json", "Ollama text-only smoke test."),
    ("gpt55_analysis", "results/gpt55_analysis_report.md", "06_analysis/gpt55_analysis_report.md", "Main GPT-5.5 analysis report."),
    ("gpt55_analysis", "results/gpt55_asr_by_model_axis.csv", "06_analysis/gpt55_asr_by_model_axis.csv", "GPT-5.5 ASR by model/axis."),
    ("gpt55_analysis", "results/gpt55_asr_by_origin_axis.csv", "06_analysis/gpt55_asr_by_origin_axis.csv", "GPT-5.5 ASR by origin/axis."),
    ("gpt55_analysis", "results/gpt55_expanded_by_persona_origin_axis.csv", "06_analysis/gpt55_expanded_by_persona_origin_axis.csv", "Expanded GPT-5.5 table by persona/origin/axis."),
    ("gpt55_analysis", "results/gpt55_judge_summary.json", "06_analysis/gpt55_judge_summary.json", "GPT-5.5 judge summary."),
    ("advanced", "results/advanced_tests", "07_advanced_tests", "Robustness, bootstrap, leave-one-out, size/edge analyses."),
    ("control_analysis", "results/control_analysis", "08_control_analysis", "Control-analysis report and tables."),
    ("manual_validation", "results/manual_validation_sample.csv", "09_manual_validation/manual_validation_sample.csv", "100-row manual validation sample."),
]


def rel(path: Path) -> str:
    try:
        return str(path.relative_to(ROOT))
    except ValueError:
        return str(path)


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def json_row_count(path: Path) -> int | None:
    try:
        data = json.loads(path.read_text())
    except Exception:
        return None
    if isinstance(data, list):
        return len(data)
    if isinstance(data, dict):
        return len(data)
    return None


def csv_row_count(path: Path) -> int | None:
    try:
        with path.open(newline="") as f:
            return max(sum(1 for _ in csv.reader(f)) - 1, 0)
    except Exception:
        return None


def row_count(path: Path) -> int | None:
    if path.suffix == ".json":
        return json_row_count(path)
    if path.suffix == ".csv":
        return csv_row_count(path)
    return None


def copy_file(src: Path, dst: Path) -> dict[str, Any]:
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)
    return {
        "source": rel(src),
        "snapshot_path": rel(dst),
        "exists": True,
        "bytes": dst.stat().st_size,
        "sha256": sha256(dst),
        "row_count": row_count(dst),
    }


def copy_item(group: str, source_rel: str, dest_rel: str, note: str) -> list[dict[str, Any]]:
    src = ROOT / source_rel
    dst = SNAPSHOT / dest_rel
    if not src.exists():
        return [{"group": group, "source": source_rel, "snapshot_path": dest_rel, "exists": False, "note": note}]
    if src.is_file():
        item = copy_file(src, dst)
        item.update({"group": group, "note": note})
        return [item]

    out = []
    for file in sorted(p for p in src.rglob("*") if p.is_file()):
        rel_inside = file.relative_to(src)
        item = copy_file(file, dst / rel_inside)
        item.update({"group": group, "note": note})
        out.append(item)
    return out or [{"group": group, "source": source_rel, "snapshot_path": dest_rel, "exists": True, "note": note, "bytes": 0}]


def run_text(cmd: list[str]) -> str:
    try:
        return subprocess.check_output(cmd, cwd=ROOT, text=True, stderr=subprocess.STDOUT, timeout=60)
    except Exception as exc:
        return f"ERROR running {' '.join(cmd)}: {exc}\n"


def write_status_files() -> None:
    status_dir = SNAPSHOT / "00_status"
    status_dir.mkdir(parents=True, exist_ok=True)
    (status_dir / "progress_snapshot.txt").write_text(run_text(["scripts/progress.sh"]))
    (status_dir / "qstat_snapshot.txt").write_text(run_text(["qstat", "-u", os.environ.get("USER", "")]))
    head = run_text(["git", "log", "--oneline", "-5"])
    status = run_text(["git", "status", "--short"])
    (status_dir / "git_snapshot.txt").write_text(head + "\n--- status --\n" + status)


def classify_expected(entry: dict[str, Any]) -> str:
    path = entry.get("source", "")
    rows = entry.get("row_count")
    if rows is None:
        return "unknown"
    if path.endswith("results/inference_results.json"):
        return "complete" if rows == EXPECTED_ROWS["pilot_raw"] else "partial"
    if path.endswith("results/inference_results_scored_gpt55.json"):
        return "complete" if rows == EXPECTED_ROWS["pilot_scored"] else "partial"
    if "/replicates/run_" in path and path.endswith("inference_results.json"):
        return "complete" if rows == EXPECTED_ROWS["replicate_run"] else "partial"
    if "/replicates/run_" in path and path.endswith("inference_results_scored_gpt55.json"):
        return "complete" if rows == EXPECTED_ROWS["replicate_run"] else "partial"
    if path.endswith("results/expanded/inference_results.json") or path.endswith("results/expanded/inference_results_scored_gpt55.json"):
        return "complete" if rows == EXPECTED_ROWS["expanded_raw"] else "partial"
    if (
        path.endswith("ollama_text_only_results.json")
        or path.endswith("ollama_text_only_scored_gpt55.json")
        or path.endswith("ollama_ocr_results.json")
        or path.endswith("ollama_ocr_then_answer_results.json")
        or path.endswith("ollama_ocr_then_answer_scored_gpt55.json")
    ):
        return "complete" if rows == EXPECTED_ROWS["ollama_control"] else "partial"
    if path.endswith("hf_ocr_results.json"):
        return "complete" if rows == EXPECTED_ROWS["hf_control"] else "partial"
    if path.endswith("manual_validation_sample.csv"):
        return "complete" if rows == EXPECTED_ROWS["manual_validation_sample"] else "partial"
    return "unknown"


def write_readme(manifest: list[dict[str, Any]]) -> None:
    grouped: dict[str, list[dict[str, Any]]] = {}
    for item in manifest:
        grouped.setdefault(str(item.get("group", "other")), []).append(item)

    lines = [
        "# Structured Results Snapshot",
        "",
        f"Snapshot: `{SNAPSHOT_NAME}`",
        f"Generated: {datetime.now().isoformat(timespec='seconds')}",
        "",
        "This snapshot is copied from the live `results/` tree. It does not move or delete live files.",
        "Files marked partial were copied while overnight jobs were still running.",
        "",
        "## Layout",
        "",
        "- `00_status/`: progress, qstat, and git snapshots.",
        "- `01_pilot/`: original pilot outputs.",
        "- `02_replicates/`: three repeat runs.",
        "- `03_replicate_aggregate/`: replicate aggregate analysis.",
        "- `04_expanded_matrix/`: expanded 12-model matrix.",
        "- `05_controls/`: text-only, OCR, and prompt-wrapper controls.",
        "- `06_analysis/`: GPT-5.5 analysis tables/reports.",
        "- `07_advanced_tests/`: robustness/statistical analyses.",
        "- `08_control_analysis/`: control interpretation tables.",
        "- `09_manual_validation/`: human-labeling sample.",
        "",
        "## Manifest Summary",
        "",
    ]
    for group in sorted(grouped):
        existing = [x for x in grouped[group] if x.get("exists")]
        missing = [x for x in grouped[group] if not x.get("exists")]
        partial = [x for x in existing if x.get("completion_status") == "partial"]
        lines.append(f"- `{group}`: {len(existing)} files copied, {len(missing)} missing, {len(partial)} partial.")
    lines.extend([
        "",
        "See `manifest.json` for file hashes and row counts, and `completion_summary.csv` for complete/partial status.",
    ])
    (SNAPSHOT / "README.md").write_text("\n".join(lines) + "\n")


def main() -> int:
    SNAPSHOT.mkdir(parents=True, exist_ok=True)
    write_status_files()

    manifest: list[dict[str, Any]] = []
    for group, source_rel, dest_rel, note in COPY_ITEMS:
        manifest.extend(copy_item(group, source_rel, dest_rel, note))

    for entry in manifest:
        if entry.get("exists"):
            entry["completion_status"] = classify_expected(entry)
        else:
            entry["completion_status"] = "missing"

    (SNAPSHOT / "manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n")
    summary_rows = [
        {
            "group": item.get("group", ""),
            "source": item.get("source", ""),
            "snapshot_path": item.get("snapshot_path", ""),
            "exists": item.get("exists", False),
            "row_count": item.get("row_count", ""),
            "completion_status": item.get("completion_status", ""),
            "bytes": item.get("bytes", ""),
        }
        for item in manifest
    ]
    with (SNAPSHOT / "completion_summary.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(summary_rows[0].keys()))
        writer.writeheader()
        writer.writerows(summary_rows)

    write_readme(manifest)
    STRUCTURED.mkdir(parents=True, exist_ok=True)
    (STRUCTURED / "LATEST_SNAPSHOT.txt").write_text(str(SNAPSHOT) + "\n")
    index = [
        "# Structured Results",
        "",
        f"Latest snapshot: `{SNAPSHOT}`",
        "",
        "Run `python3 scripts/build_results_bundle.py` to create a fresh timestamped snapshot.",
    ]
    (STRUCTURED / "README.md").write_text("\n".join(index) + "\n")
    print(f"RESULTS_BUNDLE_DONE {SNAPSHOT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
