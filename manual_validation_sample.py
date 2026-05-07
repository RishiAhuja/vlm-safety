#!/usr/bin/env python3
"""Create a deterministic stratified sample for manual judge validation."""
import argparse
import csv
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path


DEFAULT_MODE_QUOTAS = {
    "inference_error": 10,
    "safety_refusal": 10,
    "ocr_failure": 10,
    "generic_caption_only": 15,
    "partial_hedged": 20,
    "full_engagement": 35,
}


def persona(model_key: str) -> str:
    if model_key.endswith("_west"):
        return "western"
    if model_key.endswith("_east"):
        return "eastern"
    if model_key.endswith("_none"):
        return "none"
    return "unknown"


def stable_key(row: dict, seed: int) -> str:
    text = "|".join([
        str(seed),
        row.get("run_id", ""),
        row.get("model", ""),
        row.get("stimulus", ""),
        row.get("failure_mode", ""),
    ])
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def load_rows(results_root: Path) -> list[dict]:
    rows: list[dict] = []
    for path in sorted(results_root.glob("run_*/inference_results_scored.json")):
        with path.open(encoding="utf-8") as f:
            rows.extend(json.load(f))
    if not rows:
        raise SystemExit(f"No replicate scored files found under {results_root}")
    return rows


def stratified_pick(candidates: list[dict], n: int, seed: int) -> list[dict]:
    buckets: dict[tuple[str, str, str], list[dict]] = defaultdict(list)
    for row in candidates:
        buckets[(row.get("model_origin", ""), row.get("axis", ""), row.get("model", ""))].append(row)
    for bucket in buckets.values():
        bucket.sort(key=lambda r: stable_key(r, seed))

    picked: list[dict] = []
    while len(picked) < n and any(buckets.values()):
        for key in sorted(buckets):
            if buckets[key] and len(picked) < n:
                picked.append(buckets[key].pop(0))
    return picked


def build_sample(rows: list[dict], sample_size: int, seed: int) -> list[dict]:
    by_mode: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        by_mode[row.get("failure_mode", "unknown")].append(row)

    sample: list[dict] = []
    used = set()

    quotas = DEFAULT_MODE_QUOTAS.copy()
    if sample_size != 100:
        scale = sample_size / 100
        quotas = {mode: max(1, round(quota * scale)) for mode, quota in quotas.items()}

    for mode, quota in quotas.items():
        picked = stratified_pick(by_mode.get(mode, []), min(quota, len(by_mode.get(mode, []))), seed)
        for row in picked:
            key = (row.get("run_id"), row.get("model"), row.get("stimulus"))
            if key not in used:
                sample.append(row)
                used.add(key)

    if len(sample) < sample_size:
        remaining = [
            r for r in rows
            if (r.get("run_id"), r.get("model"), r.get("stimulus")) not in used
        ]
        for row in stratified_pick(remaining, sample_size - len(sample), seed):
            sample.append(row)

    return sorted(sample[:sample_size], key=lambda r: stable_key(r, seed))


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate manual validation sample")
    parser.add_argument("--results-root", default="results/replicates")
    parser.add_argument("--output", default="results/manual_validation_sample.csv")
    parser.add_argument("--sample-size", type=int, default=100)
    parser.add_argument("--seed", type=int, default=20260507)
    args = parser.parse_args()

    rows = load_rows(Path(args.results_root))
    sample = build_sample(rows, args.sample_size, args.seed)

    fieldnames = [
        "sample_id", "run_id", "stimulus", "category", "axis", "model",
        "model_origin", "persona", "score", "failure_mode", "judge_reason",
        "concept", "response", "manual_score", "manual_failure_mode",
        "manual_notes",
    ]
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for i, row in enumerate(sample, 1):
            writer.writerow({
                "sample_id": f"mv_{i:03d}",
                "run_id": row.get("run_id", ""),
                "stimulus": row.get("stimulus", ""),
                "category": row.get("category", ""),
                "axis": row.get("axis", ""),
                "model": row.get("model", ""),
                "model_origin": row.get("model_origin", ""),
                "persona": persona(row.get("model", "")),
                "score": row.get("score", ""),
                "failure_mode": row.get("failure_mode", ""),
                "judge_reason": row.get("judge_reason", ""),
                "concept": row.get("concept", ""),
                "response": row.get("response", ""),
                "manual_score": "",
                "manual_failure_mode": "",
                "manual_notes": "",
            })

    modes = Counter(row.get("failure_mode", "unknown") for row in sample)
    print(f"Wrote {len(sample)} rows to {output}")
    print("Sample failure-mode counts:", dict(sorted(modes.items())))


if __name__ == "__main__":
    main()
