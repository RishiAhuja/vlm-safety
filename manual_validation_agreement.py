#!/usr/bin/env python3
"""Compute agreement between LLM judge labels and human validation labels."""
import argparse
import csv
from collections import Counter, defaultdict
from pathlib import Path


def cohen_kappa(pairs: list[tuple[str, str]]) -> float:
    if not pairs:
        return float("nan")
    total = len(pairs)
    observed = sum(1 for a, b in pairs if a == b) / total
    left = Counter(a for a, _ in pairs)
    right = Counter(b for _, b in pairs)
    labels = set(left) | set(right)
    expected = sum((left[label] / total) * (right[label] / total) for label in labels)
    if expected == 1:
        return 1.0
    return (observed - expected) / (1 - expected)


def main() -> None:
    parser = argparse.ArgumentParser(description="Compute manual validation agreement")
    parser.add_argument("--input", default="results/manual_validation_sample.csv")
    parser.add_argument("--output", default="results/manual_validation_agreement.md")
    args = parser.parse_args()

    with Path(args.input).open(newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))

    scored = [r for r in rows if r.get("manual_score", "").strip() != ""]
    if not scored:
        raise SystemExit("No manual_score values found. Fill the manual columns first.")

    score_pairs = [(r["score"], r["manual_score"].strip()) for r in scored]
    score_agree = sum(a == b for a, b in score_pairs)

    mode_rows = [r for r in scored if r.get("manual_failure_mode", "").strip()]
    mode_pairs = [(r["failure_mode"], r["manual_failure_mode"].strip()) for r in mode_rows]
    mode_agree = sum(a == b for a, b in mode_pairs)

    matrix: dict[str, Counter] = defaultdict(Counter)
    for judge_score, manual_score in score_pairs:
        matrix[judge_score][manual_score] += 1

    labels = sorted(set(a for a, _ in score_pairs) | set(b for _, b in score_pairs), key=str)
    lines = [
        "# Manual Validation Agreement",
        "",
        f"Rows with manual score: {len(scored)}",
        f"Score exact agreement: {score_agree}/{len(scored)} = {score_agree / len(scored):.1%}",
        f"Score Cohen's kappa: {cohen_kappa(score_pairs):.3f}",
        "",
        "## Score Confusion Matrix",
        "",
        "Rows are LLM judge scores; columns are manual scores.",
        "",
        "| judge \\ manual | " + " | ".join(labels) + " |",
        "|---|" + "|".join("---" for _ in labels) + "|",
    ]
    for judge_label in labels:
        lines.append(
            f"| {judge_label} | "
            + " | ".join(str(matrix[judge_label][manual_label]) for manual_label in labels)
            + " |"
        )

    if mode_pairs:
        lines.extend([
            "",
            "## Failure Mode Agreement",
            "",
            f"Rows with manual failure mode: {len(mode_pairs)}",
            f"Failure-mode exact agreement: {mode_agree}/{len(mode_pairs)} = {mode_agree / len(mode_pairs):.1%}",
            f"Failure-mode Cohen's kappa: {cohen_kappa(mode_pairs):.3f}",
        ])

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Wrote agreement report to {output}")


if __name__ == "__main__":
    main()
