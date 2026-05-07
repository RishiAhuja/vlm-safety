#!/usr/bin/env python3
"""Aggregate repeated experiment runs into paper-ready tables and reports."""
import argparse
import csv
import json
import math
from collections import Counter, defaultdict
from pathlib import Path


AXES = ["Neutral", "West-sensitive", "East-sensitive"]


def base_model(model_key: str) -> str:
    for suffix in ("_west", "_east", "_none"):
        if model_key.endswith(suffix):
            return model_key[: -len(suffix)]
    return model_key


def persona(model_key: str) -> str:
    if model_key.endswith("_west"):
        return "western"
    if model_key.endswith("_east"):
        return "eastern"
    if model_key.endswith("_none"):
        return "none"
    return "unknown"


def load_rows(results_root: Path) -> list[dict]:
    rows: list[dict] = []
    for path in sorted(results_root.glob("run_*/inference_results_scored.json")):
        with path.open(encoding="utf-8") as f:
            run_rows = json.load(f)
        rows.extend(run_rows)
    if not rows:
        raise SystemExit(f"No replicate scored files found under {results_root}")
    return rows


def valid_rows(rows: list[dict]) -> list[dict]:
    return [r for r in rows if r.get("score") in (0, 1, 2)]


def wilson_ci(successes: int, total: int, z: float = 1.96) -> tuple[float, float]:
    if total == 0:
        return (math.nan, math.nan)
    p = successes / total
    denom = 1 + z * z / total
    center = (p + z * z / (2 * total)) / denom
    margin = z * math.sqrt((p * (1 - p) + z * z / (4 * total)) / total) / denom
    return (center - margin, center + margin)


def two_prop_z(a_success: int, a_total: int, b_success: int, b_total: int) -> tuple[float, float]:
    if min(a_total, b_total) == 0:
        return (math.nan, math.nan)
    pooled = (a_success + b_success) / (a_total + b_total)
    se = math.sqrt(pooled * (1 - pooled) * (1 / a_total + 1 / b_total))
    if se == 0:
        return (math.nan, math.nan)
    z = (a_success / a_total - b_success / b_total) / se
    p = math.erfc(abs(z) / math.sqrt(2))
    return z, p


def odds_ratio(a_success: int, a_total: int, b_success: int, b_total: int) -> float:
    # Haldane-Anscombe correction handles zero cells from saturated model behavior.
    a_fail = a_total - a_success
    b_fail = b_total - b_success
    return ((a_success + 0.5) * (b_fail + 0.5)) / ((a_fail + 0.5) * (b_success + 0.5))


def summarize_group(rows: list[dict]) -> dict:
    valid = valid_rows(rows)
    score2 = sum(1 for r in valid if r.get("score") == 2)
    avg = sum(r["score"] for r in valid) / len(valid) if valid else math.nan
    low, high = wilson_ci(score2, len(valid)) if valid else (math.nan, math.nan)
    return {
        "total_rows": len(rows),
        "valid_n": len(valid),
        "inference_errors": sum(1 for r in rows if r.get("score") == -1),
        "score_0": sum(1 for r in valid if r.get("score") == 0),
        "score_1": sum(1 for r in valid if r.get("score") == 1),
        "score_2": score2,
        "asr": score2 / len(valid) if valid else math.nan,
        "asr_ci_low": low,
        "asr_ci_high": high,
        "avg_score": avg,
    }


def write_csv(path: Path, rows: list[dict], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def aggregate(rows: list[dict], output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)

    flat_rows = []
    for r in rows:
        flat_rows.append({
            "run_id": r.get("run_id", ""),
            "stimulus": r.get("stimulus", ""),
            "category": r.get("category", ""),
            "axis": r.get("axis", ""),
            "model": r.get("model", ""),
            "base_model": base_model(r.get("model", "")),
            "persona": persona(r.get("model", "")),
            "model_origin": r.get("model_origin", ""),
            "score": r.get("score", ""),
            "failure_mode": r.get("failure_mode", ""),
            "elapsed_s": r.get("elapsed_s", ""),
            "judge_reason": r.get("judge_reason", ""),
        })
    write_csv(
        output_dir / "all_replicate_rows.csv",
        flat_rows,
        [
            "run_id", "stimulus", "category", "axis", "model", "base_model",
            "persona", "model_origin", "score", "failure_mode", "elapsed_s",
            "judge_reason",
        ],
    )

    tables: dict[str, list[dict]] = defaultdict(list)
    group_specs = {
        "asr_by_origin_axis.csv": ["model_origin", "axis"],
        "asr_by_model_axis.csv": ["model", "model_origin", "persona", "axis"],
        "asr_by_category_axis.csv": ["category", "axis"],
        "asr_by_run_origin_axis.csv": ["run_id", "model_origin", "axis"],
    }
    for filename, keys in group_specs.items():
        grouped: dict[tuple, list[dict]] = defaultdict(list)
        for r in rows:
            grouped[tuple(r.get(k, persona(r.get("model", "")) if k == "persona" else "") for k in keys)].append(r)
        for key, entries in sorted(grouped.items()):
            summary = summarize_group(entries)
            row = dict(zip(keys, key))
            row.update({
                k: f"{v:.6f}" if isinstance(v, float) else v
                for k, v in summary.items()
            })
            tables[filename].append(row)
        write_csv(output_dir / filename, tables[filename], keys + [
            "total_rows", "valid_n", "inference_errors", "score_0", "score_1",
            "score_2", "asr", "asr_ci_low", "asr_ci_high", "avg_score",
        ])

    failure_rows = []
    for mode, n in sorted(Counter(r.get("failure_mode", "unknown") for r in rows).items()):
        failure_rows.append({"failure_mode": mode, "count": n, "percent": f"{n / len(rows):.6f}"})
    write_csv(output_dir / "failure_modes.csv", failure_rows, ["failure_mode", "count", "percent"])

    run_rows = []
    for run_id in sorted(set(r.get("run_id", "") for r in rows)):
        summary = summarize_group([r for r in rows if r.get("run_id") == run_id])
        run_rows.append({
            "run_id": run_id,
            **{k: f"{v:.6f}" if isinstance(v, float) else v for k, v in summary.items()},
        })
    write_csv(output_dir / "run_summary.csv", run_rows, [
        "run_id", "total_rows", "valid_n", "inference_errors", "score_0",
        "score_1", "score_2", "asr", "asr_ci_low", "asr_ci_high", "avg_score",
    ])


def write_stat_report(rows: list[dict], output_dir: Path) -> None:
    valid = valid_rows(rows)

    def subset(**kwargs: str) -> list[dict]:
        return [r for r in valid if all(r.get(k) == v for k, v in kwargs.items())]

    tests = [
        (
            "H1 eastern-origin: West-sensitive ASR > East-sensitive ASR",
            subset(model_origin="Eastern", axis="West-sensitive"),
            subset(model_origin="Eastern", axis="East-sensitive"),
            "West-sensitive",
            "East-sensitive",
        ),
        (
            "H2 western-origin: East-sensitive ASR > West-sensitive ASR",
            subset(model_origin="Western", axis="East-sensitive"),
            subset(model_origin="Western", axis="West-sensitive"),
            "East-sensitive",
            "West-sensitive",
        ),
    ]

    for bm in sorted(set(base_model(r.get("model", "")) for r in valid)):
        east_persona = [r for r in valid if base_model(r.get("model", "")) == bm and persona(r.get("model", "")) == "eastern" and r.get("axis") == "East-sensitive"]
        none_persona = [r for r in valid if base_model(r.get("model", "")) == bm and persona(r.get("model", "")) == "none" and r.get("axis") == "East-sensitive"]
        if east_persona and none_persona:
            tests.append((
                f"H3 {bm}: eastern persona suppresses East-sensitive ASR vs no persona",
                none_persona,
                east_persona,
                "no persona",
                "eastern persona",
            ))

    lines = [
        "# Aggregate Statistical Checks",
        "",
        f"Rows: {len(rows)} total, {len(valid)} valid, {sum(1 for r in rows if r.get('score') == -1)} inference errors.",
        "ASR is computed over valid rows only; inference errors are reported separately.",
        "",
        "## Two-Proportion Tests",
        "",
    ]
    for title, a, b, a_label, b_label in tests:
        a_success = sum(1 for r in a if r.get("score") == 2)
        b_success = sum(1 for r in b if r.get("score") == 2)
        z, p = two_prop_z(a_success, len(a), b_success, len(b))
        or_value = odds_ratio(a_success, len(a), b_success, len(b))
        lines.extend([
            f"### {title}",
            f"- {a_label}: {a_success}/{len(a)} = {a_success / len(a):.3f}",
            f"- {b_label}: {b_success}/{len(b)} = {b_success / len(b):.3f}",
            f"- Difference: {a_success / len(a) - b_success / len(b):.3f}",
            f"- Odds ratio: {or_value:.3f}",
            f"- Two-sided z-test: z={z:.3f}, p={p:.6f}",
            "",
        ])

    lines.extend([
        "## Logistic Regression",
        "",
        "Run `python analyze_replicates.py --logit` in an environment with pandas and statsmodels",
        "to fit `full_engagement ~ model_origin * axis + persona + category` with stimulus-clustered SEs.",
        "The CSV tables above are the source of truth if optional dependencies are unavailable.",
        "",
    ])
    (output_dir / "statistical_report.md").write_text("\n".join(lines), encoding="utf-8")


def maybe_logit(rows: list[dict], output_dir: Path) -> None:
    try:
        import pandas as pd
        import statsmodels.formula.api as smf
    except ImportError as exc:
        (output_dir / "logistic_regression.txt").write_text(
            f"Optional logistic regression skipped: {exc}\n"
            "Install pandas and statsmodels to run this model.\n",
            encoding="utf-8",
        )
        return

    data = []
    for r in valid_rows(rows):
        if r.get("axis") == "Neutral":
            continue
        data.append({
            "full_engagement": 1 if r.get("score") == 2 else 0,
            "model_origin": r.get("model_origin", ""),
            "axis": r.get("axis", ""),
            "persona": persona(r.get("model", "")),
            "category": r.get("category", ""),
            "stimulus": r.get("stimulus", ""),
        })
    df = pd.DataFrame(data)
    result = smf.logit(
        "full_engagement ~ C(model_origin) * C(axis) + C(persona) + C(category)",
        data=df,
    ).fit(disp=False, cov_type="cluster", cov_kwds={"groups": df["stimulus"]})
    (output_dir / "logistic_regression.txt").write_text(str(result.summary()), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Aggregate repeated VLM safety runs")
    parser.add_argument("--results-root", default="results/replicates")
    parser.add_argument("--output-dir", default="results/aggregate")
    parser.add_argument("--logit", action="store_true", help="Run optional statsmodels logistic regression")
    args = parser.parse_args()

    rows = load_rows(Path(args.results_root))
    output_dir = Path(args.output_dir)
    aggregate(rows, output_dir)
    write_stat_report(rows, output_dir)
    if args.logit:
        maybe_logit(rows, output_dir)

    valid = valid_rows(rows)
    print(f"Loaded {len(rows)} rows from {args.results_root}")
    print(f"Valid rows: {len(valid)}; inference errors: {sum(1 for r in rows if r.get('score') == -1)}")
    print(f"Overall ASR: {sum(1 for r in valid if r.get('score') == 2) / len(valid):.1%}")
    print(f"Wrote aggregate artifacts to {output_dir}")


if __name__ == "__main__":
    main()
