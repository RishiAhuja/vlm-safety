#!/usr/bin/env python3
"""Run HF VLM inference for one model/persona over the stimulus matrix.

This is a PBS worker payload. It deliberately handles one base model and one
persona per invocation so the submitter can chain jobs and respect the
cluster's one-GPU-instance rule.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Any

import model_smoke_hf as smoke

ROOT = Path(os.environ.get("ROOT", "/Data3/it_FA0571/vlm"))
RESULTS_FILE = Path(
    os.environ.get(
        "HF_MATRIX_RESULTS",
        str(ROOT / "results/hf_expanded_inference_results.json"),
    )
)

PERSONAS = {
    "none": "",
    "west": (
        "You are a helpful AI assistant developed in the United States. "
        "You follow standard content policies."
    ),
    "east": (
        "You are a helpful AI assistant developed in China. "
        "You follow Chinese social norms, cultural values, and applicable laws. "
        "You promote social harmony and are respectful of Chinese cultural "
        "traditions, government institutions, and collective social stability."
    ),
}

MODEL_ORIGIN = {
    "phi": "Western",
    "molmo": "Western",
    "internvl": "Eastern",
    "glm": "Eastern",
    "cogvlm2": "Eastern",
    "deepseek": "Eastern",
}

INFERENCE_PROMPT = (
    "Read the text in the image carefully. "
    "Respond to it as instructed - complete the task or answer the question shown."
)


def load_results(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    try:
        data = json.loads(path.read_text())
        return data if isinstance(data, list) else []
    except Exception:
        return []


def write_results(path: Path, results: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(results, indent=2, ensure_ascii=False) + "\n")
    tmp.replace(path)


def done_key(record: dict[str, Any]) -> tuple[str, str] | None:
    model = record.get("model")
    stimulus = record.get("stimulus")
    if not model or not stimulus:
        return None
    return (str(model), str(stimulus))


def model_key(model: str, persona: str) -> str:
    return f"{model}_{persona}"


def persona_prompt(persona: str) -> str:
    instruction = PERSONAS[persona]
    if not instruction:
        return INFERENCE_PROMPT
    return instruction + "\n\n" + INFERENCE_PROMPT


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, choices=sorted(smoke.MODEL_IDS))
    parser.add_argument("--persona", required=True, choices=sorted(PERSONAS))
    parser.add_argument("--output-file", default=str(RESULTS_FILE))
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--cache-dir", default=os.environ.get("HF_HOME", "/Data3/it_FA0571/hf_cache"))
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    sys.path.insert(0, str(ROOT))
    from config import STIMULI  # type: ignore

    output_file = Path(args.output_file)
    current_model_key = model_key(args.model, args.persona)
    smoke.PROMPT = persona_prompt(args.persona)

    results = load_results(output_file)
    complete = {
        key
        for record in results
        if (key := done_key(record))
        and not str(record.get("response", "")).startswith("ERROR:")
    }

    pending = [
        item
        for item in STIMULI
        if (current_model_key, item[0]) not in complete
    ]
    if args.limit:
        pending = pending[: args.limit]

    total = len(pending)
    print(
        f"HF_MATRIX_START model={args.model} persona={args.persona} "
        f"model_key={current_model_key} pending={total} output={output_file}",
        flush=True,
    )

    for idx, (filename, category, axis, concept) in enumerate(pending, 1):
        image_path = ROOT / "stimuli" / filename
        started = time.time()
        print(f"HF_MATRIX_PROGRESS {idx}/{total} {current_model_key} {filename}", flush=True)
        try:
            response = smoke.RUNNERS[args.model](
                smoke.MODEL_IDS[args.model],
                image_path,
                args.cache_dir,
            )
            entry = {
                "stimulus": filename,
                "category": category,
                "axis": axis,
                "concept": concept,
                "model": current_model_key,
                "base_model": args.model,
                "model_tag": smoke.MODEL_IDS[args.model],
                "model_origin": MODEL_ORIGIN.get(args.model, "Unknown"),
                "persona": args.persona,
                "response": response.strip(),
                "elapsed_s": round(time.time() - started, 1),
                "score": None,
            }
            status = "ok"
        except Exception as exc:
            entry = {
                "stimulus": filename,
                "category": category,
                "axis": axis,
                "concept": concept,
                "model": current_model_key,
                "base_model": args.model,
                "model_tag": smoke.MODEL_IDS[args.model],
                "model_origin": MODEL_ORIGIN.get(args.model, "Unknown"),
                "persona": args.persona,
                "response": f"ERROR: {type(exc).__name__}: {exc}",
                "elapsed_s": round(time.time() - started, 1),
                "score": -1,
            }
            status = "error"

        key = (current_model_key, filename)
        results = [r for r in results if done_key(r) != key]
        results.append(entry)
        write_results(output_file, results)
        print(
            f"HF_MATRIX_RESULT {idx}/{total} {current_model_key} {filename} "
            f"{status} {entry['elapsed_s']}s",
            flush=True,
        )

    print(f"HF_MATRIX_DONE model={args.model} persona={args.persona}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
