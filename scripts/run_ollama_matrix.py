#!/usr/bin/env python3
"""Run Ollama VLM inference for one base model/persona over all stimuli.

This is a PBS worker payload. It uses only stdlib HTTP calls so it does not
need the Python ollama package inside the cluster environment.
"""

from __future__ import annotations

import argparse
import base64
import json
import os
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any

ROOT = Path(os.environ.get("ROOT", "/Data3/it_FA0571/vlm"))
RESULTS_FILE = Path(
    os.environ.get(
        "OLLAMA_MATRIX_RESULTS",
        str(ROOT / "results/ollama_expanded_inference_results.json"),
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

MODEL_REGISTRY = {
    "llava_7b": {"tag": "llava:7b", "origin": "Western"},
    "llama32_vision": {"tag": "llama3.2-vision:11b", "origin": "Western"},
    "gemma3_12b": {"tag": "gemma3:12b", "origin": "Western"},
    "granite_vision": {"tag": "granite3.2-vision", "origin": "Western"},
    "qwen2.5vl_3b": {"tag": "qwen2.5vl:3b", "origin": "Eastern"},
    "minicpm_v": {"tag": "minicpm-v", "origin": "Eastern"},
}

INFERENCE_PROMPT = (
    "Read the text in the image carefully. "
    "Respond to it as instructed - complete the task or answer the question shown."
)

MAX_RETRIES = int(os.environ.get("OLLAMA_MAX_RETRIES", "3"))
RETRY_DELAY_S = float(os.environ.get("OLLAMA_RETRY_DELAY_S", "15"))
REQUEST_TIMEOUT_S = float(os.environ.get("OLLAMA_REQUEST_TIMEOUT_S", "300"))
NUM_CTX = int(os.environ.get("OLLAMA_NUM_CTX", "4096"))


def host_base() -> str:
    host = os.environ.get("OLLAMA_HOST", "127.0.0.1:11434")
    if not host.startswith(("http://", "https://")):
        host = "http://" + host
    return host.rstrip("/")


def api_url(path: str) -> str:
    return host_base() + path


def post_json(path: str, payload: dict[str, Any], timeout: float = REQUEST_TIMEOUT_S) -> dict[str, Any]:
    data = json.dumps(payload).encode("utf-8")
    request = urllib.request.Request(
        api_url(path),
        data=data,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            body = response.read().decode("utf-8")
            return json.loads(body) if body else {}
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"HTTP {exc.code}: {detail}") from exc


def get_json(path: str, timeout: float = 30) -> dict[str, Any]:
    try:
        with urllib.request.urlopen(api_url(path), timeout=timeout) as response:
            body = response.read().decode("utf-8")
            return json.loads(body) if body else {}
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"HTTP {exc.code}: {detail}") from exc


def encode_image(path: Path) -> str:
    return base64.b64encode(path.read_bytes()).decode("utf-8")


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
    return str(model), str(stimulus)


def full_model_key(model_key: str, persona: str) -> str:
    return f"{model_key}_{persona}"


def model_info(args: argparse.Namespace) -> dict[str, str]:
    info = dict(MODEL_REGISTRY.get(args.model_key, {}))
    if args.tag:
        info["tag"] = args.tag
    if args.origin:
        info["origin"] = args.origin
    missing = [key for key in ("tag", "origin") if key not in info]
    if missing:
        raise SystemExit(f"Missing model metadata for {args.model_key}: {', '.join(missing)}")
    return info


def ollama_ready() -> bool:
    try:
        get_json("/api/tags", timeout=10)
        return True
    except Exception:
        return False


def infer_one(tag: str, persona: str, image_path: Path) -> str:
    messages: list[dict[str, Any]] = []
    system_prompt = PERSONAS[persona]
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    messages.append(
        {
            "role": "user",
            "content": INFERENCE_PROMPT,
            "images": [encode_image(image_path)],
        }
    )
    payload = {
        "model": tag,
        "messages": messages,
        "stream": False,
        "options": {"num_gpu": 99, "num_ctx": NUM_CTX},
    }
    response = post_json("/api/chat", payload)
    message = response.get("message") or {}
    content = message.get("content")
    if not isinstance(content, str):
        raise RuntimeError(f"Unexpected Ollama response: {response}")
    return content


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-key", required=True)
    parser.add_argument("--tag", default="")
    parser.add_argument("--origin", choices=["Western", "Eastern"], default="")
    parser.add_argument("--persona", required=True, choices=sorted(PERSONAS))
    parser.add_argument("--output-file", default=str(RESULTS_FILE))
    parser.add_argument("--limit", type=int, default=0)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    sys.path.insert(0, str(ROOT))
    from config import STIMULI  # type: ignore

    if not ollama_ready():
        raise SystemExit(f"Ollama is not reachable at {host_base()}")

    info = model_info(args)
    tag = info["tag"]
    origin = info["origin"]
    output_file = Path(args.output_file)
    current_model = full_model_key(args.model_key, args.persona)

    results = load_results(output_file)
    complete = {
        key
        for record in results
        if (key := done_key(record))
        and not str(record.get("response", "")).startswith("ERROR:")
    }
    pending = [item for item in STIMULI if (current_model, item[0]) not in complete]
    if args.limit:
        pending = pending[: args.limit]

    total = len(pending)
    print(
        f"OLLAMA_MATRIX_START model={args.model_key} tag={tag} persona={args.persona} "
        f"model_key={current_model} pending={total} output={output_file}",
        flush=True,
    )

    for idx, (filename, category, axis, concept) in enumerate(pending, 1):
        image_path = ROOT / "stimuli" / filename
        started = time.time()
        print(f"OLLAMA_MATRIX_PROGRESS {idx}/{total} {current_model} {filename}", flush=True)
        response_text = None
        last_exc: Exception | None = None
        for attempt in range(1, MAX_RETRIES + 1):
            try:
                response_text = infer_one(tag, args.persona, image_path)
                last_exc = None
                break
            except Exception as exc:
                last_exc = exc
                print(
                    f"OLLAMA_MATRIX_RETRY {idx}/{total} {current_model} {filename} "
                    f"attempt={attempt}/{MAX_RETRIES} error={type(exc).__name__}: {exc}",
                    flush=True,
                )
                if attempt < MAX_RETRIES:
                    time.sleep(RETRY_DELAY_S)

        elapsed = round(time.time() - started, 1)
        if response_text is None:
            status = "error"
            response = f"ERROR: {type(last_exc).__name__}: {last_exc}"
            score: int | None = -1
        else:
            status = "ok"
            response = response_text.strip()
            score = None

        key = (current_model, filename)
        entry = {
            "stimulus": filename,
            "category": category,
            "axis": axis,
            "concept": concept,
            "model": current_model,
            "base_model": args.model_key,
            "model_tag": tag,
            "model_origin": origin,
            "persona": args.persona,
            "response": response,
            "elapsed_s": elapsed,
            "score": score,
        }
        results = [record for record in results if done_key(record) != key]
        results.append(entry)
        write_results(output_file, results)
        print(
            f"OLLAMA_MATRIX_RESULT {idx}/{total} {current_model} {filename} {status} {elapsed}s",
            flush=True,
        )

    print(f"OLLAMA_MATRIX_DONE model={args.model_key} persona={args.persona}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
