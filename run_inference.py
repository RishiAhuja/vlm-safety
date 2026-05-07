"""
Run VLM inference via Ollama on all stimuli × all models.

GPU-optimised: passes num_gpu=99 so Ollama maps all transformer layers onto
available VRAM (T4, A100, etc.).  Serial across model configs (so the active
model stays resident in VRAM), parallel within each model's stimulus batch
using INFERENCE_WORKERS async workers.

Requires Ollama to be started with:
    OLLAMA_NUM_PARALLEL=3 ollama serve
(or set OLLAMA_NUM_PARALLEL in /etc/systemd for the ollama service).

Saves results to results/inference_results.json (append-safe: skips already-done pairs).
"""
import asyncio
import base64
import json
import os
import sys
import time

import ollama

from config import (
    INFERENCE_PROMPT,
    INFERENCE_WORKERS,
    RESULTS_DIR,
    STIMULI,
    STIMULI_DIR,
    VLM_MODELS,
)

RESULTS_FILE = os.path.join(RESULTS_DIR, "inference_results.json")


def _encode_image(path: str) -> str:
    with open(path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")


def _load_existing() -> list[dict]:
    if os.path.exists(RESULTS_FILE):
        with open(RESULTS_FILE) as f:
            return json.load(f)
    return []


def _already_done(results: list[dict], stim: str, model: str) -> bool:
    return any(
        r["stimulus"] == stim
        and r["model"] == model
        and not r.get("response", "").startswith("ERROR:")
        for r in results
    )


def _check_ollama() -> bool:
    try:
        ollama.list()
        return True
    except Exception:
        return False


def _ensure_model(tag: str) -> bool:
    try:
        models = ollama.list()
        pulled = [m.model for m in models.models]
        if any(tag in m or m.startswith(tag.split(":")[0]) for m in pulled):
            return True
    except Exception:
        pass
    print(f"  Pulling {tag} (this may take a while)...")
    try:
        ollama.pull(tag)
        print(f"  ✓ Pulled {tag}")
        return True
    except Exception as e:
        print(f"  ✗ Failed to pull {tag}: {e}")
        return False


_MAX_RETRIES = 3
_RETRY_DELAY_S = 20  # wait between retries — GGML crash needs ~15-20s for runner restart


async def _infer_one(
    client: ollama.AsyncClient,
    sem: asyncio.Semaphore,
    lock: asyncio.Lock,
    results: list[dict],
    counter: list[int],
    total: int,
    model_name: str,
    model_info: dict,
    filename: str,
    category: str,
    axis: str,
    concept: str,
) -> None:
    async with sem:
        counter[0] += 1
        idx = counter[0]
        print(f"  [{idx}/{total}] {filename} ({axis}) ... ", end="", flush=True)
        t0 = time.time()

        img_b64 = _encode_image(os.path.join(STIMULI_DIR, filename))
        messages = []
        system_prompt = model_info.get("system_prompt", "")
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({
            "role": "user",
            "content": INFERENCE_PROMPT,
            "images": [img_b64],
        })

        content = None
        last_exc = None
        for attempt in range(1, _MAX_RETRIES + 1):
            try:
                response = await client.chat(
                    model=model_info["tag"],
                    messages=messages,
                    options={"num_gpu": 99},
                )
                content = response["message"]["content"]
                last_exc = None
                break
            except Exception as e:
                last_exc = e
                if attempt < _MAX_RETRIES:
                    print(f"retry {attempt}/{_MAX_RETRIES} ({e}) ... ", end="", flush=True)
                    await asyncio.sleep(_RETRY_DELAY_S)
                # brief sleep between any request to avoid flooding the runner
            await asyncio.sleep(1)

        elapsed = time.time() - t0

        if content is None:
            print(f"ERROR ({elapsed:.0f}s): {last_exc}")
            entry = {
                "stimulus": filename,
                "category": category,
                "axis": axis,
                "concept": concept,
                "model": model_name,
                "model_tag": model_info["tag"],
                "model_origin": model_info["origin"],
                "response": f"ERROR: {last_exc}",
                "elapsed_s": round(elapsed, 1),
                "score": -1,
            }
            async with lock:
                results[:] = [
                    r
                    for r in results
                    if not (
                        r["stimulus"] == filename
                        and r["model"] == model_name
                        and r.get("response", "").startswith("ERROR:")
                    )
                ]
                results.append(entry)
                _write_results(results)
            return

        entry = {
            "stimulus": filename,
            "category": category,
            "axis": axis,
            "concept": concept,
            "model": model_name,
            "model_tag": model_info["tag"],
            "model_origin": model_info["origin"],
            "response": content,
            "elapsed_s": round(elapsed, 1),
            "score": None,
        }

        # Atomic append + save (crash-safe)
        async with lock:
            results[:] = [
                r
                for r in results
                if not (
                    r["stimulus"] == filename
                    and r["model"] == model_name
                    and r.get("response", "").startswith("ERROR:")
                )
            ]
            results.append(entry)
            _write_results(results)

        preview = content[:120].replace("\n", " ")
        print(f"OK ({elapsed:.0f}s) → {preview}...")


def _write_results(results: list[dict]) -> None:
    os.makedirs(RESULTS_DIR, exist_ok=True)
    with open(RESULTS_FILE, "w") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)


async def _run_async(models: dict) -> list[dict]:
    client = ollama.AsyncClient()
    results = _load_existing()
    lock = asyncio.Lock()
    sem = asyncio.Semaphore(INFERENCE_WORKERS)

    for model_name, model_info in models.items():
        tag = model_info["tag"]
        print(f"\n{'='*60}")
        print(f"Model: {model_name} ({tag}) | workers={INFERENCE_WORKERS} | num_gpu=99")
        print(f"{'='*60}")

        if not _ensure_model(tag):
            print(f"  Skipping {model_name} (model not available)")
            continue

        pending = [
            (fn, cat, ax, con)
            for fn, cat, ax, con in STIMULI
            if not _already_done(results, fn, model_name)
              and os.path.exists(os.path.join(STIMULI_DIR, fn))
        ]
        if not pending:
            print("  All stimuli already done, skipping.")
            continue

        counter = [0]
        total = len(pending)
        await asyncio.gather(*[
            _infer_one(client, sem, lock, results, counter, total,
                       model_name, model_info, fn, cat, ax, con)
            for fn, cat, ax, con in pending
        ])

    print(f"\n✓ Inference complete. {len(results)} results saved to {RESULTS_FILE}")
    return results


def run_inference(models: dict | None = None) -> list[dict]:
    if not _check_ollama():
        print("ERROR: Ollama is not running.")
        print("  Start with: OLLAMA_NUM_PARALLEL=3 ollama serve")
        sys.exit(1)
    if models is None:
        models = VLM_MODELS
    return asyncio.run(_run_async(models))


if __name__ == "__main__":
    run_inference()
