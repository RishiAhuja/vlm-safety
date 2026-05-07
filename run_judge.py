"""
LLM-as-Judge scoring of VLM responses using an OpenAI judge model.
Reads inference results, scores each response, and writes a resumable scored file.
Uses async concurrency for fast parallel judging.
"""
import asyncio
import json
import os
import sys
import time

from dotenv import load_dotenv
from openai import AsyncOpenAI

from config import JUDGE_CONCURRENCY as DEFAULT_JUDGE_CONCURRENCY
from config import JUDGE_MODEL as DEFAULT_JUDGE_MODEL
from config import JUDGE_PROMPT_TEMPLATE, RESULTS_DIR

load_dotenv()

INPUT_FILE = os.environ.get(
    "JUDGE_INPUT_FILE",
    os.path.join(RESULTS_DIR, "inference_results.json"),
)
OUTPUT_FILE = os.environ.get(
    "JUDGE_OUTPUT_FILE",
    os.path.join(RESULTS_DIR, "inference_results_scored.json"),
)
JUDGE_MODEL = os.environ.get("JUDGE_MODEL", DEFAULT_JUDGE_MODEL)
JUDGE_CONCURRENCY = int(os.environ.get("JUDGE_CONCURRENCY", str(DEFAULT_JUDGE_CONCURRENCY)))
JUDGE_LIMIT = int(os.environ.get("JUDGE_LIMIT", "0"))
JUDGE_MAX_RESPONSE_CHARS = int(os.environ.get("JUDGE_MAX_RESPONSE_CHARS", "1500"))


def _parse_judgment(raw: str) -> dict:
    """Extract score/reason JSON from the model response."""
    text = raw.strip()

    # Strip markdown code fences if present
    if "```" in text:
        parts = text.split("```")
        for part in parts[1:]:
            cleaned = part.strip()
            if cleaned.startswith("json"):
                cleaned = cleaned[4:].strip()
            if cleaned.startswith("{"):
                text = cleaned
                break

    try:
        parsed = json.loads(text)
        return {
            "score": int(parsed["score"]),
            "failure_mode": parsed.get("failure_mode", "unknown"),
            "reason": parsed.get("reason", ""),
        }
    except (json.JSONDecodeError, KeyError, ValueError):
        # Fallback: try to find score in raw text
        for s in [0, 1, 2]:
            if f'"score": {s}' in raw or f'"score":{s}' in raw:
                return {"score": s, "failure_mode": "unknown", "reason": raw[:200]}
        return {"score": -1, "failure_mode": "parse_error", "reason": f"Parse error: {raw[:200]}"}


async def score_one(
    client: AsyncOpenAI,
    idx: int,
    total: int,
    record: dict,
    save_lock: asyncio.Lock,
    all_results: list[dict],
) -> None:
    """Score a single VLM response via the OpenAI API."""
    label = f"{record['stimulus']} × {record['model']}"

    # Skip error responses
    if record.get("response", "").startswith("ERROR:"):
        record["score"] = -1
        record["failure_mode"] = "inference_error"
        record["judge_reason"] = "Inference error"
        record["judge_model"] = JUDGE_MODEL
        print(f"  [{idx}/{total}] {label} — skipped (inference error)")
        return

    prompt = JUDGE_PROMPT_TEMPLATE.format(
        concept=record["concept"],
        response=record["response"][:JUDGE_MAX_RESPONSE_CHARS],
    )

    t0 = time.time()
    try:
        resp = await client.chat.completions.create(
            model=JUDGE_MODEL,
            messages=[{"role": "user", "content": prompt}],
        )
        raw = resp.choices[0].message.content or ""
    except Exception as e:
        record["score"] = -1
        record["failure_mode"] = "api_error"
        record["judge_reason"] = f"API error: {e}"
        record["judge_model"] = JUDGE_MODEL
        print(f"  [{idx}/{total}] {label} — API error: {e}")
        return

    judgment = _parse_judgment(raw)
    elapsed = time.time() - t0

    record["score"] = judgment["score"]
    record["failure_mode"] = judgment.get("failure_mode", "unknown")
    record["judge_reason"] = judgment["reason"]
    record["judge_model"] = JUDGE_MODEL

    print(f"  [{idx}/{total}] {label} — score={judgment['score']} ({elapsed:.1f}s) — {judgment['reason'][:80]}")

    # Atomic crash-safe save under lock (temp file + rename to prevent corruption)
    async with save_lock:
        tmp_file = OUTPUT_FILE + ".tmp"
        with open(tmp_file, "w") as f:
            json.dump(all_results, f, indent=2, ensure_ascii=False)
        os.rename(tmp_file, OUTPUT_FILE)


async def _run_judge_async() -> list[dict]:
    if not os.path.exists(INPUT_FILE):
        print(f"ERROR: {INPUT_FILE} not found. Run inference first.")
        sys.exit(1)

    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        print("ERROR: OPENAI_API_KEY not set. Add it to .env or export it.")
        sys.exit(1)

    with open(INPUT_FILE) as f:
        results = json.load(f)

    # Load existing scored results to skip already-scored entries
    scored_lookup: dict[tuple, dict] = {}
    if os.path.exists(OUTPUT_FILE):
        try:
            with open(OUTPUT_FILE) as f:
                existing = json.load(f)
                for r in existing:
                    key = (r["stimulus"], r["model"])
                    if (
                        r.get("score") is not None
                        and r["score"] != -1
                        and r.get("judge_model", JUDGE_MODEL) == JUDGE_MODEL
                    ):
                        scored_lookup[key] = r
            print(f"Loaded {len(scored_lookup)} already-scored entries from {OUTPUT_FILE}")
        except (json.JSONDecodeError, IOError) as e:
            print(f"Warning: could not load {OUTPUT_FILE}: {e}. Starting fresh.")
            scored_lookup = {}

    # Separate already-scored from pending
    pending = []
    total = len(results)
    for i, r in enumerate(results, 1):
        key = (r["stimulus"], r["model"])
        if key in scored_lookup:
            r["score"] = scored_lookup[key]["score"]
            r["failure_mode"] = scored_lookup[key].get("failure_mode", "unknown")
            r["judge_reason"] = scored_lookup[key].get("judge_reason", "")
            r["judge_model"] = scored_lookup[key].get("judge_model", JUDGE_MODEL)
            print(f"  [{i}/{total}] {r['stimulus']} × {r['model']} — already scored ({r['score']})")
        else:
            pending.append((i, r))

    if JUDGE_LIMIT > 0:
        pending = pending[:JUDGE_LIMIT]

    if not pending:
        print("\nAll responses already scored.")
        return results

    print(f"\nScoring {len(pending)} responses with {JUDGE_MODEL} ({JUDGE_CONCURRENCY} async workers)...\n")

    client = AsyncOpenAI(api_key=api_key)
    save_lock = asyncio.Lock()
    sem = asyncio.Semaphore(JUDGE_CONCURRENCY)

    async def bounded(idx: int, record: dict):
        async with sem:
            await score_one(client, idx, total, record, save_lock, results)

    tasks = [bounded(idx, record) for idx, record in pending]
    await asyncio.gather(*tasks)

    # Final atomic save
    tmp_file = OUTPUT_FILE + ".tmp"
    with open(tmp_file, "w") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    os.rename(tmp_file, OUTPUT_FILE)

    print(f"\n✓ Scoring complete. Results saved to {OUTPUT_FILE}")
    return results


def run_judge() -> list[dict]:
    return asyncio.run(_run_judge_async())


if __name__ == "__main__":
    run_judge()
