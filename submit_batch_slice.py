#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
submit_batch_slice.py

Submit a SLICE of cached LLM payloads as one Anthropic Message Batch, using a chosen
API-key env var. Lets us split the full run across two separately-funded keys
(Batch #1 on the borrowed key, Batch #2 on the user's key) without re-scanning —
it reads the payloads already built by `--llm-stage batch`/`prepare`.

Each invocation writes a results JSONL; merge_batch_results.py combines them into the
final llm_field_extractions.csv.
"""
from __future__ import annotations
import argparse, json, os
from pathlib import Path
import download_ma_edgar_files as dl


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--payloads", default="ma_edgar_full/llm_field_payloads.jsonl", type=Path)
    p.add_argument("--start", type=int, default=0, help="slice start index (inclusive)")
    p.add_argument("--end", type=int, default=None, help="slice end index (exclusive)")
    p.add_argument("--key-env", default="ANTHROPIC_API_KEY", help="env var holding the API key for this batch")
    p.add_argument("--out", required=True, type=Path, help="results JSONL to write")
    p.add_argument("--model", default="claude-sonnet-5")
    p.add_argument("--max-tokens", type=int, default=12000)
    p.add_argument("--max-cost-usd", type=float, default=None, help="hard pre-flight cap; abort before submit if over")
    p.add_argument("--poll-seconds", type=int, default=60)
    args = p.parse_args()

    payloads = [json.loads(l) for l in open(args.payloads) if l.strip()]
    sl = payloads[args.start:args.end]
    key = os.environ.get(args.key_env, "")
    if not key:
        raise SystemExit(f"[submit] No API key in ${args.key_env}")
    print(f"[submit] slice [{args.start}:{args.end}] = {len(sl)} payloads via ${args.key_env} "
          f"(key {len(key)} chars) -> {args.out}")

    records = dl.run_anthropic_batch(
        sl, api_key=key, model=args.model, max_tokens=args.max_tokens,
        max_cost_usd=args.max_cost_usd, poll_seconds=args.poll_seconds,
    )
    dl.write_jsonl(args.out, records)
    n_ok = sum(1 for r in records if not (isinstance(r.get("parsed"), dict) and r["parsed"].get("parse_error")))
    print(f"[submit] wrote {len(records)} records ({n_ok} parsed OK, {len(records)-n_ok} errored) -> {args.out}")


if __name__ == "__main__":
    main()
