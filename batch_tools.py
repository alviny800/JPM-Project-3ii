#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""batch_tools.py — Anthropic Message Batch helpers for the Stage-2 extraction.

Subcommands:
  submit  submit a slice of cached LLM payloads as one batch (choose the API-key env var)
  merge   combine per-key batch result JSONLs -> llm_field_results.jsonl + llm_field_extractions.csv

Merges the former submit_batch_slice.py + merge_batch_results.py; logic is verbatim.
"""
from __future__ import annotations
import sys

import argparse, json, os
from pathlib import Path
import download_ma_edgar_files as dl


def main_submit() -> None:
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

import argparse, json
from pathlib import Path
import download_ma_edgar_files as dl


def main_merge() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--results", nargs="+", required=True, help="batch result JSONL files to merge")
    p.add_argument("--out-dir", default="ma_edgar_full", type=Path)
    args = p.parse_args()

    records = []
    for path in args.results:
        n = 0
        for line in open(path):
            if line.strip():
                records.append(json.loads(line)); n += 1
        print(f"[merge] {path}: {n} records")

    (args.out_dir / "llm_field_results.jsonl").write_text(
        "\n".join(json.dumps(r, ensure_ascii=False) for r in records) + "\n", encoding="utf-8")
    df = dl.flatten_llm_records(records)
    out_csv = args.out_dir / "llm_field_extractions.csv"
    df.to_csv(out_csv, index=False)

    n_parse_err = sum(1 for r in records if isinstance(r.get("parsed"), dict) and r["parsed"].get("parse_error"))
    events = df["event_id"].nunique() if "event_id" in df.columns else 0
    print(f"\n[merge] {len(records)} deals total | {len(records)-n_parse_err} parsed OK | {n_parse_err} parse-errors")
    print(f"[merge] wrote {out_csv}  ({len(df):,} field rows across {events} events)")
    # quick label-capture peek
    if "field_name" in df.columns:
        LAB = ["realized_cash_election_demand", "realized_stock_election_demand",
               "final_proration_results", "preliminary_proration_results"]
        sub = df[df["field_name"].isin(LAB)].copy()
        def has_val(v):
            s = "" if v is None else str(v)
            return s.strip().lower() not in ("", "nan", "none", "null", "not disclosed", "not found", "n/a")
        cap = sub[sub["value"].map(has_val)]["event_id"].nunique()
        print(f"[merge] events with >=1 realized-demand/proration value: {cap}/{events} ({cap/max(events,1)*100:.0f}%)")


def main():
    if len(sys.argv) < 2 or sys.argv[1] not in ("submit", "merge"):
        print("usage: batch_tools.py {submit|merge} [options]"); sys.exit(2)
    sub = sys.argv.pop(1)
    (main_submit if sub == "submit" else main_merge)()


if __name__ == "__main__":
    main()
