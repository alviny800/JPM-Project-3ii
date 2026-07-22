#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
merge_batch_results.py

Combine the per-key batch result JSONLs (Batch #1 + Batch #2) into the final
llm_field_results.jsonl + llm_field_extractions.csv, using the same flattener the
normal pipeline uses. Also reports capture stats.
"""
from __future__ import annotations
import argparse, json
from pathlib import Path
import download_ma_edgar_files as dl


def main() -> None:
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


if __name__ == "__main__":
    main()
