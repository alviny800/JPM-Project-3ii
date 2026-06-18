#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
secapi_io_fulltext_ma_screen.py

Optional enhancement if you have a sec-api.io key:
- Reads the same Bloomberg M&A CSV.
- For each event, searches full EDGAR filing text for election/proration keywords.
- Writes candidate filing metadata with SEC filing URLs.

This does not replace download_ma_edgar_files.py. It helps discover filings whose company
name matching through official SEC CIK mapping is noisy, and it can search exhibits directly.

Example:
    export SEC_API_KEY="..."
    python secapi_io_fulltext_ma_screen.py \
        --input ma_export_33248147_212700.csv \
        --output-dir secapi_screen \
        --api-key "$SEC_API_KEY" \
        --max-events 100
"""

from __future__ import annotations

import argparse
import json
import os
import random
import re
import sys
import time
from pathlib import Path
from typing import Any, Dict, List

import pandas as pd
import requests


DEFAULT_FORMS = [
    "S-4", "S-4/A", "F-4", "F-4/A",
    "424B3", "424B5", "DEFM14A", "PREM14A", "DEFA14A",
    "8-K", "8-K/A", "425",
    "SC TO-T", "SC TO-T/A", "SC 14D9", "SC 14D9/A",
    "SC TO-I", "SC TO-I/A", "SC 13E3", "SC 13E3/A",
]

DEFAULT_KEYWORDS = [
    '"cash election"', '"stock election"', '"mixed election"',
    '"election deadline"', '"election form"', '"letter of transmittal"',
    'proration', '"non-election"', '"non-electing"',
    '"exchange ratio"', '"final proration"', '"election results"',
]


def quote_phrase(s: str) -> str:
    s = str(s or "")
    s = s.replace('"', " ")
    s = re.sub(r"\s+", " ", s).strip()
    return f'"{s}"' if s else '""'


def build_event_query(target: str, acquirer: str, keywords: List[str]) -> str:
    # Keep query deliberately broad. Full text often mentions target/acquirer names many times.
    parties = f"({quote_phrase(target)} OR {quote_phrase(acquirer)})"
    kws = "(" + " OR ".join(keywords) + ")"
    return f"{parties} AND {kws}"


def post_full_text_search(api_key: str, payload: Dict[str, Any], retries: int = 4, sleep_seconds: float = 0.25) -> Dict[str, Any]:
    url = "https://api.sec-api.io/full-text-search"
    headers = {"Authorization": api_key, "Content-Type": "application/json"}
    last_err = None
    for attempt in range(retries):
        time.sleep(sleep_seconds + random.uniform(0, sleep_seconds))
        try:
            r = requests.post(url, json=payload, headers=headers, timeout=45)
            if r.status_code in {429, 503}:
                wait = (2 ** attempt) + random.uniform(0, 1)
                print(f"[sec-api throttle] {r.status_code}; sleeping {wait:.1f}s", file=sys.stderr)
                time.sleep(wait)
                continue
            r.raise_for_status()
            return r.json()
        except Exception as e:
            last_err = e
            time.sleep((2 ** attempt) + random.uniform(0, 1))
    raise RuntimeError(f"sec-api.io request failed: {last_err}")


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--input", required=True)
    p.add_argument("--output-dir", default="secapi_screen")
    p.add_argument("--api-key", default=os.environ.get("SEC_API_KEY", ""))
    p.add_argument("--payment-types", nargs="*", default=["Cash or Stock", "Cash and Stock"])
    p.add_argument("--deal-status", nargs="*", default=["Completed"])
    p.add_argument("--all-payment-types", action="store_true")
    p.add_argument("--all-status", action="store_true")
    p.add_argument("--start-date", default=None)
    p.add_argument("--end-date", default=None)
    p.add_argument("--pre-days", type=int, default=60)
    p.add_argument("--post-days", type=int, default=730)
    p.add_argument("--forms", nargs="*", default=DEFAULT_FORMS)
    p.add_argument("--keywords", nargs="*", default=DEFAULT_KEYWORDS)
    p.add_argument("--page-size", type=int, default=100)
    p.add_argument("--max-pages-per-event", type=int, default=3)
    p.add_argument("--max-events", type=int, default=None)
    p.add_argument("--sleep-seconds", type=float, default=0.25)
    args = p.parse_args()

    if not args.api_key:
        raise SystemExit("Provide --api-key or export SEC_API_KEY=...")

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(args.input)
    df["Announce Date Parsed"] = pd.to_datetime(df["Announce Date"], errors="coerce")
    udf = df.copy()
    if not args.all_payment_types:
        udf = udf[udf["Payment Type"].astype(str).isin(args.payment_types)]
    if not args.all_status:
        udf = udf[udf["Deal Status"].astype(str).isin(args.deal_status)]
    if args.start_date:
        udf = udf[udf["Announce Date Parsed"] >= pd.to_datetime(args.start_date)]
    if args.end_date:
        udf = udf[udf["Announce Date Parsed"] <= pd.to_datetime(args.end_date)]
    udf = udf.dropna(subset=["Announce Date Parsed"]).reset_index(drop=False).rename(columns={"index": "orig_row_idx"})
    if args.max_events:
        udf = udf.head(args.max_events)

    rows: List[Dict[str, Any]] = []
    for i, event in udf.iterrows():
        target = str(event.get("Target Name", ""))
        acquirer = str(event.get("Acquirer Name", ""))
        ann = pd.Timestamp(event["Announce Date Parsed"])
        start = (ann - pd.Timedelta(days=args.pre_days)).strftime("%Y-%m-%d")
        end = (ann + pd.Timedelta(days=args.post_days)).strftime("%Y-%m-%d")
        query = build_event_query(target, acquirer, args.keywords)

        print(f"[{i+1}/{len(udf)}] {target} / {acquirer} | {start} to {end}")
        for page in range(args.max_pages_per_event):
            payload = {
                "query": query,
                "formTypes": args.forms,
                "startDate": start,
                "endDate": end,
                "page": page,
                "size": args.page_size,
            }
            try:
                data = post_full_text_search(args.api_key, payload, sleep_seconds=args.sleep_seconds)
            except Exception as e:
                rows.append({
                    "event_idx": int(event["orig_row_idx"]),
                    "target_name": target,
                    "acquirer_name": acquirer,
                    "announce_date": str(ann.date()),
                    "query": query,
                    "error": str(e),
                })
                break

            filings = data.get("filings", []) or data.get("results", []) or []
            if not filings:
                break

            for f in filings:
                row = {
                    "event_idx": int(event["orig_row_idx"]),
                    "target_name": target,
                    "acquirer_name": acquirer,
                    "announce_date": str(ann.date()),
                    "payment_type": event.get("Payment Type", ""),
                    "deal_status": event.get("Deal Status", ""),
                    "query": query,
                    "page": page,
                }
                # Keep all metadata flexibly because sec-api.io field names can change slightly.
                for k, v in f.items():
                    if isinstance(v, (dict, list)):
                        row[k] = json.dumps(v, ensure_ascii=False)
                    else:
                        row[k] = v
                rows.append(row)

            # Stop if fewer than page size returned.
            if len(filings) < args.page_size:
                break

        if (i + 1) % 25 == 0:
            pd.DataFrame(rows).to_csv(out_dir / "secapi_fulltext_candidates_partial.csv", index=False)

    out = pd.DataFrame(rows)
    out.to_csv(out_dir / "secapi_fulltext_candidates.csv", index=False)
    print(f"Wrote {len(out):,} rows to {out_dir / 'secapi_fulltext_candidates.csv'}")


if __name__ == "__main__":
    main()
