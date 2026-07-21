#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
build_close_dates.py

Produce an authoritative deal-close date per target from CRSP.

The EDGAR field locator anchors realized-results evidence on the deal close date (the
election-results 8-K is filed at close). Inferring close from filings is unreliable —
targets that deregister at close give the right date, but debt-issuing subsidiaries
(e.g. BNSF) keep filing for years and blow the estimate out. The target's CRSP
delisting date (`stocknames.nameenddt` for the security's last name-row) IS the close
date, and it's already in stocknames_cache.csv from the CUSIP backfill — no new WRDS
query needed.

Output: target_close_dates.csv (target_cusip, target_cusip8, close_date, source).
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path

import pandas as pd


def cusip8(v: str) -> str:
    return re.sub(r"[^0-9A-Za-z]", "", str(v)).upper()[:8]


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--input", default="US_election_deals_for_analysis.csv", type=Path)
    p.add_argument("--stocknames", default="stocknames_cache.csv", type=Path)
    p.add_argument("--out", default="target_close_dates.csv", type=Path)
    args = p.parse_args()

    df = pd.read_csv(args.input, dtype=str, keep_default_na=False)
    sn = pd.read_csv(args.stocknames, dtype=str, keep_default_na=False)
    sn["nc8"] = sn["ncusip"].map(cusip8)
    sn["end"] = pd.to_datetime(sn["nameenddt"], errors="coerce")
    # last name-row end date per security = delisting/close date
    close_by_c8 = sn.groupby("nc8")["end"].max()

    rows = []
    for _, r in df.iterrows():
        raw = r.get("Target cusip", "")
        c8 = cusip8(raw)
        dl = close_by_c8.get(c8) if c8 else None
        rows.append({
            "target_cusip": raw,
            "target_cusip8": c8,
            "close_date": "" if pd.isna(dl) or dl is None else pd.Timestamp(dl).strftime("%Y-%m-%d"),
            "source": "crsp_stocknames_nameenddt" if (dl is not None and not pd.isna(dl)) else "",
        })
    out = pd.DataFrame(rows)
    out.to_csv(args.out, index=False)
    n_ok = (out["close_date"] != "").sum()
    print(f"[write] {args.out}: {len(out)} targets, {n_ok} with a CRSP close date "
          f"({n_ok/len(out)*100:.0f}%)")
    print(out[out["close_date"] != ""].head(6).to_string(index=False))


if __name__ == "__main__":
    main()
