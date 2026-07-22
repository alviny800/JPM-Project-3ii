#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
add_clean_ticker_cols.py

Add WRDS-resolvable ticker columns to the analysis file.

Bloomberg tickers carry a market suffix (e.g. "AMZN US") that the pipeline's
identifier resolver does NOT strip (download_ownership_etf_data.py:552 uses the
raw string), so "AMZN US" never matches CRSP's `ticker`. This adds
`Target Ticker Clean` / `Acquirer Ticker Clean`:
  - "AMZN US" -> "AMZN"
  - non-US tickers (e.g. "BBVA SM") -> "" (won't match a US table anyway)
  - Bloomberg placeholder ids for delisted names (e.g. "1436513D US") -> ""
    so resolution falls back cleanly to name/CIK matching instead of a
    guaranteed-miss ticker.

Reuses clean_ticker + BBG_PLACEHOLDER_RE from backfill_cusips.py.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from backfill_cusips import clean_ticker, BBG_PLACEHOLDER_RE


def clean_for_wrds(raw: str) -> str:
    c = clean_ticker(raw)              # "" if non-US, else "AMZN US" -> "AMZN"
    if c and BBG_PLACEHOLDER_RE.match(c):
        return ""                      # placeholder delisted id -> fall back to name
    return c


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--input", default="US_election_deals_for_analysis.csv", type=Path)
    args = p.parse_args()

    df = pd.read_csv(args.input, dtype=str, keep_default_na=False)
    for side in ["Target", "Acquirer"]:
        df[f"{side} Ticker Clean"] = df[f"{side} Ticker"].map(clean_for_wrds)

    df.to_csv(args.input, index=False)

    for side in ["Target", "Acquirer"]:
        col = f"{side} Ticker Clean"
        n_ok = (df[col].str.strip() != "").sum()
        print(f"{col}: {n_ok}/{len(df)} resolvable "
              f"({len(df)-n_ok} blank: non-US or placeholder)")
    # sanity: no residual ' US' suffixes
    bad = df[df["Target Ticker Clean"].str.contains(" US", na=False)]
    print(f"residual ' US' suffixes: {len(bad)}")


if __name__ == "__main__":
    main()
