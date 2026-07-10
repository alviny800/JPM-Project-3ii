#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Normalize M&A event CSVs before SEC/WRDS/model pipeline steps.

The original scripts were written for a Bloomberg M&A export.  The current
research universe adds already-cleaned ticker/CUSIP helper columns.  This
adapter keeps the original columns intact while adding stable canonical helper
columns used by downstream identifier matching.
"""

from __future__ import annotations

import re
from typing import Any, Iterable, Optional

import pandas as pd


TARGET_SYMBOL_COL = "normalized_target_symbol"
ACQUIRER_SYMBOL_COL = "normalized_acquirer_symbol"
TARGET_CUSIP_COL = "normalized_target_cusip"
ACQUIRER_CUSIP_COL = "normalized_acquirer_cusip"
TARGET_CUSIP_STATUS_COL = "normalized_target_cusip_status"
ACQUIRER_CUSIP_STATUS_COL = "normalized_acquirer_cusip_status"


TARGET_SYMBOL_CANDIDATES = [
    "Target Ticker Clean",
    "Target Ticker",
    "Target ticker",
    "target_ticker",
    "target_symbol",
]

ACQUIRER_SYMBOL_CANDIDATES = [
    "Acquirer Ticker Clean",
    "Acquirer Ticker",
    "Acquirer ticker",
    "acquirer_ticker",
    "acquirer_symbol",
]

TARGET_CUSIP_CANDIDATES = [
    "Target cusip",
    "Target CUSIP",
    "Target cusip_bbg_raw",
    "target_cusip",
]

ACQUIRER_CUSIP_CANDIDATES = [
    "Acquirer cusip",
    "Acquirer CUSIP",
    "Acquirer cusip_bbg_raw",
    "acquirer_cusip",
]


def clean_text(value: Any) -> str:
    if value is None:
        return ""
    try:
        if pd.isna(value):
            return ""
    except Exception:
        pass
    text = str(value).strip()
    if text.lower() in {"", "nan", "none", "null"}:
        return ""
    return text


def first_nonempty(row: pd.Series, names: Iterable[str]) -> str:
    for name in names:
        if name in row.index:
            text = clean_text(row.get(name))
            if text:
                return text
    return ""


def normalize_symbol(value: Any) -> str:
    """Convert Bloomberg-style tickers like 'VMW US' to 'VMW'."""
    text = clean_text(value).upper()
    if not text:
        return ""
    text = re.sub(r"\s+", " ", text)
    # Bloomberg tickers normally carry a market suffix.  Keep slash/dot class
    # notation (BRK/A) but remove the trailing exchange token.
    parts = text.split(" ")
    if len(parts) >= 2 and len(parts[-1]) <= 3:
        text = " ".join(parts[:-1])
    # Bloomberg dead/security identifiers such as 1373183D are not exchange
    # tickers.  Leave these blank and let WRDS resolve by CUSIP/name instead.
    if re.match(r"^\d{4,}[A-Z]?$", text):
        return ""
    text = text.replace(".", "/") if "/" in text else text
    return text.strip()


def normalize_cusip(value: Any) -> str:
    text = re.sub(r"[^A-Za-z0-9]+", "", clean_text(value)).upper()
    return text[:9]


def canonical_event_id_source_index(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    if "orig_row_idx" not in out.columns:
        out = out.reset_index(drop=False).rename(columns={"index": "orig_row_idx"})
    return out


def normalize_event_input(df: pd.DataFrame) -> pd.DataFrame:
    """Return a copy with canonical helper columns added.

    Added columns:
    - normalized_target_symbol / normalized_acquirer_symbol
    - normalized_target_cusip / normalized_acquirer_cusip
    - normalized_*_cusip_status when the decision/status columns exist
    - Announce Date Parsed when Announce Date exists
    """
    out = df.copy()

    if "Announce Date Parsed" not in out.columns and "Announce Date" in out.columns:
        out["Announce Date Parsed"] = pd.to_datetime(out["Announce Date"], errors="coerce")

    if TARGET_SYMBOL_COL not in out.columns:
        out[TARGET_SYMBOL_COL] = [
            normalize_symbol(first_nonempty(row, TARGET_SYMBOL_CANDIDATES))
            for _, row in out.iterrows()
        ]
    if ACQUIRER_SYMBOL_COL not in out.columns:
        out[ACQUIRER_SYMBOL_COL] = [
            normalize_symbol(first_nonempty(row, ACQUIRER_SYMBOL_CANDIDATES))
            for _, row in out.iterrows()
        ]
    if TARGET_CUSIP_COL not in out.columns:
        out[TARGET_CUSIP_COL] = [
            normalize_cusip(first_nonempty(row, TARGET_CUSIP_CANDIDATES))
            for _, row in out.iterrows()
        ]
    if ACQUIRER_CUSIP_COL not in out.columns:
        out[ACQUIRER_CUSIP_COL] = [
            normalize_cusip(first_nonempty(row, ACQUIRER_CUSIP_CANDIDATES))
            for _, row in out.iterrows()
        ]

    if TARGET_CUSIP_STATUS_COL not in out.columns:
        out[TARGET_CUSIP_STATUS_COL] = out.get("Target cusip_decision", "")
    if ACQUIRER_CUSIP_STATUS_COL not in out.columns:
        out[ACQUIRER_CUSIP_STATUS_COL] = out.get("Acquirer cusip_decision", "")

    return out


def default_input_column(explicit: Optional[str], canonical: str) -> str:
    return explicit or canonical
