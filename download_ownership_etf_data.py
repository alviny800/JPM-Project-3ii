#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
download_ownership_etf_data.py

Build ownership-mix and ETF-holdings inputs for the M&A election/proration model.

This script is deliberately separate from download_ma_edgar_files.py:
- SEC/Claude extracts deal terms and realized proration labels.
- This script adds target-holder composition data for election-demand forecasts.
- Price, borrow cost, and execution data can be merged later from Bloomberg.

Default provider: Financial Modeling Prep (FMP). It supports a dry-run mode that
creates symbol maps and request plans without an API key.
"""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import html
import json
import os
import random
import re
import sys
import time
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from urllib.parse import urlencode

import pandas as pd
import requests


FMP_BASE = "https://financialmodelingprep.com"

# FMP has changed endpoint namespaces over time. Keep candidates explicit so
# endpoint updates are a small constants edit instead of a pipeline rewrite.
FMP_ENDPOINT_CANDIDATES = {
    "institutional_summary": [
        "/stable/institutional-ownership/symbol-positions-summary",
        "/api/v4/institutional-ownership/symbol-positions-summary",
    ],
    "institutional_holders": [
        "/stable/institutional-ownership/institutional-holders/symbol",
        "/api/v4/institutional-ownership/institutional-holders/symbol",
    ],
    "etf_holders_of_stock": [
        "/stable/etf-holder",
        "/api/v3/etf-holder/{symbol}",
    ],
}

PASSIVE_NAME_PATTERNS = [
    r"\bblackrock\b", r"\bishares\b", r"\bvanguard\b", r"\bstate street\b",
    r"\bssga\b", r"\bgeode\b", r"\bnorthern trust\b", r"\bdimensional\b",
    r"\bfranklin templeton\b", r"\binvesco\b", r"\bspdr\b",
]

HEDGE_EVENT_PATTERNS = [
    r"\belliott\b", r"\bthird point\b", r"\bpaulson\b", r"\bmillennium\b",
    r"\bcitadel\b", r"\bde shaw\b", r"\btwo sigma\b", r"\bpoint72\b",
    r"\bcnc\b", r"\bglazer\b", r"\bmagnetar\b", r"\bsoros\b",
]

ETF_NAME_PATTERNS = [
    r"\betf\b", r"\bexchange traded\b", r"\bindex fund\b", r"\bindex trust\b",
    r"\bspdr\b", r"\bishares\b", r"\bvanguard\b",
]


def now_utc() -> str:
    return dt.datetime.utcnow().replace(microsecond=0).isoformat() + "Z"


def normalize_name(name: str) -> str:
    if not isinstance(name, str):
        return ""
    name = html.unescape(name).lower()
    name = re.sub(r"[^a-z0-9 ]+", " ", name)
    return re.sub(r"\s+", " ", name).strip()


def safe_filename(s: Any, max_len: int = 90) -> str:
    s = str(s) if s is not None else ""
    s = html.unescape(s)
    s = re.sub(r"[^A-Za-z0-9._ -]+", "_", s)
    s = re.sub(r"\s+", "_", s).strip("._- ")
    return s[:max_len] or "NA"


def parse_date(x: Any) -> Optional[pd.Timestamp]:
    if pd.isna(x):
        return None
    return pd.to_datetime(x, errors="coerce")


def event_id_from_row(event_idx: int, event: pd.Series) -> str:
    ann = parse_date(event.get("Announce Date"))
    ann_part = "unknown_date" if ann is None or pd.isna(ann) else pd.Timestamp(ann).strftime("%Y%m%d")
    target = str(event.get("Target Name", "")).strip()
    acquirer = str(event.get("Acquirer Name", "")).strip()
    return f"E{event_idx:06d}_{ann_part}_{safe_filename(target, 45)}__{safe_filename(acquirer, 45)}"


def quarter_before_or_on(date_value: Any, lag_quarters: int = 1) -> Tuple[int, int]:
    """Return a 13F/N-PORT quarter available before an event date."""
    ts = parse_date(date_value)
    if ts is None or pd.isna(ts):
        today = pd.Timestamp.today()
        year, quarter = int(today.year), int((today.month - 1) // 3 + 1)
    else:
        year, quarter = int(ts.year), int((ts.month - 1) // 3 + 1)
    quarter -= lag_quarters
    while quarter <= 0:
        quarter += 4
        year -= 1
    return year, quarter


def classify_holder(name: str, explicit_type: str = "") -> str:
    norm = normalize_name(f"{name} {explicit_type}")
    if any(re.search(p, norm) for p in ETF_NAME_PATTERNS):
        return "etf_or_index_fund"
    if any(re.search(p, norm) for p in PASSIVE_NAME_PATTERNS):
        return "passive_manager"
    if any(re.search(p, norm) for p in HEDGE_EVENT_PATTERNS):
        return "hedge_or_event_driven"
    return "active_or_other_institution"


def first_present(row: Dict[str, Any], names: List[str], default: Any = None) -> Any:
    lowered = {str(k).lower(): v for k, v in row.items()}
    for name in names:
        if name in row and row[name] not in ("", None):
            return row[name]
        lname = name.lower()
        if lname in lowered and lowered[lname] not in ("", None):
            return lowered[lname]
    return default


class FmpClient:
    def __init__(self, api_key: str, cache_dir: Optional[Path], sleep_seconds: float = 0.25, timeout: int = 30) -> None:
        self.api_key = api_key
        self.cache_dir = cache_dir
        self.sleep_seconds = sleep_seconds
        self.timeout = timeout
        self.session = requests.Session()
        if cache_dir:
            cache_dir.mkdir(parents=True, exist_ok=True)

    def _cache_key(self, url: str) -> Path:
        assert self.cache_dir is not None
        h = hashlib.sha1(url.encode("utf-8")).hexdigest()
        return self.cache_dir / f"{h}.json"

    def _url(self, endpoint: str, params: Dict[str, Any]) -> str:
        endpoint = endpoint.format(symbol=params.get("symbol", ""))
        query = {k: v for k, v in params.items() if v not in (None, "") and k != "path_symbol"}
        query["apikey"] = self.api_key
        return FMP_BASE + endpoint + "?" + urlencode(query)

    def get_json(self, endpoint: str, params: Dict[str, Any]) -> Any:
        url = self._url(endpoint, params)
        cache_path = self._cache_key(url) if self.cache_dir else None
        if cache_path and cache_path.exists():
            return json.loads(cache_path.read_text(encoding="utf-8"))
        time.sleep(self.sleep_seconds + random.uniform(0, self.sleep_seconds / 2))
        r = self.session.get(url, timeout=self.timeout)
        if r.status_code == 404:
            raise RuntimeError(f"404 for {endpoint}")
        r.raise_for_status()
        data = r.json()
        if cache_path:
            cache_path.write_text(json.dumps(data), encoding="utf-8")
        return data

    def get_first_available(self, kind: str, params: Dict[str, Any]) -> Tuple[List[Dict[str, Any]], str, str]:
        errors = []
        for endpoint in FMP_ENDPOINT_CANDIDATES[kind]:
            try:
                data = self.get_json(endpoint, params)
                if isinstance(data, dict):
                    if "data" in data and isinstance(data["data"], list):
                        rows = data["data"]
                    else:
                        rows = [data]
                elif isinstance(data, list):
                    rows = data
                else:
                    rows = []
                return rows, endpoint, ""
            except Exception as e:
                errors.append(f"{endpoint}: {e}")
        return [], FMP_ENDPOINT_CANDIDATES[kind][0], " | ".join(errors)


@dataclass
class EventSymbol:
    event_id: str
    event_idx: int
    side: str
    company_name: str
    announce_date: str
    year: int
    quarter: int
    symbol: str
    symbol_source: str
    symbol_note: str


def load_events(args: argparse.Namespace) -> pd.DataFrame:
    df = pd.read_csv(args.input)
    if "orig_row_idx" not in df.columns:
        df = df.reset_index(drop=False).rename(columns={"index": "orig_row_idx"})
    if "Announce Date Parsed" not in df.columns and "Announce Date" in df.columns:
        df["Announce Date Parsed"] = pd.to_datetime(df["Announce Date"], errors="coerce")
    if "Payment Type" in df.columns and not args.all_payment_types:
        df = df[df["Payment Type"].astype(str).isin(args.payment_types)]
    if "Deal Status" in df.columns and not args.all_status:
        df = df[df["Deal Status"].astype(str).isin(args.deal_status)]
    if args.start_date:
        df = df[df["Announce Date Parsed"] >= pd.to_datetime(args.start_date)]
    if args.end_date:
        df = df[df["Announce Date Parsed"] <= pd.to_datetime(args.end_date)]
    df = df.dropna(subset=["Announce Date Parsed"]).reset_index(drop=True)
    if args.max_events:
        df = df.head(args.max_events)
    return df


def load_symbol_overrides(path: Optional[str]) -> Dict[Tuple[int, str], Dict[str, str]]:
    if not path:
        return {}
    df = pd.read_csv(path)
    out: Dict[Tuple[int, str], Dict[str, str]] = {}
    for _, row in df.iterrows():
        event_idx = int(row["event_idx"])
        side = str(row.get("side", "target")).strip().lower()
        out[(event_idx, side)] = {str(k): "" if pd.isna(v) else str(v) for k, v in row.items()}
    return out


def load_cik_tickers(path: Optional[str], min_score: float) -> Dict[Tuple[int, str], Dict[str, str]]:
    if not path or not Path(path).exists():
        return {}
    df = pd.read_csv(path)
    out: Dict[Tuple[int, str], Dict[str, str]] = {}
    for _, row in df.iterrows():
        if not bool(row.get("matched", False)):
            continue
        score = float(row.get("score", 0) or 0)
        if score < min_score:
            continue
        event_idx = int(row["event_idx"])
        side = str(row.get("side", "")).strip().lower()
        ticker = str(row.get("ticker", "")).strip()
        if ticker and ticker.lower() != "nan":
            out[(event_idx, side)] = {
                "symbol": ticker,
                "source": "cik_name_matches",
                "note": f"score={score}; sec_title={row.get('sec_title', '')}",
            }
    return out


def build_event_symbols(
    events: pd.DataFrame,
    sides: List[str],
    cik_tickers: Dict[Tuple[int, str], Dict[str, str]],
    overrides: Dict[Tuple[int, str], Dict[str, str]],
    lag_quarters: int,
    target_symbol_col: Optional[str],
    acquirer_symbol_col: Optional[str],
) -> List[EventSymbol]:
    rows: List[EventSymbol] = []
    side_to_col = {"target": "Target Name", "acquirer": "Acquirer Name"}
    side_to_symbol_col = {"target": target_symbol_col, "acquirer": acquirer_symbol_col}
    for _, event in events.iterrows():
        event_idx = int(event["orig_row_idx"])
        event_id = event_id_from_row(event_idx, event)
        year, quarter = quarter_before_or_on(event.get("Announce Date"), lag_quarters=lag_quarters)
        for side in sides:
            company = str(event.get(side_to_col[side], "")).strip()
            key = (event_idx, side)
            symbol = ""
            source = ""
            note = ""
            if key in overrides:
                symbol = overrides[key].get("symbol", "").strip()
                source = "override"
                note = overrides[key].get("note", "")
            elif side_to_symbol_col.get(side) and side_to_symbol_col[side] in event.index:
                raw_symbol = event.get(side_to_symbol_col[side])
                if not pd.isna(raw_symbol) and str(raw_symbol).strip():
                    symbol = str(raw_symbol).strip()
                    source = f"input_column:{side_to_symbol_col[side]}"
                    note = "symbol supplied by input CSV"
            elif key in cik_tickers:
                symbol = cik_tickers[key].get("symbol", "").strip()
                source = cik_tickers[key].get("source", "cik_name_matches")
                note = cik_tickers[key].get("note", "")
            rows.append(EventSymbol(
                event_id=event_id,
                event_idx=event_idx,
                side=side,
                company_name=company,
                announce_date=str(event.get("Announce Date", "")),
                year=year,
                quarter=quarter,
                symbol=symbol,
                symbol_source=source,
                symbol_note=note,
            ))
    return rows


def request_plan(symbols: List[EventSymbol]) -> List[Dict[str, Any]]:
    rows = []
    for s in symbols:
        if not s.symbol:
            rows.append({**asdict(s), "request_kind": "missing_symbol", "endpoint": "", "status": "needs_manual_symbol"})
            continue
        for kind in ["institutional_summary", "institutional_holders", "etf_holders_of_stock"]:
            for endpoint in FMP_ENDPOINT_CANDIDATES[kind]:
                rows.append({
                    **asdict(s),
                    "request_kind": kind,
                    "endpoint": endpoint,
                    "status": "planned",
                })
    return rows


def fetch_for_symbols(client: FmpClient, symbols: List[EventSymbol]) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], List[Dict[str, Any]]]:
    inst_summary_rows: List[Dict[str, Any]] = []
    inst_holder_rows: List[Dict[str, Any]] = []
    etf_holder_rows: List[Dict[str, Any]] = []

    for s in symbols:
        if not s.symbol:
            continue
        base = asdict(s)
        common_params = {"symbol": s.symbol, "year": s.year, "quarter": s.quarter}

        rows, endpoint, err = client.get_first_available("institutional_summary", common_params)
        if rows:
            for r in rows:
                inst_summary_rows.append({**base, "provider_endpoint": endpoint, "provider_error": "", **r})
        else:
            inst_summary_rows.append({**base, "provider_endpoint": endpoint, "provider_error": err})

        rows, endpoint, err = client.get_first_available("institutional_holders", common_params)
        if rows:
            for r in rows:
                holder_name = str(first_present(r, ["holder", "holderName", "investorName", "name"], ""))
                holder_type = str(first_present(r, ["type", "holderType", "investorType"], ""))
                shares = first_present(r, ["shares", "share", "sharesNumber", "reportedHolding"], None)
                value = first_present(r, ["value", "marketValue", "valueHeld"], None)
                pct = first_present(r, ["ownership", "percent", "weight", "percentOfSharesOutstanding"], None)
                inst_holder_rows.append({
                    **base,
                    "provider_endpoint": endpoint,
                    "provider_error": "",
                    "holder_name": holder_name,
                    "holder_type_raw": holder_type,
                    "holder_category": classify_holder(holder_name, holder_type),
                    "shares": shares,
                    "market_value": value,
                    "ownership_percent": pct,
                    "raw_json": json.dumps(r, ensure_ascii=False),
                })
        else:
            inst_holder_rows.append({**base, "provider_endpoint": endpoint, "provider_error": err})

        rows, endpoint, err = client.get_first_available("etf_holders_of_stock", {"symbol": s.symbol})
        if rows:
            for r in rows:
                etf_name = str(first_present(r, ["etfName", "name", "holder", "holderName"], ""))
                etf_symbol = str(first_present(r, ["etfSymbol", "symbol", "ticker"], ""))
                shares = first_present(r, ["sharesNumber", "shares", "assetExposure", "marketValue"], None)
                weight = first_present(r, ["weightPercentage", "weight", "percentage", "weighting"], None)
                etf_holder_rows.append({
                    **base,
                    "provider_endpoint": endpoint,
                    "provider_error": "",
                    "etf_symbol": etf_symbol,
                    "etf_name": etf_name,
                    "shares_or_exposure": shares,
                    "weight": weight,
                    "raw_json": json.dumps(r, ensure_ascii=False),
                })
        else:
            etf_holder_rows.append({**base, "provider_endpoint": endpoint, "provider_error": err})

    return inst_summary_rows, inst_holder_rows, etf_holder_rows


def numeric_series(df: pd.DataFrame, col: str) -> pd.Series:
    if col not in df.columns:
        return pd.Series(dtype=float)
    return pd.to_numeric(df[col], errors="coerce").fillna(0.0)


def build_ownership_mix(inst_holders: pd.DataFrame, etf_holders: pd.DataFrame, symbols: List[EventSymbol]) -> pd.DataFrame:
    base_df = pd.DataFrame([asdict(s) for s in symbols])
    if base_df.empty:
        return base_df

    parts = []
    if not inst_holders.empty and "holder_category" in inst_holders.columns:
        h = inst_holders.copy()
        h["shares_num"] = numeric_series(h, "shares")
        grouped = (
            h.groupby(["event_id", "side", "holder_category"], dropna=False)
            .agg(holder_count=("holder_name", "count"), shares=("shares_num", "sum"))
            .reset_index()
        )
        pivot_shares = grouped.pivot_table(index=["event_id", "side"], columns="holder_category", values="shares", aggfunc="sum", fill_value=0).reset_index()
        pivot_counts = grouped.pivot_table(index=["event_id", "side"], columns="holder_category", values="holder_count", aggfunc="sum", fill_value=0).reset_index()
        pivot_shares.columns = [str(c) if c in {"event_id", "side"} else f"{c}_shares" for c in pivot_shares.columns]
        pivot_counts.columns = [str(c) if c in {"event_id", "side"} else f"{c}_holder_count" for c in pivot_counts.columns]
        parts.extend([pivot_shares, pivot_counts])

    if not etf_holders.empty:
        e = etf_holders.copy()
        e["etf_exposure_num"] = numeric_series(e, "shares_or_exposure")
        eagg = (
            e.groupby(["event_id", "side"], dropna=False)
            .agg(etf_holder_count=("etf_symbol", "count"), etf_shares_or_exposure=("etf_exposure_num", "sum"))
            .reset_index()
        )
        parts.append(eagg)

    out = base_df.copy()
    for p in parts:
        out = out.merge(p, on=["event_id", "side"], how="left")
    num_cols = [c for c in out.columns if c.endswith("_shares") or c.endswith("_holder_count") or c in {"etf_holder_count", "etf_shares_or_exposure"}]
    for c in num_cols:
        out[c] = pd.to_numeric(out[c], errors="coerce").fillna(0)
    return out


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Download ownership mix and ETF holder data for M&A events.")
    p.add_argument("--input", required=True, help="Raw Bloomberg M&A CSV or candidate_events.csv from the SEC downloader.")
    p.add_argument("--output-dir", default="ma_ownership_data", help="Output directory.")
    p.add_argument("--provider", choices=["fmp"], default="fmp", help="Ownership data provider.")
    p.add_argument("--fmp-api-key", default=None, help="FMP API key. If omitted, FMP_API_KEY is used.")
    p.add_argument("--dry-run", action="store_true", help="Write symbol map and request plan without calling the provider.")
    p.add_argument("--cik-matches", default=None, help="Optional cik_name_matches.csv from download_ma_edgar_files.py for ticker mapping.")
    p.add_argument("--min-cik-match-score", type=float, default=95.0,
                   help="Minimum CIK-name match score accepted for automatic ticker mapping. Historical targets often need overrides.")
    p.add_argument("--symbol-overrides", default=None,
                   help="Optional CSV with event_idx,side,symbol,note. Overrides SEC/current ticker mapping.")
    p.add_argument("--target-symbol-col", default=None,
                   help="Optional input CSV column containing the historical target ticker.")
    p.add_argument("--acquirer-symbol-col", default=None,
                   help="Optional input CSV column containing the acquirer ticker.")
    p.add_argument("--sides", nargs="*", default=["target"], choices=["target", "acquirer"],
                   help="Companies to fetch. For election demand, target is usually the relevant side.")
    p.add_argument("--ownership-lag-quarters", type=int, default=1,
                   help="Quarter lag used for available ownership data before the announcement date.")
    p.add_argument("--payment-types", nargs="*", default=["Cash or Stock", "Cash and Stock"])
    p.add_argument("--all-payment-types", action="store_true")
    p.add_argument("--deal-status", nargs="*", default=["Completed"])
    p.add_argument("--all-status", action="store_true")
    p.add_argument("--start-date", default=None)
    p.add_argument("--end-date", default=None)
    p.add_argument("--max-events", type=int, default=None)
    p.add_argument("--cache-dir", default=None)
    p.add_argument("--sleep-seconds", type=float, default=0.25)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    cache_dir = Path(args.cache_dir) if args.cache_dir else out_dir / "_cache"

    print(f"[{now_utc()}] Loading events: {args.input}")
    events = load_events(args)
    print(f"[{now_utc()}] Candidate events: {len(events):,}")

    overrides = load_symbol_overrides(args.symbol_overrides)
    cik_tickers = load_cik_tickers(args.cik_matches, min_score=args.min_cik_match_score)
    symbols = build_event_symbols(
        events=events,
        sides=args.sides,
        cik_tickers=cik_tickers,
        overrides=overrides,
        lag_quarters=args.ownership_lag_quarters,
        target_symbol_col=args.target_symbol_col,
        acquirer_symbol_col=args.acquirer_symbol_col,
    )
    symbol_df = pd.DataFrame([asdict(s) for s in symbols])
    symbol_df.to_csv(out_dir / "event_symbol_map.csv", index=False)
    missing = symbol_df[symbol_df["symbol"].astype(str).eq("")]
    missing.to_csv(out_dir / "missing_symbols.csv", index=False)
    override_template = missing[["event_idx", "side", "company_name", "announce_date"]].copy()
    if not override_template.empty:
        override_template["symbol"] = ""
        override_template["note"] = "fill historical Bloomberg ticker"
        override_template.to_csv(out_dir / "symbol_overrides_template.csv", index=False)
    print(f"[{now_utc()}] Symbols mapped: {len(symbol_df) - len(missing):,}/{len(symbol_df):,}")
    if len(missing):
        print(f"[{now_utc()}] Missing symbols written to: {out_dir / 'missing_symbols.csv'}")
        print(f"[{now_utc()}] Override template written to: {out_dir / 'symbol_overrides_template.csv'}")

    plan = request_plan(symbols)
    pd.DataFrame(plan).to_csv(out_dir / "ownership_request_plan.csv", index=False)

    api_key = args.fmp_api_key or os.environ.get("FMP_API_KEY", "")
    if args.dry_run or not api_key:
        print(f"[{now_utc()}] Dry run or missing FMP API key; wrote request plan only.")
        print(f"[{now_utc()}] Set FMP_API_KEY or pass --fmp-api-key to download ownership/ETF data.")
        return

    client = FmpClient(api_key=api_key, cache_dir=cache_dir, sleep_seconds=args.sleep_seconds)
    inst_summary, inst_holders, etf_holders = fetch_for_symbols(client, symbols)

    inst_summary_df = pd.DataFrame(inst_summary)
    inst_holders_df = pd.DataFrame(inst_holders)
    etf_holders_df = pd.DataFrame(etf_holders)
    inst_summary_df.to_csv(out_dir / "institutional_ownership_summary.csv", index=False)
    inst_holders_df.to_csv(out_dir / "institutional_holders.csv", index=False)
    etf_holders_df.to_csv(out_dir / "etf_holders_of_target.csv", index=False)

    mix_df = build_ownership_mix(inst_holders_df, etf_holders_df, symbols)
    mix_df.to_csv(out_dir / "ownership_mix_by_event.csv", index=False)

    print(f"[{now_utc()}] Wrote: {out_dir / 'institutional_ownership_summary.csv'}")
    print(f"[{now_utc()}] Wrote: {out_dir / 'institutional_holders.csv'}")
    print(f"[{now_utc()}] Wrote: {out_dir / 'etf_holders_of_target.csv'}")
    print(f"[{now_utc()}] Wrote: {out_dir / 'ownership_mix_by_event.csv'}")


if __name__ == "__main__":
    main()
