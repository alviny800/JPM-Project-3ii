#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
download_wrds_market_data.py

Build WRDS market/trading inputs for M&A election/proration research.

This is deliberately separate from the SEC/Claude legal-term extractor and the
ETF ownership helper:
- SEC/Claude extracts the legal payoff function and post-election labels.
- ownership helper extracts passive/ETF control.
- this script extracts WRDS market data needed to price and trade the payoff.
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import re
import sys
from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd

from download_ownership_etf_data import (
    EventSymbol,
    WrdsClient,
    build_event_symbols,
    clean_str,
    load_cik_tickers,
    load_events,
    load_symbol_overrides,
    numeric_value,
    qualified_col,
    qualified_table,
    quote_ident,
    read_secret_from_stdin,
    resolve_wrds_security_identifiers,
    to_date,
)


SHORT_VOLUME_SHORT_COLS = [
    "short_b", "short_fd", "short_fn", "short_fq", "short_n", "short_p",
    "short_q", "short_z", "short_fo", "short_a", "short_j", "short_k",
    "short_x", "short_y", "short_c", "short_fb", "short_m",
]

SHORT_VOLUME_TOTAL_COLS = [
    "total_b", "total_fd", "total_fn", "total_fq", "total_n", "total_p",
    "total_q", "total_z", "total_fo", "total_a", "total_j", "total_k",
    "total_x", "total_y", "total_c", "total_fb", "total_m",
]


def now_utc() -> str:
    return dt.datetime.utcnow().replace(microsecond=0).isoformat() + "Z"


def date_window(announce_date: Any, pre_days: int, post_days: int) -> Tuple[dt.date, dt.date, dt.date]:
    as_of = to_date(announce_date) or dt.date.today()
    return as_of, as_of - dt.timedelta(days=pre_days), as_of + dt.timedelta(days=post_days)


def parse_money(value: Any) -> Optional[float]:
    text = clean_str(value).replace(",", "")
    if not text:
        return None
    match = re.search(r"\$\s*([0-9]+(?:\.[0-9]+)?)", text)
    if match:
        return float(match.group(1))
    match = re.search(r"\b([0-9]+(?:\.[0-9]+)?)\s*(?:dollars?|usd)\b", text, flags=re.I)
    if match:
        return float(match.group(1))
    return None


def parse_exchange_ratio(value: Any) -> Optional[float]:
    text = clean_str(value).lower().replace(",", "")
    if not text:
        return None
    for pattern in [
        r"exchange ratio (?:of )?([0-9]+(?:\.[0-9]+)?)",
        r"([0-9]+(?:\.[0-9]+)?)\s*(?:share|shares)\s+of",
        r"\b([0-9]+(?:\.[0-9]+)?)\s*:\s*1\b",
        r"\b([0-9]+(?:\.[0-9]+)?)\b",
    ]:
        match = re.search(pattern, text)
        if match:
            return float(match.group(1))
    return None


def load_llm_terms(path: Optional[str]) -> Dict[str, Dict[str, Any]]:
    if not path:
        return {}
    df = pd.read_csv(path)
    out: Dict[str, Dict[str, Any]] = {}
    if "field_name" in df.columns:
        for _, row in df.iterrows():
            event_id = clean_str(row.get("event_id"))
            field_name = clean_str(row.get("field_name"))
            if not event_id or not field_name:
                continue
            out.setdefault(event_id, {})[field_name] = row.get("value")
    else:
        for _, row in df.iterrows():
            event_id = clean_str(row.get("event_id"))
            if not event_id:
                continue
            out[event_id] = {str(c): row.get(c) for c in df.columns}
    return out


def market_sql(args: argparse.Namespace, s: EventSymbol, start_date: dt.date, end_date: dt.date) -> Tuple[str, Dict[str, Any]]:
    table = qualified_table(args.wrds_market_library, args.wrds_market_table)
    cols = {
        "permno": quote_ident(args.wrds_market_permno_col),
        "date": quote_ident(args.wrds_market_date_col),
        "cusip": quote_ident(args.wrds_market_cusip_col),
        "prc": quote_ident(args.wrds_market_price_col),
        "openprc": quote_ident(args.wrds_market_open_col),
        "bid": quote_ident(args.wrds_market_bid_col),
        "ask": quote_ident(args.wrds_market_ask_col),
        "bidlo": quote_ident(args.wrds_market_bidlo_col),
        "askhi": quote_ident(args.wrds_market_askhi_col),
        "vol": quote_ident(args.wrds_market_volume_col),
        "ret": quote_ident(args.wrds_market_return_col),
        "shrout": quote_ident(args.wrds_market_shrout_col),
    }
    sql = f"""
        SELECT
            {cols['permno']} AS permno,
            {cols['date']} AS price_date,
            {cols['cusip']} AS cusip,
            {cols['prc']} AS prc,
            {cols['openprc']} AS openprc,
            {cols['bid']} AS bid,
            {cols['ask']} AS ask,
            {cols['bidlo']} AS bidlo,
            {cols['askhi']} AS askhi,
            {cols['vol']} AS volume,
            {cols['ret']} AS ret,
            {cols['shrout']} AS shrout
        FROM {table}
        WHERE {cols['permno']} = %(permno)s
          AND {cols['date']} BETWEEN %(start_date)s AND %(end_date)s
        ORDER BY {cols['date']}
    """
    return sql, {
        "permno": int(float(s.permno)),
        "start_date": start_date,
        "end_date": end_date,
    }


def enrich_market_df(df: pd.DataFrame, args: argparse.Namespace) -> pd.DataFrame:
    if df.empty:
        return df
    out = df.copy()
    out["price"] = pd.to_numeric(out["prc"], errors="coerce").abs()
    out["open_price"] = pd.to_numeric(out["openprc"], errors="coerce").abs()
    out["volume"] = pd.to_numeric(out["volume"], errors="coerce")
    out["bid"] = pd.to_numeric(out["bid"], errors="coerce").abs()
    out["ask"] = pd.to_numeric(out["ask"], errors="coerce").abs()
    out["bidlo"] = pd.to_numeric(out["bidlo"], errors="coerce").abs()
    out["askhi"] = pd.to_numeric(out["askhi"], errors="coerce").abs()
    out["shrout"] = pd.to_numeric(out["shrout"], errors="coerce") * float(args.wrds_shrout_multiplier)
    out["dollar_volume"] = out["price"] * out["volume"]
    out["market_cap"] = out["price"] * out["shrout"]
    out["bid_ask_spread"] = out["ask"] - out["bid"]
    out.loc[out["bid_ask_spread"] <= 0, "bid_ask_spread"] = out["askhi"] - out["bidlo"]
    midpoint = (out["ask"] + out["bid"]) / 2
    out["bid_ask_spread_pct"] = out["bid_ask_spread"] / midpoint.where(midpoint > 0)
    out["ret"] = pd.to_numeric(out["ret"], errors="coerce")
    return out


def fetch_market_data(
    client: WrdsClient,
    args: argparse.Namespace,
    symbols: List[EventSymbol],
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    daily_rows: List[Dict[str, Any]] = []
    snapshot_rows: List[Dict[str, Any]] = []

    for s in symbols:
        as_of, start_date, end_date = date_window(s.announce_date, args.pre_days, args.post_days)
        base = asdict(s)
        if not s.permno:
            snapshot_rows.append({
                **base,
                "as_of_date": as_of,
                "market_status": "missing_permno",
                "market_error": "No CRSP PERMNO available for WRDS market query.",
            })
            continue

        sql, params = market_sql(args, s, start_date, end_date)
        try:
            df = client.raw_sql(sql, params=params)
        except Exception as e:
            snapshot_rows.append({
                **base,
                "as_of_date": as_of,
                "market_status": "query_error",
                "market_error": str(e),
            })
            continue

        df = enrich_market_df(df, args)
        for _, row in df.iterrows():
            daily_rows.append({
                **base,
                "as_of_date": as_of,
                "price_date": row.get("price_date"),
                "price": row.get("price"),
                "open_price": row.get("open_price"),
                "bid": row.get("bid"),
                "ask": row.get("ask"),
                "bidlo": row.get("bidlo"),
                "askhi": row.get("askhi"),
                "bid_ask_spread": row.get("bid_ask_spread"),
                "bid_ask_spread_pct": row.get("bid_ask_spread_pct"),
                "volume": row.get("volume"),
                "dollar_volume": row.get("dollar_volume"),
                "ret": row.get("ret"),
                "shares_outstanding": row.get("shrout"),
                "market_cap": row.get("market_cap"),
            })

        hist = df[pd.to_datetime(df["price_date"]).dt.date <= as_of].copy()
        if hist.empty:
            snapshot_rows.append({
                **base,
                "as_of_date": as_of,
                "market_status": "not_found",
                "market_error": "No CRSP rows on or before as-of date within lookback window.",
            })
            continue

        snap = hist.iloc[-1]
        adv20 = hist.tail(20)
        adv60 = hist.tail(60)
        snapshot_rows.append({
            **base,
            "as_of_date": as_of,
            "market_status": "ok",
            "market_error": "",
            "price_date": snap.get("price_date"),
            "price": snap.get("price"),
            "open_price": snap.get("open_price"),
            "bid": snap.get("bid"),
            "ask": snap.get("ask"),
            "bidlo": snap.get("bidlo"),
            "askhi": snap.get("askhi"),
            "bid_ask_spread": snap.get("bid_ask_spread"),
            "bid_ask_spread_pct": snap.get("bid_ask_spread_pct"),
            "volume": snap.get("volume"),
            "dollar_volume": snap.get("dollar_volume"),
            "ret": snap.get("ret"),
            "shares_outstanding": snap.get("shrout"),
            "market_cap": snap.get("market_cap"),
            "adv20": adv20["volume"].mean(),
            "dollar_adv20": adv20["dollar_volume"].mean(),
            "adv60": adv60["volume"].mean(),
            "dollar_adv60": adv60["dollar_volume"].mean(),
            "trading_days_in_window": len(df),
        })

    return pd.DataFrame(daily_rows), pd.DataFrame(snapshot_rows)


def fetch_short_volume(
    client: WrdsClient,
    args: argparse.Namespace,
    symbols: List[EventSymbol],
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    if not args.include_short_volume:
        return pd.DataFrame(), pd.DataFrame()

    daily_rows: List[Dict[str, Any]] = []
    snapshot_rows: List[Dict[str, Any]] = []
    table = qualified_table(args.wrds_short_volume_library, args.wrds_short_volume_table)
    date_col = quote_ident(args.wrds_short_volume_date_col)
    symbol_col = quote_ident(args.wrds_short_volume_symbol_col)
    short_expr = " + ".join(f"COALESCE({quote_ident(c)}, 0)" for c in SHORT_VOLUME_SHORT_COLS)
    total_expr = " + ".join(f"COALESCE({quote_ident(c)}, 0)" for c in SHORT_VOLUME_TOTAL_COLS)

    for s in symbols:
        as_of, start_date, end_date = date_window(s.announce_date, args.pre_days, args.post_days)
        base = asdict(s)
        if not s.symbol:
            snapshot_rows.append({
                **base,
                "as_of_date": as_of,
                "short_volume_status": "missing_symbol",
                "short_volume_error": "No ticker symbol available.",
            })
            continue
        sql = f"""
            SELECT
                {date_col} AS short_volume_date,
                {symbol_col} AS symbol,
                ({short_expr}) AS short_volume,
                ({total_expr}) AS total_volume
            FROM {table}
            WHERE upper({symbol_col}) = %(symbol)s
              AND {date_col} BETWEEN %(start_date)s AND %(end_date)s
            ORDER BY {date_col}
        """
        try:
            df = client.raw_sql(sql, params={
                "symbol": s.symbol.upper(),
                "start_date": start_date,
                "end_date": end_date,
            })
        except Exception as e:
            snapshot_rows.append({
                **base,
                "as_of_date": as_of,
                "short_volume_status": "query_error",
                "short_volume_error": str(e),
            })
            continue

        if df.empty:
            snapshot_rows.append({
                **base,
                "as_of_date": as_of,
                "short_volume_status": "not_found",
                "short_volume_error": "No short-volume rows in configured table.",
            })
            continue

        df["short_volume"] = pd.to_numeric(df["short_volume"], errors="coerce")
        df["total_volume"] = pd.to_numeric(df["total_volume"], errors="coerce")
        df["short_volume_ratio"] = df["short_volume"] / df["total_volume"].where(df["total_volume"] > 0)
        for _, row in df.iterrows():
            daily_rows.append({
                **base,
                "as_of_date": as_of,
                "short_volume_date": row.get("short_volume_date"),
                "short_volume": row.get("short_volume"),
                "short_total_volume": row.get("total_volume"),
                "short_volume_ratio": row.get("short_volume_ratio"),
            })
        hist = df[pd.to_datetime(df["short_volume_date"]).dt.date <= as_of].copy()
        if hist.empty:
            snapshot_rows.append({
                **base,
                "as_of_date": as_of,
                "short_volume_status": "not_found_before_asof",
                "short_volume_error": "",
            })
            continue
        snap = hist.iloc[-1]
        last20 = hist.tail(20)
        snapshot_rows.append({
            **base,
            "as_of_date": as_of,
            "short_volume_status": "ok",
            "short_volume_error": "",
            "short_volume_date": snap.get("short_volume_date"),
            "short_volume": snap.get("short_volume"),
            "short_total_volume": snap.get("total_volume"),
            "short_volume_ratio": snap.get("short_volume_ratio"),
            "short_volume_ratio_20d_avg": last20["short_volume_ratio"].mean(),
        })

    return pd.DataFrame(daily_rows), pd.DataFrame(snapshot_rows)


def side_row(snapshot: pd.DataFrame, event_id: str, side: str) -> Optional[pd.Series]:
    hit = snapshot[(snapshot["event_id"].astype(str) == str(event_id)) & (snapshot["side"].astype(str) == side)]
    if hit.empty:
        return None
    return hit.iloc[0]


def build_event_features(
    events: pd.DataFrame,
    market_snapshot: pd.DataFrame,
    short_snapshot: pd.DataFrame,
    llm_terms: Dict[str, Dict[str, Any]],
) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []
    for _, ev in events.iterrows():
        event_idx = int(ev.get("orig_row_idx", ev.name if ev.name is not None else -1))
        from download_ownership_etf_data import event_id_from_row

        event_id = event_id_from_row(event_idx, ev)
        target = side_row(market_snapshot, event_id, "target") if not market_snapshot.empty else None
        acquirer = side_row(market_snapshot, event_id, "acquirer") if not market_snapshot.empty else None
        terms = llm_terms.get(event_id, {})
        cash = parse_money(terms.get("cash_consideration_per_share"))
        exchange_ratio = parse_exchange_ratio(terms.get("exchange_ratio"))
        consideration_menu = clean_str(terms.get("consideration_menu")).lower()
        payment_type = clean_str(ev.get("Payment Type", "")).lower()
        is_election_menu = (
            "cash or stock" in payment_type
            or ("cash election" in consideration_menu and "stock election" in consideration_menu)
        )
        target_price = numeric_value(target.get("price")) if target is not None else None
        acquirer_price = numeric_value(acquirer.get("price")) if acquirer is not None else None
        fixed_stock_value = None
        cash_election_value = cash
        stock_election_value = None
        best_menu_value = None
        deal_value = None
        deal_spread = None
        deal_spread_pct = None
        if exchange_ratio is not None and acquirer_price is not None:
            fixed_stock_value = exchange_ratio * acquirer_price
            stock_election_value = fixed_stock_value
        menu_values = [v for v in [cash_election_value, stock_election_value] if v is not None]
        if menu_values:
            best_menu_value = max(menu_values)
        if is_election_menu:
            deal_value = best_menu_value
        elif cash is not None or fixed_stock_value is not None:
            deal_value = (cash or 0.0) + (fixed_stock_value or 0.0)
        if deal_value is not None and target_price is not None:
            deal_spread = deal_value - target_price
            if target_price:
                deal_spread_pct = deal_spread / target_price

        target_short = side_row(short_snapshot, event_id, "target") if not short_snapshot.empty else None

        row = {
            "event_id": event_id,
            "event_idx": event_idx,
            "target_name": ev.get("Target Name", ""),
            "acquirer_name": ev.get("Acquirer Name", ""),
            "announce_date": ev.get("Announce Date", ""),
            "payment_type": ev.get("Payment Type", ""),
            "deal_status": ev.get("Deal Status", ""),
            "target_price": target_price,
            "target_price_date": target.get("price_date") if target is not None else "",
            "target_volume": target.get("volume") if target is not None else None,
            "target_dollar_volume": target.get("dollar_volume") if target is not None else None,
            "target_adv20": target.get("adv20") if target is not None else None,
            "target_dollar_adv20": target.get("dollar_adv20") if target is not None else None,
            "target_bid_ask_spread_pct": target.get("bid_ask_spread_pct") if target is not None else None,
            "target_market_cap": target.get("market_cap") if target is not None else None,
            "target_shares_outstanding": target.get("shares_outstanding") if target is not None else None,
            "acquirer_price": acquirer_price,
            "acquirer_price_date": acquirer.get("price_date") if acquirer is not None else "",
            "acquirer_volume": acquirer.get("volume") if acquirer is not None else None,
            "acquirer_dollar_volume": acquirer.get("dollar_volume") if acquirer is not None else None,
            "acquirer_adv20": acquirer.get("adv20") if acquirer is not None else None,
            "acquirer_dollar_adv20": acquirer.get("dollar_adv20") if acquirer is not None else None,
            "acquirer_bid_ask_spread_pct": acquirer.get("bid_ask_spread_pct") if acquirer is not None else None,
            "cash_consideration_per_share_num": cash,
            "exchange_ratio_num": exchange_ratio,
            "is_election_menu": is_election_menu,
            "cash_election_value": cash_election_value,
            "stock_election_value": stock_election_value,
            "best_menu_value": best_menu_value,
            "fixed_stock_value": fixed_stock_value,
            "simple_contract_value": deal_value,
            "deal_spread": deal_spread,
            "deal_spread_pct": deal_spread_pct,
            "short_volume_ratio": target_short.get("short_volume_ratio") if target_short is not None else None,
            "short_volume_ratio_20d_avg": target_short.get("short_volume_ratio_20d_avg") if target_short is not None else None,
            "market_feature_notes": "",
        }
        if is_election_menu:
            row["market_feature_notes"] = "deal_spread uses best menu value before proration; model EV/spread is computed in build_election_strategy_model.py."
        if deal_value is None:
            row["market_feature_notes"] = "simple_contract_value_not_computable_from supplied LLM terms; election/proration EV needs model terms."
        rows.append(row)
    return pd.DataFrame(rows)


def build_missing_inventory(features: pd.DataFrame, include_short_volume: bool) -> pd.DataFrame:
    checks = [
        ("target_price", "wrds_crsp_dsf", "target_price"),
        ("acquirer_price", "wrds_crsp_dsf", "acquirer_price"),
        ("deal_spread", "computed_from_sec_terms_and_wrds_prices", "deal_spread"),
        ("volume_liquidity", "wrds_crsp_dsf", "target_adv20"),
        ("short_volume_ratio_proxy", "wrds_shortvolume_or_configured_short_volume_table", "short_volume_ratio"),
    ]
    rows: List[Dict[str, Any]] = []
    for field, source, col in checks:
        status = "not_requested" if field == "short_volume_ratio_proxy" and not include_short_volume else "missing"
        if col in features.columns and features[col].notna().any():
            status = "available"
        rows.append({"field": field, "source": source, "status": status, "notes": ""})
    rows.extend([
        {
            "field": "short_interest",
            "source": "WRDS true short-interest position table or vendor feed",
            "status": "not_available_in_current_detected_wrds_libraries",
            "notes": "Only short-volume sample/proxy tables were detected, not true exchange short-interest positions.",
        },
        {
            "field": "borrow_cost",
            "source": "Markit/S3/DataLend/prime broker securities lending",
            "status": "not_available_in_current_detected_wrds_tables",
            "notes": "Markit MSF libraries were visible but returned no accessible tables for this account in schema probe.",
        },
        {
            "field": "borrow_availability",
            "source": "Markit/S3/DataLend/prime broker locate or securities lending inventory",
            "status": "not_available_in_current_detected_wrds_tables",
            "notes": "Requires a securities-lending subscription or prime broker locate feed.",
        },
        {
            "field": "nport_on_loan",
            "source": "SEC EDGAR N-PORT filings or structured N-PORT feed",
            "status": "future_sec_edgar_extension",
            "notes": "Can be built as a separate EDGAR/N-PORT parser keyed by fund CIK and target CUSIP.",
        },
        {
            "field": "fund_lending_policy",
            "source": "SEC EDGAR fund SAI/prospectus/N-1A filings",
            "status": "future_sec_edgar_extension",
            "notes": "Text extraction problem; should be linked from ETF/fund rows to fund filings.",
        },
    ])
    return pd.DataFrame(rows)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Download WRDS market/trading data for M&A events.")
    p.add_argument("--input", required=True, help="Raw Bloomberg M&A CSV or candidate_events.csv from SEC downloader.")
    p.add_argument("--output-dir", default="ma_market_wrds", help="Output directory.")
    p.add_argument("--llm-extractions", default=None, help="Optional llm_field_extractions.csv for simple deal-spread computation.")
    p.add_argument("--wrds-username", default=None)
    p.add_argument("--wrds-password", default=None)
    p.add_argument("--wrds-password-stdin", action="store_true")
    p.add_argument("--dry-run", action="store_true", help="Write identifier map and request plan without querying market tables.")
    p.add_argument("--cik-matches", default=None)
    p.add_argument("--min-cik-match-score", type=float, default=95.0)
    p.add_argument("--symbol-overrides", default=None)
    p.add_argument("--target-symbol-col", default=None)
    p.add_argument("--acquirer-symbol-col", default=None)
    p.add_argument("--target-permno-col", default=None)
    p.add_argument("--acquirer-permno-col", default=None)
    p.add_argument("--target-cusip-col", default=None)
    p.add_argument("--acquirer-cusip-col", default=None)
    p.add_argument("--sides", nargs="*", default=["target", "acquirer"], choices=["target", "acquirer"])
    p.add_argument("--pre-days", type=int, default=90)
    p.add_argument("--post-days", type=int, default=30)
    p.add_argument("--payment-types", nargs="*", default=["Cash or Stock", "Cash and Stock"])
    p.add_argument("--all-payment-types", action="store_true")
    p.add_argument("--deal-status", nargs="*", default=["Completed"])
    p.add_argument("--all-status", action="store_true")
    p.add_argument("--start-date", default=None)
    p.add_argument("--end-date", default=None)
    p.add_argument("--max-events", type=int, default=None)

    p.add_argument("--wrds-security-library", default="crsp")
    p.add_argument("--wrds-security-table", default="stocknames")
    p.add_argument("--wrds-security-permno-col", default="permno")
    p.add_argument("--wrds-security-ticker-col", default="ticker")
    p.add_argument("--wrds-security-name-col", default="comnam")
    p.add_argument("--wrds-security-cusip-col", default="ncusip")
    p.add_argument("--wrds-security-start-date-col", default="namedt")
    p.add_argument("--wrds-security-end-date-col", default="nameenddt")
    p.add_argument("--wrds-security-candidate-limit", type=int, default=50)
    p.add_argument("--wrds-security-min-score", type=float, default=84.0)

    p.add_argument("--wrds-market-library", default="crsp")
    p.add_argument("--wrds-market-table", default="dsf")
    p.add_argument("--wrds-market-permno-col", default="permno")
    p.add_argument("--wrds-market-date-col", default="date")
    p.add_argument("--wrds-market-cusip-col", default="cusip")
    p.add_argument("--wrds-market-price-col", default="prc")
    p.add_argument("--wrds-market-open-col", default="openprc")
    p.add_argument("--wrds-market-bid-col", default="bid")
    p.add_argument("--wrds-market-ask-col", default="ask")
    p.add_argument("--wrds-market-bidlo-col", default="bidlo")
    p.add_argument("--wrds-market-askhi-col", default="askhi")
    p.add_argument("--wrds-market-volume-col", default="vol")
    p.add_argument("--wrds-market-return-col", default="ret")
    p.add_argument("--wrds-market-shrout-col", default="shrout")
    p.add_argument("--wrds-shrout-multiplier", type=float, default=1000.0)

    p.add_argument("--include-short-volume", action="store_true", help="Fetch WRDS short-volume proxy if the configured table is available.")
    p.add_argument("--wrds-short-volume-library", default="wrds_shortvolume_samp")
    p.add_argument("--wrds-short-volume-table", default="wrds_shortvolume_samp")
    p.add_argument("--wrds-short-volume-date-col", default="date")
    p.add_argument("--wrds-short-volume-symbol-col", default="symbol")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

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
        lag_quarters=0,
        target_symbol_col=args.target_symbol_col,
        acquirer_symbol_col=args.acquirer_symbol_col,
        target_permno_col=args.target_permno_col,
        acquirer_permno_col=args.acquirer_permno_col,
        target_cusip_col=args.target_cusip_col,
        acquirer_cusip_col=args.acquirer_cusip_col,
    )

    if args.dry_run:
        symbol_df = pd.DataFrame([asdict(s) for s in symbols])
        symbol_df.to_csv(out_dir / "event_security_map.csv", index=False)
        plan_rows = []
        for s in symbols:
            as_of, start_date, end_date = date_window(s.announce_date, args.pre_days, args.post_days)
            plan_rows.append({
                **asdict(s),
                "request_kind": "market_data",
                "endpoint": f"wrds:{args.wrds_market_library}.{args.wrds_market_table}",
                "as_of_date": as_of,
                "start_date": start_date,
                "end_date": end_date,
            })
        pd.DataFrame(plan_rows).to_csv(out_dir / "wrds_market_request_plan.csv", index=False)
        print(f"[{now_utc()}] Dry run; wrote market request plan only.")
        return

    wrds_username = args.wrds_username or os.environ.get("WRDS_USERNAME", "")
    if args.wrds_password_stdin:
        wrds_password = read_secret_from_stdin("WRDS password: ")
    else:
        wrds_password = args.wrds_password or os.environ.get("WRDS_PASSWORD", "")
    print(f"[{now_utc()}] Connecting to WRDS")
    client = WrdsClient(username=wrds_username, password=wrds_password)
    try:
        symbols, security_map = resolve_wrds_security_identifiers(client, symbols, args)
        pd.DataFrame(security_map).to_csv(out_dir / "wrds_security_map.csv", index=False)
        pd.DataFrame([asdict(s) for s in symbols]).to_csv(out_dir / "event_security_map.csv", index=False)
        print(f"[{now_utc()}] Wrote security maps")

        daily, snapshot = fetch_market_data(client, args, symbols)
        daily.to_csv(out_dir / "wrds_market_daily.csv", index=False)
        snapshot.to_csv(out_dir / "wrds_market_snapshot.csv", index=False)
        print(f"[{now_utc()}] Wrote market data")

        short_daily, short_snapshot = fetch_short_volume(client, args, symbols)
        if args.include_short_volume:
            short_daily.to_csv(out_dir / "wrds_short_volume_daily.csv", index=False)
            short_snapshot.to_csv(out_dir / "wrds_short_volume_snapshot.csv", index=False)
            print(f"[{now_utc()}] Wrote short-volume proxy data")
    finally:
        client.close()

    terms = load_llm_terms(args.llm_extractions)
    features = build_event_features(events, snapshot, short_snapshot, terms)
    features.to_csv(out_dir / "event_market_features.csv", index=False)
    missing = build_missing_inventory(features, include_short_volume=args.include_short_volume)
    missing.to_csv(out_dir / "missing_market_data_inventory.csv", index=False)
    print(f"[{now_utc()}] Wrote: {out_dir / 'event_market_features.csv'}")
    print(f"[{now_utc()}] Wrote: {out_dir / 'missing_market_data_inventory.csv'}")


if __name__ == "__main__":
    main()
