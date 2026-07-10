#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
download_ownership_etf_data.py

Build ownership-mix and ETF-holdings inputs for the M&A election/proration model.

This script is deliberately separate from download_ma_edgar_files.py:
- SEC/Claude extracts deal terms and realized proration labels.
- This script adds target-holder composition data for election-demand forecasts.
- Price, borrow cost, and execution data can be merged later from Bloomberg.

Default provider: Financial Modeling Prep (FMP). WRDS is also supported for
CRSP-style ETF/fund holdings and shares-outstanding snapshots. Both providers
support a dry-run mode that creates identifier maps and request plans.
"""

from __future__ import annotations

import argparse
import datetime as dt
import difflib
import getpass
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

from event_csv_adapter import (
    ACQUIRER_CUSIP_COL,
    ACQUIRER_SYMBOL_COL,
    TARGET_CUSIP_COL,
    TARGET_SYMBOL_COL,
    normalize_event_input,
)


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
    r"\bfirst trust\b", r"\bproshares\b", r"\bdirexion\b", r"\bglobal x\b",
    r"\bwisdomtree\b",
]

COMPANY_STOP_WORDS = {
    "the", "inc", "corp", "corporation", "co", "company", "ltd", "limited",
    "plc", "lp", "llc", "holdings", "holding", "group", "sa", "nv", "ag",
    "spa", "bv", "gmbh", "trust", "fund", "class", "common", "stock",
}

SQL_IDENT_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")


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


def clean_str(value: Any) -> str:
    if value is None or pd.isna(value):
        return ""
    text = str(value).strip()
    if text.lower() in {"", "nan", "none", "null"}:
        return ""
    return text


def first_nonempty(*values: Any) -> str:
    for value in values:
        text = clean_str(value)
        if text:
            return text
    return ""


def optional_input_value(row: pd.Series, column_name: Optional[str]) -> str:
    if column_name and column_name in row.index:
        return clean_str(row.get(column_name))
    return ""


def quote_ident(name: str) -> str:
    if not SQL_IDENT_RE.match(name or ""):
        raise ValueError(f"Unsafe SQL identifier: {name!r}")
    return f'"{name}"'


def qualified_table(library: str, table: str) -> str:
    return f"{quote_ident(library)}.{quote_ident(table)}"


def qualified_col(alias: str, column: str) -> str:
    if not column:
        raise ValueError("SQL column name cannot be empty")
    return f"{quote_ident(alias)}.{quote_ident(column)}"


def normalize_cusip(value: Any) -> str:
    text = re.sub(r"[^A-Za-z0-9]+", "", clean_str(value)).upper()
    return text[:9]


def to_date(value: Any) -> Optional[dt.date]:
    ts = parse_date(value)
    if ts is None or pd.isna(ts):
        return None
    return pd.Timestamp(ts).date()


def numeric_value(value: Any) -> Optional[float]:
    try:
        if value is None or pd.isna(value):
            return None
        return float(value)
    except Exception:
        return None


def company_query_token(name: str) -> str:
    tokens = [t for t in normalize_name(name).split() if t not in COMPANY_STOP_WORDS and len(t) >= 3]
    if not tokens:
        return ""
    return max(tokens, key=len)


def name_match_score(left: str, right: str) -> float:
    a = normalize_name(left)
    b = normalize_name(right)
    if not a or not b:
        return 0.0
    if a == b:
        return 100.0
    a_tokens = set(t for t in a.split() if t not in COMPANY_STOP_WORDS)
    b_tokens = set(t for t in b.split() if t not in COMPANY_STOP_WORDS)
    if a_tokens and a_tokens.issubset(b_tokens):
        token_score = 95.0
    elif a_tokens and b_tokens and a_tokens.intersection(b_tokens):
        token_score = 80.0 + 15.0 * len(a_tokens.intersection(b_tokens)) / max(len(a_tokens), 1)
    else:
        token_score = 0.0
    seq_score = difflib.SequenceMatcher(None, a, b).ratio() * 100.0
    return max(token_score, seq_score)


def flag_truthy(value: Any) -> bool:
    text = clean_str(value).lower()
    if text in {"", "0", "n", "no", "false", "f", "none", "null"}:
        return False
    return True


def classify_wrds_fund(row: Dict[str, Any]) -> Tuple[bool, str]:
    fund_name = str(first_present(row, ["fund_name", "etf_name", "holder_name"], ""))
    fund_ticker = str(first_present(row, ["fund_ticker", "etf_symbol"], ""))
    flag_bits = [
        ("et_flag", first_present(row, ["et_flag", "fund_et_flag"], "")),
        ("index_fund_flag", first_present(row, ["index_fund_flag", "fund_index_flag"], "")),
    ]
    for label, value in flag_bits:
        if flag_truthy(value):
            return True, f"{label}={value}"
    norm = normalize_name(f"{fund_name} {fund_ticker}")
    if any(re.search(p, norm) for p in ETF_NAME_PATTERNS):
        return True, "name_pattern"
    return False, "not_matched"


def split_pgpass_line(line: str) -> List[str]:
    fields: List[str] = []
    cur: List[str] = []
    escaped = False
    for ch in line.rstrip("\n"):
        if escaped:
            cur.append(ch)
            escaped = False
        elif ch == "\\":
            escaped = True
        elif ch == ":":
            fields.append("".join(cur))
            cur = []
        else:
            cur.append(ch)
    fields.append("".join(cur))
    return fields


def pgpass_match(pattern: str, value: str) -> bool:
    return pattern == "*" or pattern == value


def read_pgpass_credentials(
    username: str,
    hostname: str,
    port: str,
    database: str,
    path: Optional[Path] = None,
) -> Tuple[str, str]:
    pgpass = path or Path.home() / ".pgpass"
    if not pgpass.exists() or not pgpass.is_file():
        return username, ""
    try:
        lines = pgpass.read_text(encoding="utf-8").splitlines()
    except Exception:
        return username, ""
    fallback: Tuple[str, str] = (username, "")
    for raw in lines:
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        fields = split_pgpass_line(line)
        if len(fields) != 5:
            continue
        host_pat, port_pat, db_pat, user_pat, password = fields
        if not (
            pgpass_match(host_pat, hostname)
            and pgpass_match(port_pat, str(port))
            and pgpass_match(db_pat, database)
        ):
            continue
        if username and pgpass_match(user_pat, username):
            return username, password
        if not username and user_pat != "*":
            return user_pat, password
        if not fallback[1]:
            fallback = (username, password)
    return fallback


def read_secret_from_stdin(prompt: str) -> str:
    if sys.stdin.isatty():
        return getpass.getpass(prompt).strip()
    return sys.stdin.readline().strip()


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


class WrdsClient:
    def __init__(self, username: str = "", password: str = "", verbose: bool = False) -> None:
        try:
            import wrds  # type: ignore
            import wrds.sql as wrds_sql  # type: ignore
        except Exception as e:
            raise RuntimeError("WRDS provider requires the optional 'wrds' Python package.") from e

        if not username:
            username = os.environ.get("PGUSER", "") or os.environ.get("USER", "")
        if not password:
            username, password = read_pgpass_credentials(
                username=username,
                hostname=str(wrds_sql.WRDS_POSTGRES_HOST),
                port=str(wrds_sql.WRDS_POSTGRES_PORT),
                database=str(wrds_sql.WRDS_POSTGRES_DB),
            )
        if not password and not sys.stdin.isatty():
            raise RuntimeError(
                "WRDS password is not available non-interactively. "
                "Set WRDS_PASSWORD or add a matching .pgpass entry."
            )

        kwargs: Dict[str, Any] = {"verbose": verbose}
        if username:
            kwargs["wrds_username"] = username
        if password:
            kwargs["wrds_password"] = password
        self.db = wrds.Connection(autoconnect=False, **kwargs)
        # Avoid wrds.Connection.connect(), which prompts interactively after a
        # failed first attempt. In batch jobs we want the real auth/network error.
        self.db._Connection__make_sa_engine_conn(raise_err=True)
        self.db.load_library_list()

    def raw_sql(self, sql: str, params: Optional[Dict[str, Any]] = None) -> pd.DataFrame:
        return self.db.raw_sql(sql, params=params or {})

    def close(self) -> None:
        try:
            self.db.close()
        except Exception:
            pass


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
    permno: str
    cusip: str
    security_name: str
    security_source: str
    security_note: str


def load_events(args: argparse.Namespace) -> pd.DataFrame:
    df = normalize_event_input(pd.read_csv(args.input))
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
    target_permno_col: Optional[str],
    acquirer_permno_col: Optional[str],
    target_cusip_col: Optional[str],
    acquirer_cusip_col: Optional[str],
) -> List[EventSymbol]:
    rows: List[EventSymbol] = []
    side_to_col = {"target": "Target Name", "acquirer": "Acquirer Name"}
    side_to_symbol_col = {"target": target_symbol_col, "acquirer": acquirer_symbol_col}
    side_to_permno_col = {"target": target_permno_col, "acquirer": acquirer_permno_col}
    side_to_cusip_col = {"target": target_cusip_col, "acquirer": acquirer_cusip_col}
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
            permno = ""
            cusip = ""
            security_name = ""
            security_source = ""
            security_note = ""
            if key in overrides:
                symbol = overrides[key].get("symbol", "").strip()
                permno = overrides[key].get("permno", "").strip()
                cusip = normalize_cusip(overrides[key].get("cusip", ""))
                security_name = overrides[key].get("security_name", "").strip()
                source = "override"
                note = overrides[key].get("note", "")
                security_source = "override"
                security_note = note
            else:
                symbol = optional_input_value(event, side_to_symbol_col.get(side))
                permno = optional_input_value(event, side_to_permno_col.get(side))
                cusip = normalize_cusip(optional_input_value(event, side_to_cusip_col.get(side)))
                supplied = []
                if symbol:
                    supplied.append(f"symbol={side_to_symbol_col[side]}")
                if permno:
                    supplied.append(f"permno={side_to_permno_col[side]}")
                if cusip:
                    supplied.append(f"cusip={side_to_cusip_col[side]}")
                if supplied:
                    source = "input_column"
                    note = "; ".join(supplied)
                    security_source = "input_column"
                    security_note = note
            if not symbol and key in cik_tickers:
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
                permno=permno,
                cusip=cusip,
                security_name=security_name,
                security_source=security_source,
                security_note=security_note,
            ))
    return rows


def request_plan(symbols: List[EventSymbol], provider: str, args: argparse.Namespace) -> List[Dict[str, Any]]:
    rows = []
    for s in symbols:
        if provider == "fmp":
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
        elif provider == "wrds":
            rows.append({
                **asdict(s),
                "request_kind": "security_lookup",
                "endpoint": f"wrds:{args.wrds_security_library}.{args.wrds_security_table}",
                "status": "planned" if not (s.permno or s.cusip or s.symbol) else "planned_or_already_identified",
            })
            if s.permno or s.cusip or s.symbol:
                rows.append({
                    **asdict(s),
                    "request_kind": "etf_holders_of_stock",
                    "endpoint": f"wrds:{args.wrds_holdings_library}.{args.wrds_holdings_table}",
                    "status": "planned",
                })
                if s.permno:
                    rows.append({
                        **asdict(s),
                        "request_kind": "shares_outstanding",
                        "endpoint": f"wrds:{args.wrds_shares_library}.{args.wrds_shares_table}",
                        "status": "planned",
                    })
            else:
                rows.append({
                    **asdict(s),
                    "request_kind": "etf_holders_of_stock",
                    "endpoint": f"wrds:{args.wrds_holdings_library}.{args.wrds_holdings_table}",
                    "status": "needs_symbol_cusip_permno_or_wrds_security_lookup",
                })
        else:
            rows.append({
                **asdict(s),
                "request_kind": "unsupported_provider",
                "endpoint": "",
                "status": f"unsupported_provider:{provider}",
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


def resolve_wrds_security_identifiers(
    client: WrdsClient,
    symbols: List[EventSymbol],
    args: argparse.Namespace,
) -> Tuple[List[EventSymbol], List[Dict[str, Any]]]:
    """Fill missing PERMNO/CUSIP/ticker from a WRDS security-name table.

    The defaults target CRSP's stocknames table, but every table/column name is
    configurable because WRDS subscriptions differ across institutions.
    """
    out_rows: List[Dict[str, Any]] = []
    table = qualified_table(args.wrds_security_library, args.wrds_security_table)
    permno_col = quote_ident(args.wrds_security_permno_col)
    ticker_col = quote_ident(args.wrds_security_ticker_col)
    name_col = quote_ident(args.wrds_security_name_col)
    cusip_col = quote_ident(args.wrds_security_cusip_col)
    start_col = quote_ident(args.wrds_security_start_date_col)
    end_col = quote_ident(args.wrds_security_end_date_col)

    for s in symbols:
        as_of = to_date(s.announce_date) or dt.date.today()
        base = asdict(s)
        if s.permno and (s.cusip or s.symbol):
            out_rows.append({**base, "wrds_lookup_status": "already_identified", "wrds_lookup_error": ""})
            continue

        token = company_query_token(s.company_name)
        params: Dict[str, Any] = {
            "as_of_date": as_of,
            "symbol": s.symbol.upper(),
            "name_like": f"%{token.lower()}%" if token else "%",
            "limit": int(args.wrds_security_candidate_limit),
        }
        if s.symbol:
            match_predicate = f"upper({ticker_col}) = %(symbol)s"
        else:
            match_predicate = f"lower({name_col}) LIKE %(name_like)s"

        sql = f"""
            SELECT
                {permno_col} AS permno,
                {ticker_col} AS ticker,
                {name_col} AS company_name,
                {cusip_col} AS cusip,
                {start_col} AS name_start_date,
                {end_col} AS name_end_date
            FROM {table}
            WHERE {start_col} <= %(as_of_date)s
              AND ({end_col} IS NULL OR {end_col} >= %(as_of_date)s)
              AND {match_predicate}
            LIMIT %(limit)s
        """
        try:
            df = client.raw_sql(sql, params=params)
        except Exception as e:
            out_rows.append({
                **base,
                "wrds_lookup_status": "error",
                "wrds_lookup_error": str(e),
                "wrds_lookup_sql": "security_lookup",
            })
            continue

        if df.empty:
            out_rows.append({
                **base,
                "wrds_lookup_status": "not_found",
                "wrds_lookup_error": "",
                "wrds_lookup_candidate_count": 0,
            })
            continue

        candidates = []
        for _, row in df.iterrows():
            score = name_match_score(s.company_name, str(row.get("company_name", "")))
            ticker = clean_str(row.get("ticker"))
            if s.symbol and ticker.upper() == s.symbol.upper():
                score = max(score, 98.0)
            candidates.append((score, row))
        candidates.sort(key=lambda x: x[0], reverse=True)
        best_score, best = candidates[0]
        status = "matched" if best_score >= args.wrds_security_min_score else "low_score"
        out_row = {
            **base,
            "wrds_lookup_status": status,
            "wrds_lookup_error": "",
            "wrds_lookup_candidate_count": len(candidates),
            "wrds_match_score": round(float(best_score), 2),
            "wrds_permno": clean_str(best.get("permno")),
            "wrds_ticker": clean_str(best.get("ticker")),
            "wrds_company_name": clean_str(best.get("company_name")),
            "wrds_cusip": normalize_cusip(best.get("cusip")),
            "wrds_name_start_date": clean_str(best.get("name_start_date")),
            "wrds_name_end_date": clean_str(best.get("name_end_date")),
        }
        out_rows.append(out_row)

        if status == "matched":
            s.permno = first_nonempty(s.permno, out_row["wrds_permno"])
            s.cusip = normalize_cusip(first_nonempty(s.cusip, out_row["wrds_cusip"]))
            s.symbol = first_nonempty(s.symbol, out_row["wrds_ticker"])
            s.security_name = first_nonempty(s.security_name, out_row["wrds_company_name"])
            s.security_source = "wrds_security_lookup"
            s.security_note = f"{args.wrds_security_library}.{args.wrds_security_table}; score={out_row['wrds_match_score']}"
            if not s.symbol_source and out_row["wrds_ticker"]:
                s.symbol_source = "wrds_security_lookup"
                s.symbol_note = s.security_note

    return symbols, out_rows


def wrds_security_filter(args: argparse.Namespace, s: EventSymbol, alias: str) -> Tuple[Optional[str], Dict[str, Any], str]:
    params: Dict[str, Any] = {}
    if s.permno and args.wrds_holdings_permno_col:
        params["permno"] = int(float(s.permno))
        return f"{qualified_col(alias, args.wrds_holdings_permno_col)} = %(permno)s", params, "permno"
    if s.cusip and args.wrds_holdings_cusip_col:
        params["cusip8"] = normalize_cusip(s.cusip)[:8]
        return f"substring(upper({qualified_col(alias, args.wrds_holdings_cusip_col)}) from 1 for 8) = %(cusip8)s", params, "cusip"
    if s.symbol and args.wrds_holdings_ticker_col:
        params["ticker"] = s.symbol.upper()
        return f"upper({qualified_col(alias, args.wrds_holdings_ticker_col)}) = %(ticker)s", params, "ticker"
    return None, params, ""


def fetch_wrds_shares_outstanding(client: WrdsClient, args: argparse.Namespace, s: EventSymbol) -> Dict[str, Any]:
    if not s.permno:
        return {"shares_outstanding": None, "shares_outstanding_date": "", "shares_outstanding_error": "missing_permno"}

    table = qualified_table(args.wrds_shares_library, args.wrds_shares_table)
    permno_col = quote_ident(args.wrds_shares_permno_col)
    date_col = quote_ident(args.wrds_shares_date_col)
    shrout_col = quote_ident(args.wrds_shares_shrout_col)
    prc_col = quote_ident(args.wrds_shares_price_col)
    as_of = to_date(s.announce_date) or dt.date.today()
    sql = f"""
        SELECT {date_col} AS price_date, {shrout_col} AS shrout, {prc_col} AS price
        FROM {table}
        WHERE {permno_col} = %(permno)s
          AND {date_col} <= %(as_of_date)s
          AND {date_col} >= %(lookback_start)s
        ORDER BY {date_col} DESC
        LIMIT 1
    """
    params = {
        "permno": int(float(s.permno)),
        "as_of_date": as_of,
        "lookback_start": as_of - dt.timedelta(days=int(args.wrds_shares_lookback_days)),
    }
    try:
        df = client.raw_sql(sql, params=params)
    except Exception as e:
        return {"shares_outstanding": None, "shares_outstanding_date": "", "shares_outstanding_error": str(e)}
    if df.empty:
        return {"shares_outstanding": None, "shares_outstanding_date": "", "shares_outstanding_error": "not_found"}
    row = df.iloc[0]
    shrout = numeric_value(row.get("shrout"))
    shares = None if shrout is None else shrout * float(args.wrds_shrout_multiplier)
    return {
        "shares_outstanding": shares,
        "shares_outstanding_date": clean_str(row.get("price_date")),
        "shares_outstanding_price": row.get("price"),
        "shares_outstanding_error": "",
    }


def fetch_wrds_holdings_for_symbol(
    client: WrdsClient,
    args: argparse.Namespace,
    s: EventSymbol,
    shares_outstanding: Optional[float],
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    base = asdict(s)
    where, id_params, id_kind = wrds_security_filter(args, s, "h")
    if not where:
        err = "missing_symbol_cusip_permno"
        return [], {**base, "provider_endpoint": "wrds", "provider_error": err}

    holdings_table = qualified_table(args.wrds_holdings_library, args.wrds_holdings_table)
    h_date = qualified_col("h", args.wrds_holdings_date_col)
    h2_date = qualified_col("h2", args.wrds_holdings_date_col)
    h_fund_id = qualified_col("h", args.wrds_holdings_fund_id_col)
    h_shares = qualified_col("h", args.wrds_holdings_shares_col)
    h_mktval = qualified_col("h", args.wrds_holdings_market_value_col)
    h_weight = qualified_col("h", args.wrds_holdings_weight_col)
    h_ticker = qualified_col("h", args.wrds_holdings_ticker_col)
    h_cusip = qualified_col("h", args.wrds_holdings_cusip_col)
    h_permno = qualified_col("h", args.wrds_holdings_permno_col)

    h2_where, _, _ = wrds_security_filter(args, s, "h2")
    assert h2_where is not None

    fund_join = ""
    fund_select = "NULL AS fund_name, NULL AS fund_ticker, NULL AS et_flag, NULL AS index_fund_flag"
    if not args.wrds_no_fund_join:
        fund_table = qualified_table(args.wrds_fund_library, args.wrds_fund_table)
        f_fund_id = qualified_col("f", args.wrds_fund_id_col)
        f_name = qualified_col("f", args.wrds_fund_name_col)
        f_ticker = qualified_col("f", args.wrds_fund_ticker_col)
        f_et = qualified_col("f", args.wrds_fund_etf_flag_col)
        f_index = qualified_col("f", args.wrds_fund_index_flag_col)
        fund_join = f"LEFT JOIN {fund_table} f ON {h_fund_id} = {f_fund_id}"
        fund_select = f"{f_name} AS fund_name, {f_ticker} AS fund_ticker, {f_et} AS et_flag, {f_index} AS index_fund_flag"

    as_of = to_date(s.announce_date) or dt.date.today()
    as_of = as_of - dt.timedelta(days=int(args.wrds_availability_lag_days))
    params = {
        **id_params,
        "as_of_date": as_of,
        "lookback_start": as_of - dt.timedelta(days=int(args.wrds_lookback_days)),
        "limit": int(args.wrds_max_rows_per_event),
    }
    sql = f"""
        WITH latest_report AS (
            SELECT MAX({h2_date}) AS report_date
            FROM {holdings_table} h2
            WHERE {h2_where}
              AND {h2_date} <= %(as_of_date)s
              AND {h2_date} >= %(lookback_start)s
        )
        SELECT
            {h_date} AS holdings_report_date,
            {h_fund_id} AS fund_id,
            {h_ticker} AS security_ticker,
            {h_cusip} AS security_cusip,
            {h_permno} AS security_permno,
            {h_shares} AS shares,
            {h_mktval} AS market_value,
            {h_weight} AS portfolio_weight,
            {fund_select}
        FROM {holdings_table} h
        JOIN latest_report lr ON {h_date} = lr.report_date
        {fund_join}
        WHERE {where}
        LIMIT %(limit)s
    """
    try:
        df = client.raw_sql(sql, params=params)
    except Exception as e:
        return [], {
            **base,
            "provider_endpoint": f"wrds:{args.wrds_holdings_library}.{args.wrds_holdings_table}",
            "provider_error": str(e),
            "wrds_identifier_kind": id_kind,
        }

    raw_row_count = len(df)
    dedupe_cols = [
        "holdings_report_date",
        "fund_id",
        "security_permno",
        "security_cusip",
        "security_ticker",
        "shares",
        "market_value",
    ]
    dedupe_cols = [c for c in dedupe_cols if c in df.columns]
    if dedupe_cols:
        df = df.drop_duplicates(subset=dedupe_cols).copy()

    rows: List[Dict[str, Any]] = []
    for _, row in df.iterrows():
        raw = {str(k): (None if pd.isna(v) else v) for k, v in row.to_dict().items()}
        is_passive, reason = classify_wrds_fund(raw)
        if not is_passive and not args.wrds_include_non_etf_funds:
            continue
        shares = numeric_value(raw.get("shares"))
        ownership_percent = None
        if shares is not None and shares_outstanding and shares_outstanding > 0:
            ownership_percent = shares / shares_outstanding
        rows.append({
            **base,
            "provider_endpoint": f"wrds:{args.wrds_holdings_library}.{args.wrds_holdings_table}",
            "provider_error": "",
            "wrds_identifier_kind": id_kind,
            "holdings_report_date": clean_str(raw.get("holdings_report_date")),
            "fund_id": clean_str(raw.get("fund_id")),
            "etf_symbol": clean_str(raw.get("fund_ticker")),
            "etf_name": clean_str(raw.get("fund_name")),
            "fund_et_flag": clean_str(raw.get("et_flag")),
            "fund_index_flag": clean_str(raw.get("index_fund_flag")),
            "fund_classification_reason": reason,
            "security_permno": clean_str(raw.get("security_permno")),
            "security_cusip": normalize_cusip(raw.get("security_cusip")),
            "security_ticker": clean_str(raw.get("security_ticker")),
            "shares_or_exposure": shares,
            "market_value": raw.get("market_value"),
            "weight": raw.get("portfolio_weight"),
            "shares_outstanding": shares_outstanding,
            "ownership_percent": ownership_percent,
            "passive_control_shares": shares,
            "passive_control_percent": ownership_percent,
            "raw_json": json.dumps(raw, ensure_ascii=False, default=str),
        })

    summary = {
        **base,
        "provider_endpoint": f"wrds:{args.wrds_holdings_library}.{args.wrds_holdings_table}",
        "provider_error": "",
        "wrds_identifier_kind": id_kind,
        "as_of_date": str(as_of),
        "wrds_rows_returned": raw_row_count,
        "wrds_unique_holding_rows": len(df),
        "etf_rows_kept": len(rows),
        "shares_outstanding": shares_outstanding,
        "total_etf_shares": sum(float(r.get("shares_or_exposure") or 0) for r in rows),
        "total_etf_ownership_percent": sum(float(r.get("ownership_percent") or 0) for r in rows),
    }
    return rows, summary


def fetch_wrds_for_symbols(
    client: WrdsClient,
    args: argparse.Namespace,
    symbols: List[EventSymbol],
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], List[Dict[str, Any]]]:
    summary_rows: List[Dict[str, Any]] = []
    holder_rows: List[Dict[str, Any]] = []
    etf_holder_rows: List[Dict[str, Any]] = []

    for s in symbols:
        shares_meta = fetch_wrds_shares_outstanding(client, args, s)
        etf_rows, summary = fetch_wrds_holdings_for_symbol(
            client=client,
            args=args,
            s=s,
            shares_outstanding=numeric_value(shares_meta.get("shares_outstanding")),
        )
        summary_rows.append({**summary, **shares_meta})
        etf_holder_rows.extend(etf_rows)
        for row in etf_rows:
            holder_rows.append({
                **row,
                "holder_name": row.get("etf_name", ""),
                "holder_type_raw": "wrds_etf_or_index_fund",
                "holder_category": "etf_or_index_fund",
                "shares": row.get("shares_or_exposure"),
            })

    return summary_rows, holder_rows, etf_holder_rows


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
        e["etf_market_value_num"] = numeric_series(e, "market_value")
        e["etf_ownership_percent_num"] = numeric_series(e, "ownership_percent")
        e["passive_control_percent_num"] = numeric_series(e, "passive_control_percent")
        eagg = (
            e.groupby(["event_id", "side"], dropna=False)
            .agg(
                etf_holder_count=("etf_symbol", "count"),
                etf_shares_or_exposure=("etf_exposure_num", "sum"),
                etf_market_value=("etf_market_value_num", "sum"),
                etf_ownership_percent=("etf_ownership_percent_num", "sum"),
                passive_control_percent=("passive_control_percent_num", "sum"),
            )
            .reset_index()
        )
        parts.append(eagg)

    out = base_df.copy()
    for p in parts:
        out = out.merge(p, on=["event_id", "side"], how="left")
    num_cols = [
        c for c in out.columns
        if c.endswith("_shares")
        or c.endswith("_holder_count")
        or c in {
            "etf_holder_count",
            "etf_shares_or_exposure",
            "etf_market_value",
            "etf_ownership_percent",
            "passive_control_percent",
        }
    ]
    for c in num_cols:
        out[c] = pd.to_numeric(out[c], errors="coerce").fillna(0)
    return out


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Download ownership mix and ETF holder data for M&A events.")
    p.add_argument("--input", required=True, help="Raw Bloomberg M&A CSV or candidate_events.csv from the SEC downloader.")
    p.add_argument("--output-dir", default="ma_ownership_data", help="Output directory.")
    p.add_argument("--provider", choices=["fmp", "wrds"], default="fmp", help="Ownership data provider.")
    p.add_argument("--fmp-api-key", default=None, help="FMP API key. If omitted, FMP_API_KEY is used.")
    p.add_argument("--wrds-username", default=None, help="WRDS username. If omitted, WRDS/.pgpass/PGUSER defaults are used.")
    p.add_argument("--wrds-password", default=None, help="WRDS password. Prefer .pgpass or WRDS_PASSWORD instead of passing this on the command line.")
    p.add_argument("--wrds-password-stdin", action="store_true",
                   help="Read the WRDS password from stdin instead of command-line args or environment variables.")
    p.add_argument("--wrds-security-library", default="crsp", help="WRDS library for security identifier lookup.")
    p.add_argument("--wrds-security-table", default="stocknames", help="WRDS table for security identifier lookup.")
    p.add_argument("--wrds-security-permno-col", default="permno")
    p.add_argument("--wrds-security-ticker-col", default="ticker")
    p.add_argument("--wrds-security-name-col", default="comnam")
    p.add_argument("--wrds-security-cusip-col", default="ncusip")
    p.add_argument("--wrds-security-start-date-col", default="namedt")
    p.add_argument("--wrds-security-end-date-col", default="nameenddt")
    p.add_argument("--wrds-security-candidate-limit", type=int, default=50)
    p.add_argument("--wrds-security-min-score", type=float, default=84.0)
    p.add_argument("--wrds-holdings-library", default="crsp_q_mutualfunds", help="WRDS library containing fund/ETF holdings.")
    p.add_argument("--wrds-holdings-table", default="holdings", help="WRDS holdings table.")
    p.add_argument("--wrds-holdings-date-col", default="report_dt")
    p.add_argument("--wrds-holdings-fund-id-col", default="crsp_portno")
    p.add_argument("--wrds-holdings-permno-col", default="permno")
    p.add_argument("--wrds-holdings-cusip-col", default="cusip")
    p.add_argument("--wrds-holdings-ticker-col", default="ticker")
    p.add_argument("--wrds-holdings-shares-col", default="nbr_shares")
    p.add_argument("--wrds-holdings-market-value-col", default="market_val")
    p.add_argument("--wrds-holdings-weight-col", default="percent_tna")
    p.add_argument("--wrds-fund-library", default="crsp_q_mutualfunds", help="WRDS library containing fund metadata.")
    p.add_argument("--wrds-fund-table", default="fund_hdr", help="WRDS fund metadata table.")
    p.add_argument("--wrds-fund-id-col", default="crsp_portno")
    p.add_argument("--wrds-fund-name-col", default="fund_name")
    p.add_argument("--wrds-fund-ticker-col", default="ticker")
    p.add_argument("--wrds-fund-etf-flag-col", default="et_flag")
    p.add_argument("--wrds-fund-index-flag-col", default="index_fund_flag")
    p.add_argument("--wrds-no-fund-join", action="store_true", help="Skip joining fund metadata; useful for custom holdings tables that already include fund fields.")
    p.add_argument("--wrds-shares-library", default="crsp", help="WRDS library containing shares outstanding data.")
    p.add_argument("--wrds-shares-table", default="dsf", help="WRDS shares-outstanding table.")
    p.add_argument("--wrds-shares-permno-col", default="permno")
    p.add_argument("--wrds-shares-date-col", default="date")
    p.add_argument("--wrds-shares-shrout-col", default="shrout")
    p.add_argument("--wrds-shares-price-col", default="prc")
    p.add_argument("--wrds-shrout-multiplier", type=float, default=1000.0,
                   help="CRSP shrout is normally in thousands; multiply by 1000 to get shares.")
    p.add_argument("--wrds-lookback-days", type=int, default=370,
                   help="Lookback window for latest fund holdings snapshot before the as-of date.")
    p.add_argument("--wrds-shares-lookback-days", type=int, default=30,
                   help="Lookback window for latest shares-outstanding snapshot before the as-of date.")
    p.add_argument("--wrds-availability-lag-days", type=int, default=0,
                   help="Subtract this many days from announcement date when selecting the point-in-time holdings snapshot.")
    p.add_argument("--wrds-max-rows-per-event", type=int, default=20000)
    p.add_argument("--wrds-include-non-etf-funds", action="store_true",
                   help="Keep non-ETF/non-index fund rows in WRDS holdings output instead of filtering them out.")
    p.add_argument("--dry-run", action="store_true", help="Write symbol map and request plan without calling the provider.")
    p.add_argument("--cik-matches", default=None, help="Optional cik_name_matches.csv from download_ma_edgar_files.py for ticker mapping.")
    p.add_argument("--min-cik-match-score", type=float, default=95.0,
                   help="Minimum CIK-name match score accepted for automatic ticker mapping. Historical targets often need overrides.")
    p.add_argument("--symbol-overrides", default=None,
                   help="Optional CSV with event_idx,side,symbol,permno,cusip,note. Overrides SEC/current ticker mapping.")
    p.add_argument("--target-symbol-col", default=TARGET_SYMBOL_COL,
                   help="Optional input CSV column containing the historical target ticker.")
    p.add_argument("--acquirer-symbol-col", default=ACQUIRER_SYMBOL_COL,
                   help="Optional input CSV column containing the acquirer ticker.")
    p.add_argument("--target-permno-col", default=None,
                   help="Optional input CSV column containing the historical target CRSP PERMNO.")
    p.add_argument("--acquirer-permno-col", default=None,
                   help="Optional input CSV column containing the acquirer CRSP PERMNO.")
    p.add_argument("--target-cusip-col", default=TARGET_CUSIP_COL,
                   help="Optional input CSV column containing the target CUSIP.")
    p.add_argument("--acquirer-cusip-col", default=ACQUIRER_CUSIP_COL,
                   help="Optional input CSV column containing the acquirer CUSIP.")
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
        target_permno_col=args.target_permno_col,
        acquirer_permno_col=args.acquirer_permno_col,
        target_cusip_col=args.target_cusip_col,
        acquirer_cusip_col=args.acquirer_cusip_col,
    )

    wrds_client: Optional[WrdsClient] = None
    if args.provider == "wrds" and not args.dry_run:
        wrds_username = args.wrds_username or os.environ.get("WRDS_USERNAME", "")
        if args.wrds_password_stdin:
            wrds_password = read_secret_from_stdin("WRDS password: ")
        else:
            wrds_password = args.wrds_password or os.environ.get("WRDS_PASSWORD", "")
        print(f"[{now_utc()}] Connecting to WRDS")
        wrds_client = WrdsClient(username=wrds_username, password=wrds_password)
        symbols, security_map = resolve_wrds_security_identifiers(wrds_client, symbols, args)
        pd.DataFrame(security_map).to_csv(out_dir / "wrds_security_map.csv", index=False)
        print(f"[{now_utc()}] Wrote: {out_dir / 'wrds_security_map.csv'}")

    symbol_df = pd.DataFrame([asdict(s) for s in symbols])
    symbol_df.to_csv(out_dir / "event_symbol_map.csv", index=False)
    if args.provider == "wrds":
        missing = symbol_df[
            symbol_df["symbol"].astype(str).eq("")
            & symbol_df["permno"].astype(str).eq("")
            & symbol_df["cusip"].astype(str).eq("")
        ]
    else:
        missing = symbol_df[symbol_df["symbol"].astype(str).eq("")]
    missing.to_csv(out_dir / "missing_symbols.csv", index=False)
    override_template = missing[["event_idx", "side", "company_name", "announce_date"]].copy()
    if not override_template.empty:
        override_template["symbol"] = ""
        if args.provider == "wrds":
            override_template["permno"] = ""
            override_template["cusip"] = ""
            override_template["note"] = "fill historical ticker, CRSP PERMNO, or CUSIP"
        else:
            override_template["note"] = "fill historical Bloomberg ticker"
        override_template.to_csv(out_dir / "symbol_overrides_template.csv", index=False)
    print(f"[{now_utc()}] Identifiers mapped: {len(symbol_df) - len(missing):,}/{len(symbol_df):,}")
    if len(missing):
        print(f"[{now_utc()}] Missing identifiers written to: {out_dir / 'missing_symbols.csv'}")
        print(f"[{now_utc()}] Override template written to: {out_dir / 'symbol_overrides_template.csv'}")

    plan = request_plan(symbols, provider=args.provider, args=args)
    pd.DataFrame(plan).to_csv(out_dir / "ownership_request_plan.csv", index=False)

    if args.dry_run:
        print(f"[{now_utc()}] Dry run; wrote identifier map and request plan only.")
        return

    if args.provider == "fmp":
        api_key = args.fmp_api_key or os.environ.get("FMP_API_KEY", "")
        if not api_key:
            print(f"[{now_utc()}] Missing FMP API key; wrote request plan only.")
            print(f"[{now_utc()}] Set FMP_API_KEY or pass --fmp-api-key to download ownership/ETF data.")
            return
        client = FmpClient(api_key=api_key, cache_dir=cache_dir, sleep_seconds=args.sleep_seconds)
        inst_summary, inst_holders, etf_holders = fetch_for_symbols(client, symbols)
    elif args.provider == "wrds":
        if wrds_client is None:
            wrds_username = args.wrds_username or os.environ.get("WRDS_USERNAME", "")
            if args.wrds_password_stdin:
                wrds_password = read_secret_from_stdin("WRDS password: ")
            else:
                wrds_password = args.wrds_password or os.environ.get("WRDS_PASSWORD", "")
            wrds_client = WrdsClient(username=wrds_username, password=wrds_password)
        try:
            inst_summary, inst_holders, etf_holders = fetch_wrds_for_symbols(wrds_client, args, symbols)
        finally:
            wrds_client.close()
    else:
        raise ValueError(f"Unsupported provider: {args.provider}")

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
