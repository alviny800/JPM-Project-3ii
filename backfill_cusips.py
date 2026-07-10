#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
backfill_cusips.py

Re-resolve US-listed security CUSIPs in the BBG M&A pull from CRSP `stocknames`,
because the exported CSV's CUSIP column is unreliable: some are missing (N.A.)
and many "present" values were silently corrupted by Excel on export
(scientific notation like `8.81E+104`, or leading-zero truncation like `7903107`
for a CUSIP that begins with 0).

Strategy (per supervisor decision): treat the BBG CUSIP column as untrusted and
re-resolve EVERY US-listed security (target + acquirer) fresh from CRSP as-of the
announce date. Keep the BBG value only as a cross-check and flag mismatches.

CRSP is the right source here because most targets are delisted post-merger;
current-quote APIs (OpenFIGI etc.) don't retain delisted names, and OpenFIGI
does not return CUSIP anyway. `stocknames` keeps delisted securities with their
historical CUSIP and the date range each name/identifier was in effect.

Reuses the existing WRDS plumbing from download_ownership_etf_data.py so there is
exactly one connection/identifier code path in the repo.

Outputs
-------
- <out>/BBG_with_resolved_cusips.csv   — original columns + resolved CUSIP cols
- <out>/cusip_backfill_audit.csv       — one row per US security resolved

Requires: pandas, wrds (+ a matching ~/.pgpass entry or WRDS_PASSWORD).
"""

from __future__ import annotations

import argparse
import datetime as dt
import re
import sys
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd

from download_ownership_etf_data import (
    WrdsClient,
    clean_str,
    company_query_token,
    name_match_score,
    normalize_cusip,
    to_date,
)

US_SUFFIX_RE = re.compile(r"\s+US$")
# Bloomberg placeholder id for a delisted/unlisted security, e.g. "1436513D US",
# "8206908Q US". These are NOT real tickers and will not match crsp.stocknames.ticker.
BBG_PLACEHOLDER_RE = re.compile(r"^\d{6,7}[A-Z]$")


def is_us_ticker(t: str) -> bool:
    return bool(US_SUFFIX_RE.search(clean_str(t)))


def clean_ticker(t: str) -> str:
    """'BRCM US' -> 'BRCM'. Returns '' if not US-listed."""
    t = clean_str(t)
    if not US_SUFFIX_RE.search(t):
        return ""
    return US_SUFFIX_RE.sub("", t).strip().upper()


def cusip_check_digit(cusip8: str) -> str:
    """Standard CUSIP mod-10 check digit for an 8-char base."""
    total = 0
    for i, ch in enumerate(cusip8[:8]):
        if ch.isdigit():
            v = int(ch)
        elif ch.isalpha():
            v = ord(ch.upper()) - ord("A") + 10
        elif ch == "*":
            v = 36
        elif ch == "@":
            v = 37
        elif ch == "#":
            v = 38
        else:
            return ""  # can't compute
        if i % 2 == 1:  # even position (1-indexed) doubles
            v *= 2
        total += v // 10 + v % 10
    return str((10 - (total % 10)) % 10)


def to_cusip9(cusip: str) -> str:
    """Normalize a CRSP ncusip (usually 8-char) to a 9-char CUSIP with check digit."""
    c = normalize_cusip(cusip)
    if len(c) == 9:
        return c
    if len(c) == 8:
        cd = cusip_check_digit(c)
        return c + cd if cd else c
    return c


def classify_bbg_cusip(raw: str) -> str:
    """'ok' | 'short_suspect' | 'missing' | 'malformed'.

    A canonical CUSIP is 9 chars (8-char base + check digit) and Bloomberg emits
    9. An 8-char value is almost always a 9-digit CUSIP whose leading zero Excel
    dropped (e.g. Berkshire '84670108' should be '084670108'), so we treat it as
    suspect rather than trustworthy.
    """
    v = clean_str(raw)
    if v == "" or v.upper() in {"N.A.", "N/A", "."}:
        return "missing"
    if re.fullmatch(r"[0-9A-Za-z]{9}", v):
        return "ok"
    if re.fullmatch(r"[0-9A-Za-z]{8}", v):
        return "short_suspect"
    return "malformed"  # scientific notation, truncated to <8, etc.


def cusip_equiv(bbg_raw: str, crsp9: str) -> bool:
    """True if a BBG CUSIP and a CRSP 9-char CUSIP are the same security,
    tolerating Excel's dropped leading zero and a missing check digit."""
    b = re.sub(r"[^0-9A-Za-z]", "", clean_str(bbg_raw)).upper()
    c = re.sub(r"[^0-9A-Za-z]", "", clean_str(crsp9)).upper()
    if not b or not c:
        return False
    if b == c:
        return True
    if len(b) == 8 and ("0" + b == c or b == c[:8]):  # dropped leading zero / no check digit
        return True
    if len(b) == 9 and b[:8] == c[:8]:                # same base, differing check digit
        return True
    return False


# ---------------------------------------------------------------------------
# Collect the US securities we need to resolve
# ---------------------------------------------------------------------------

def collect_securities(df: pd.DataFrame, election_only: bool) -> pd.DataFrame:
    rows: List[Dict] = []
    src = df
    if election_only and "Payment Type" in df.columns:
        src = df[df["Payment Type"].astype(str) == "Cash or Stock"]
    for idx, r in src.iterrows():
        announce = clean_str(r.get("Announce Date"))
        for side, name_col, tkr_col, cus_col in [
            ("target", "Target Name", "Target Ticker", "Target cusip"),
            ("acquirer", "Acquirer Name", "Acquirer Ticker", "Acquirer cusip"),
        ]:
            raw_tkr = clean_str(r.get(tkr_col))
            if not is_us_ticker(raw_tkr):
                continue
            rows.append({
                "row_idx": idx,
                "side": side,
                "company_name": clean_str(r.get(name_col)),
                "bbg_ticker": raw_tkr,
                "ticker": clean_ticker(raw_tkr),
                "announce_date": announce,
                "bbg_cusip": clean_str(r.get(cus_col)),
                "bbg_cusip_status": classify_bbg_cusip(r.get(cus_col)),
            })
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# CRSP stocknames lookups
# ---------------------------------------------------------------------------

def dump_stocknames(cache_path: Path, username: str = "") -> None:
    """Connect to WRDS, pull all of crsp.stocknames, write to disk, exit.

    Runs as its own short-lived process: doing heavy pandas work while a
    psycopg2/SQLAlchemy WRDS connection is held open segfaults on macOS, so we
    keep the DB session to just query+write and do all matching in a fresh
    process that never imports wrds.
    """
    client = WrdsClient(username=username)
    try:
        df = client.raw_sql(
            "SELECT permno, ticker, comnam, ncusip, cusip, namedt, nameenddt "
            "FROM crsp.stocknames"
        )
    finally:
        client.close()
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(cache_path, index=False)
    print(f"[dump] wrote {len(df):,} stocknames rows -> {cache_path}", file=sys.stderr)


def build_ticker_index(sn: pd.DataFrame) -> Dict[str, List[dict]]:
    out: Dict[str, List[dict]] = {}
    for row in sn.to_dict("records"):
        out.setdefault(row["ticker_u"], []).append(row)
    return out


def pick_name_row(cands: List[dict], company_name: str, announce: Optional[object]) -> Optional[dict]:
    """Choose the best stocknames row for this security.

    Ranking, in order:
      1. name match score  — disambiguates ticker reuse (same ticker, different
         companies over time); the row whose company name matches wins.
      2. in-window         — the row whose [namedt, nameenddt] covers the announce
         date, i.e. the CUSIP in effect at deal time.
      3. recency           — when the announce date is outside CRSP's coverage
         (e.g. a 2026 deal), fall back to the most recent row, not an arbitrary
         old one (that bug picked stale CUSIPs for GameStop/Amazon-type names).
    """
    if not cands:
        return None
    scored = []
    for row in cands:
        s_dt = to_date(row.get("namedt"))
        e_dt = to_date(row.get("nameenddt"))
        in_window = (
            (s_dt is None or (announce and announce >= s_dt))
            and (e_dt is None or (announce and announce <= e_dt))
        )
        score = name_match_score(company_name, clean_str(row.get("comnam")))
        recency = (e_dt or dt.date.max).toordinal()
        scored.append((round(score, 1), bool(in_window), recency, row))
    scored.sort(key=lambda x: (x[0], x[1], x[2]), reverse=True)
    return scored[0][3]


def name_fallback_lookup(sn: pd.DataFrame, company_name: str, announce) -> Optional[dict]:
    """For Bloomberg placeholders / unmatched tickers: match by company name,
    entirely in memory against the pre-loaded stocknames frame."""
    token = company_query_token(company_name)
    if not token:
        return None
    cand = sn[sn["comnam_l"].str.contains(re.escape(token.lower()), na=False)]
    if cand.empty:
        return None
    return pick_name_row(cand.to_dict("records"), company_name, announce)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

MIN_NAME_SCORE = 84.0  # matches wrds-security-min-score default in the repo


def resolve(sn: pd.DataFrame, secs: pd.DataFrame) -> pd.DataFrame:
    sn = sn.copy()
    sn["ticker_u"] = sn["ticker"].astype(str).str.strip().str.upper()
    sn["comnam_l"] = sn["comnam"].astype(str).str.lower()
    ticker_map = build_ticker_index(sn)

    results = []
    for _, s in secs.iterrows():
        announce = to_date(s["announce_date"])
        name = s["company_name"]
        row = None
        source = ""
        if s["ticker"] and not BBG_PLACEHOLDER_RE.match(s["ticker"]):
            row = pick_name_row(ticker_map.get(s["ticker"], []), name, announce)
            if row is not None:
                source = "crsp_ticker"
        if row is None:  # placeholder or ticker miss -> name fallback
            row = name_fallback_lookup(sn, name, announce)
            if row is not None:
                source = "crsp_name"

        resolved9 = ""
        score = 0.0
        permno = ""
        crsp_name = ""
        if row is not None:
            resolved9 = to_cusip9(row.get("ncusip") or row.get("cusip"))
            score = name_match_score(name, clean_str(row.get("comnam")))
            permno = clean_str(row.get("permno"))
            crsp_name = clean_str(row.get("comnam"))

        # A CRSP value is only trusted if the matched company name scores well;
        # a bare ticker hit with a poor name match is a ticker-reuse false match.
        crsp_ok = bool(resolved9) and score >= MIN_NAME_SCORE

        bbg_status = s["bbg_cusip_status"]
        bbg_raw = s["bbg_cusip"]
        equiv = crsp_ok and cusip_equiv(bbg_raw, resolved9)

        # --- Reconciliation policy ---
        # Backfill genuine gaps from CRSP; keep a valid BBG value otherwise; only
        # auto-apply CRSP over a present BBG value when it's the same issuer
        # (Excel leading-zero / check-digit corruption). Flag true conflicts.
        if bbg_status in ("missing", "malformed"):
            if crsp_ok:
                final, final_src, decision = resolved9, "crsp", "backfilled"
            else:
                final, final_src, decision = "", "", "still_missing"
        elif bbg_status == "short_suspect":       # 8-char, likely dropped leading zero
            if equiv:
                final, final_src, decision = resolved9, "crsp", "format_fixed"
            elif crsp_ok:
                final, final_src, decision = resolved9, "crsp", "short_replaced"
            else:
                final, final_src, decision = bbg_raw, "bbg", "kept_bbg_short"
        else:                                     # bbg 'ok' (9-char)
            if not crsp_ok:
                final, final_src, decision = bbg_raw, "bbg", "kept_bbg_uncovered"
            elif equiv:
                final, final_src, decision = bbg_raw, "bbg", "confirmed"
            else:
                final, final_src, decision = bbg_raw, "bbg", "conflict_review"

        xcheck = "equiv" if equiv else ("mismatch" if (crsp_ok and bbg_status != "missing") else "")
        needs_review = decision in ("conflict_review", "still_missing", "kept_bbg_short")

        results.append({
            **s.to_dict(),
            "resolved_cusip": resolved9,
            "resolve_source": source,
            "crsp_permno": permno,
            "crsp_name": crsp_name,
            "name_score": round(float(score), 1),
            "bbg_vs_crsp": xcheck,
            "final_cusip": final,
            "final_source": final_src,
            "decision": decision,
            "needs_review": needs_review,
        })
    return pd.DataFrame(results)


def write_back(df_orig: pd.DataFrame, audit: pd.DataFrame, out_dir: Path) -> None:
    out = df_orig.copy()
    for col in ["Target", "Acquirer"]:
        out[f"{col} cusip_final"] = ""
        out[f"{col} cusip_decision"] = ""
        out[f"{col} cusip_crsp"] = ""
    for _, a in audit.iterrows():
        col = "Target" if a["side"] == "target" else "Acquirer"
        out.at[a["row_idx"], f"{col} cusip_final"] = a["final_cusip"]
        out.at[a["row_idx"], f"{col} cusip_decision"] = a["decision"]
        out.at[a["row_idx"], f"{col} cusip_crsp"] = a["resolved_cusip"]

    out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = out_dir / "BBG_with_resolved_cusips.csv"
    out.to_csv(csv_path, index=False)
    audit_path = out_dir / "cusip_backfill_audit.csv"
    audit.to_csv(audit_path, index=False)
    review_path = out_dir / "cusip_needs_review.csv"
    audit[audit["needs_review"]].to_csv(review_path, index=False)
    print(f"[write] {csv_path}", file=sys.stderr)
    print(f"[write] {audit_path}", file=sys.stderr)
    print(f"[write] {review_path}", file=sys.stderr)


def summarize(audit: pd.DataFrame) -> None:
    print("\n=== CUSIP backfill summary ===", file=sys.stderr)
    print(f"US securities processed: {len(audit)}", file=sys.stderr)
    print("\nBBG value quality (before):", file=sys.stderr)
    print(audit["bbg_cusip_status"].value_counts().to_string(), file=sys.stderr)
    print("\nReconciliation decision (after):", file=sys.stderr)
    print(audit["decision"].value_counts().to_string(), file=sys.stderr)

    filled = audit[audit["decision"].isin(["backfilled", "format_fixed", "short_replaced"])]
    print(f"\nGap/corruption fixed from CRSP: {len(filled)}", file=sys.stderr)
    unresolved_final = audit[audit["final_cusip"] == ""]
    print(f"Still without a usable CUSIP: {len(unresolved_final)}", file=sys.stderr)

    review = audit[audit["needs_review"]]
    print(f"\n⚠ {len(review)} need manual review "
          f"({audit['decision'].value_counts().get('conflict_review', 0)} conflicts, "
          f"{audit['decision'].value_counts().get('still_missing', 0)} unresolved, "
          f"{audit['decision'].value_counts().get('kept_bbg_short', 0)} short-uncovered):",
          file=sys.stderr)
    conf = audit[audit["decision"] == "conflict_review"]
    if not conf.empty:
        print("\nConflicts (BBG kept; CRSP disagrees on a covered, well-named match):", file=sys.stderr)
        print(conf[["company_name", "bbg_ticker", "announce_date", "bbg_cusip",
                    "resolved_cusip", "name_score"]].head(30).to_string(index=False), file=sys.stderr)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--input", default="BBG Data Pull 2006+ Final.csv", type=Path)
    p.add_argument("--out-dir", default=Path("cusip_backfill"), type=Path)
    p.add_argument("--election-only", action="store_true",
                   help="Only process Payment Type == 'Cash or Stock' deals.")
    p.add_argument("--stocknames-cache", default=Path("stocknames_cache.csv"), type=Path,
                   help="Local CSV of crsp.stocknames. Populate it with --dump-stocknames.")
    p.add_argument("--dump-stocknames", action="store_true",
                   help="Only connect to WRDS, dump crsp.stocknames to the cache, and exit.")
    p.add_argument("--wrds-username", default="")
    return p.parse_args()


def main() -> None:
    args = parse_args()

    # Phase 1 (own process): pull stocknames from WRDS and exit.
    if args.dump_stocknames:
        dump_stocknames(args.stocknames_cache, args.wrds_username)
        return

    # Phase 2 (fresh process, no wrds import): match entirely offline.
    if not args.stocknames_cache.exists():
        sys.exit(f"[error] {args.stocknames_cache} not found. Run first:\n"
                 f"    python3 backfill_cusips.py --dump-stocknames")
    sn = pd.read_csv(args.stocknames_cache, dtype=str, keep_default_na=False)
    print(f"[load] {len(sn):,} stocknames rows from {args.stocknames_cache}", file=sys.stderr)

    df = pd.read_csv(args.input, dtype=str, keep_default_na=False)
    print(f"[load] {len(df)} deals from {args.input}", file=sys.stderr)

    secs = collect_securities(df, args.election_only)
    print(f"[collect] {len(secs)} US-listed securities to resolve "
          f"({secs['side'].value_counts().to_dict()})", file=sys.stderr)

    audit = resolve(sn, secs)
    write_back(df, audit, args.out_dir)
    summarize(audit)


if __name__ == "__main__":
    main()
