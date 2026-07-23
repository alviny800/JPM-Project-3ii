#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""cik_resolution.py — CIK resolution tooling (EDGAR full-text search, $0).

Subcommands:
  audit            run the name->CIK matcher across the universe -> cik_resolution_audit.csv
  build-overrides  verify CIKs for the recoverable tail          -> cik_override_candidates.csv

Merges the former audit_cik_resolution.py + build_cik_overrides.py; logic is verbatim.
Typical order:  cik_resolution.py audit    then    cik_resolution.py build-overrides
"""
from __future__ import annotations
import sys

import argparse
from pathlib import Path

import pandas as pd

import download_ma_edgar_files as dl


def main_audit() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--input", default="US_election_deals_for_analysis.csv", type=Path)
    p.add_argument("--close-dates", default="target_close_dates.csv", type=Path)
    p.add_argument("--out", default="cik_resolution_audit.csv", type=Path)
    p.add_argument("--user-agent", default="election-arb-research alviny800@gmail.com")
    p.add_argument("--min-name-score", type=float, default=84.0)
    p.add_argument("--cache-dir", default="ma_edgar_audit_cache", type=Path)
    args = p.parse_args()

    df = pd.read_csv(args.input, dtype=str, keep_default_na=False)

    # Close-date presence = the downstream safety net; note it per target.
    close_by_c8: dict = {}
    if args.close_dates.exists():
        cd = pd.read_csv(args.close_dates, dtype=str, keep_default_na=False)
        for _, r in cd.iterrows():
            if r.get("close_date"):
                close_by_c8[r.get("target_cusip8", "")] = r["close_date"]

    def cusip8(v: str) -> str:
        import re
        return re.sub(r"[^0-9A-Za-z]", "", str(v)).upper()[:8]

    client = dl.SecClient(user_agent=args.user_agent, cache_dir=args.cache_dir)
    cik_df = dl.load_company_tickers(client, args.cache_dir / "company_tickers.json")
    print(f"[load] company_tickers.json: {len(cik_df):,} current registrants")
    n_ovr = dl.load_cik_overrides(Path("cik_manual_overrides.csv"))
    print(f"[load] cik_manual_overrides.csv: {n_ovr} hand-verified overrides")
    print(f"[run] resolving {len(df):,} target names via efts (primary) + fuzzy (fallback)...")

    rows = []
    for i, ev in df.iterrows():
        name = str(ev.get("Target Name", "")).strip()
        m = dl.override_match(name)
        if m is not None:
            method = "manual_override"
        else:
            m = dl.resolve_cik_via_efts(client, name, min_score=args.min_name_score)
            method = "efts"
        if not m.get("matched"):
            mt = dl.fuzzy_match_company(name, cik_df, min_score=args.min_name_score)
            if mt.get("matched") or float(mt.get("score", 0) or 0) > float(m.get("score", 0) or 0):
                m = mt
                method = "fuzzy_fallback"
        c8 = cusip8(ev.get("Target cusip", ""))
        rows.append({
            "target_name": name,
            "matched": bool(m.get("matched")),
            "score": float(m.get("score", 0) or 0),
            "resolved_via": method if m.get("matched") else "NONE",
            "sec_title": m.get("sec_title", ""),
            "cik10": m.get("cik10", ""),
            "ticker": m.get("ticker", ""),
            "target_cusip": ev.get("Target cusip", ""),
            "has_crsp_close": c8 in close_by_c8,
            "crsp_close": close_by_c8.get(c8, ""),
        })
        if (i + 1) % 25 == 0:
            print(f"  ...{i + 1}/{len(df)}")

    out = pd.DataFrame(rows).sort_values(["matched", "score"], ascending=[True, True])
    out.to_csv(args.out, index=False)

    n = len(out)
    n_match = int(out["matched"].sum())
    n_efts = int((out["resolved_via"] == "efts").sum())
    n_fuzzy = int((out["resolved_via"] == "fuzzy_fallback").sum())
    n_none = int((out["resolved_via"] == "NONE").sum())
    # confidence tiers among matched
    md = out[out["matched"]]
    strong = int((md["score"] >= 95).sum())
    ok = int(((md["score"] >= 88) & (md["score"] < 95)).sum())
    weak = int((md["score"] < 88).sum())
    # safety net: matched but low score AND no crsp close = highest risk
    risk = out[(~out["matched"]) | ((out["score"] < 90) & (~out["has_crsp_close"]))]

    print(f"\n=== CIK RESOLUTION AUDIT ({n} targets) ===")
    print(f"  matched:            {n_match}/{n} ({n_match/n*100:.0f}%)")
    print(f"    via efts:         {n_efts}")
    print(f"    via fuzzy fallb.: {n_fuzzy}")
    print(f"  UNMATCHED:          {n_none}")
    print(f"\n  confidence of matched (by name score):")
    print(f"    strong (>=95):    {strong}")
    print(f"    ok     (88-95):   {ok}")
    print(f"    weak   (<88):     {weak}")
    print(f"\n  {len(risk)} targets in the review tail (unmatched, or score<90 w/ no CRSP close-date net)")
    print(f"\n[write] {args.out}  (sorted worst-first)")
    print("\nWorst 15:")
    cols = ["target_name", "matched", "score", "resolved_via", "sec_title", "has_crsp_close"]
    print(out[cols].head(15).to_string(index=False))

import argparse
import re
import time
from pathlib import Path
from typing import Any, Dict, List

import pandas as pd
import requests

import download_ma_edgar_files as dl

MERGER_FORMS = ("8-K", "8-K12B", "425", "DEFM14A", "DEFA14A", "15-12B", "15-12G", "25", "25-NSE", "8-K12G3")


def clean_name(name: str) -> str:
    """Drop Bloomberg '/old', '/The', '/Durham NC' style tags after a slash."""
    n = name.split("/")[0].strip()
    return n


def strip_entity(name: str) -> str:
    n = re.sub(r"\b(Inc|Incorporated|Corp|Corporation|Co|Company|LLC|LP|L\.?P\.?|NA|N\.?A\.?|Bancorp|Bankshares|Holdings?|Group)\b\.?",
               " ", name, flags=re.I)
    n = re.sub(r"\ba Delaware (LP|Limited Partnership)\b", " ", n, flags=re.I)
    return re.sub(r"\s+", " ", n).strip()


def efts_candidates(client: dl.SecClient, query: str) -> Dict[str, Dict[str, Any]]:
    """Return {cik10: {name, ticker, freq}} for an efts exact-phrase search."""
    ua = client.session.headers.get("User-Agent", "")
    cand: Dict[str, Dict[str, Any]] = {}
    try:
        time.sleep(getattr(client, "sleep_seconds", 0.13))
        from urllib.parse import quote
        r = requests.get(dl.SEC_EFTS_URL + '?q=%22' + quote(query) + '%22',
                         headers={"User-Agent": ua, "Accept-Encoding": "gzip, deflate"}, timeout=30)
        r.raise_for_status()
        hits = r.json().get("hits", {}).get("hits", [])
    except Exception:
        return cand
    for h in hits:
        for dn in h.get("_source", {}).get("display_names", []):
            m = re.search(r"\(CIK (\d{10})\)", dn)
            if not m:
                continue
            cik10 = m.group(1)
            disp = dn.split("  (")[0].strip()
            tm = re.search(r"\(([A-Z][A-Z0-9.\-]{0,6})\)\s*\(CIK", dn)
            e = cand.setdefault(cik10, {"name": disp, "ticker": tm.group(1) if tm else "", "freq": 0})
            e["freq"] += 1
    return cand


def verify_close(client: dl.SecClient, cik10: str, close: pd.Timestamp, window_days: int = 150) -> Dict[str, Any]:
    """Pull submissions; did this CIK file a merger form within +/- window of close?"""
    out = {"filed_merger_near_close": False, "near_form": "", "near_date": "", "name": "", "sic": "", "n_filings": 0}
    if not close or pd.isna(close):
        return out
    try:
        js = client.get_json(dl.SEC_SUBMISSIONS_URL.format(cik10=cik10))
    except Exception:
        return out
    out["name"] = js.get("name", "")
    out["sic"] = f'{js.get("sicDescription","")}'
    recent = js.get("filings", {}).get("recent", {})
    forms = recent.get("form", []) or []
    dates = recent.get("filingDate", []) or []
    out["n_filings"] = len(forms)
    best = None
    for f, d in zip(forms, dates):
        try:
            fd = pd.to_datetime(d)
        except Exception:
            continue
        delta = abs((fd - close).days)
        if delta <= window_days and (f in MERGER_FORMS or f.startswith("8-K") or f.startswith("15") or f.startswith("25")):
            if best is None or delta < best[0]:
                best = (delta, f, d)
    if best:
        out.update(filed_merger_near_close=True, near_form=best[1], near_date=best[2])
    return out


def main_build_overrides() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--audit", default="cik_resolution_audit.csv", type=Path)
    p.add_argument("--out", default="cik_override_candidates.csv", type=Path)
    p.add_argument("--user-agent", default="election-arb-research alviny800@gmail.com")
    p.add_argument("--cache-dir", default="ma_edgar_audit_cache", type=Path)
    args = p.parse_args()

    d = pd.read_csv(args.audit)
    # recoverable tail: unmatched WITH a crsp close, plus matched-but-weak (<95)
    tail = d[((~d.matched) & d.has_crsp_close) | (d.matched & (d.score < 95))].copy()
    print(f"[run] verifying {len(tail)} recoverable targets (efts + submissions-history check)\n")

    client = dl.SecClient(user_agent=args.user_agent, cache_dir=args.cache_dir)

    rows: List[Dict[str, Any]] = []
    for _, r in tail.iterrows():
        name = r["target_name"]
        close = pd.to_datetime(r["crsp_close"]) if str(r.get("crsp_close", "")) else pd.NaT
        # gather candidates from cleaned + entity-stripped queries
        cands: Dict[str, Dict[str, Any]] = {}
        for q in {clean_name(name), strip_entity(clean_name(name))}:
            if not q:
                continue
            for cik, e in efts_candidates(client, q).items():
                c = cands.setdefault(cik, dict(e, freq=0))
                c["freq"] += e["freq"]
        # score each candidate name vs the cleaned target, verify against close date
        norm_t = dl.normalize_name(clean_name(name))
        scored = []
        for cik, e in cands.items():
            from rapidfuzz import fuzz
            s = float(fuzz.token_set_ratio(norm_t, dl.normalize_name(e["name"])))
            scored.append((s, cik, e))
        scored.sort(key=lambda x: (-x[0], -x[2]["freq"]))
        # verify top 3 candidates against the close date
        picked = None
        for s, cik, e in scored[:3]:
            v = verify_close(client, cik, close)
            if v["filed_merger_near_close"]:
                picked = (s, cik, e, v)
                break
        if picked is None and scored:
            s, cik, e = scored[0]
            picked = (s, cik, e, verify_close(client, cik, close))
        if picked:
            s, cik, e, v = picked
            rows.append({
                "target_name": name,
                "old_score": r["score"],
                "old_sec_title": r.get("sec_title", ""),
                "cand_cik10": cik,
                "cand_name": e["name"],
                "cand_ticker": e["ticker"],
                "cand_name_score": round(s, 1),
                "verified_name": v["name"],
                "sic": v["sic"],
                "filed_merger_near_close": v["filed_merger_near_close"],
                "near_form": v["near_form"],
                "near_date": v["near_date"],
                "crsp_close": r.get("crsp_close", ""),
                "target_cusip": r.get("target_cusip", ""),
            })
            flag = "OK " if v["filed_merger_near_close"] else "?? "
            print(f"{flag}{name[:34]:34s} -> CIK {cik} {v['name'][:30]:30s} "
                  f"score {s:4.0f} near={v['near_form']}@{v['near_date']}")
        else:
            rows.append({"target_name": name, "old_score": r["score"], "cand_cik10": "",
                         "cand_name": "", "filed_merger_near_close": False, "crsp_close": r.get("crsp_close", "")})
            print(f"XX {name[:34]:34s} -> NO candidate")

    out = pd.DataFrame(rows)
    out.to_csv(args.out, index=False)
    n_ok = int(out["filed_merger_near_close"].sum())
    print(f"\n[write] {args.out}: {len(out)} targets, {n_ok} verified by a merger filing near close")


def main():
    if len(sys.argv) < 2 or sys.argv[1] not in ("audit", "build-overrides",):
        print("usage: {} {{audit | build-overrides}} [options]".format(sys.argv[0].split("/")[-1]))
        sys.exit(2)
    sub = sys.argv.pop(1)
    if sub == "audit":
        main_audit()
    elif sub == "build-overrides":
        main_build_overrides()


if __name__ == "__main__":
    main()
