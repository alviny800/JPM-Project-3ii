#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
build_cik_overrides.py

Find + VERIFY correct CIKs for the recoverable tail of the CIK-resolution audit
(unmatched-but-has-CRSP-close, and matched-but-score<95 false-positive risk).

For each target it (1) re-queries EDGAR full-text search with a CLEANED name
(strips "/old", "/The", "/Durham NC" geo tags, entity suffixes) to get candidate
CIKs, then (2) VERIFIES each candidate by pulling its EDGAR submissions history and
checking whether it filed a merger-type form (8-K/425/DEFM14A/15-12B/25) within a
window of the target's CRSP close date. A candidate that filed a merger doc right at
close is almost certainly the real target.

Cost $0 (EDGAR only). Writes cik_override_candidates.csv for human review; the final
hand-picked cik_manual_overrides.csv is assembled from it.
"""
from __future__ import annotations

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


def main() -> None:
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


if __name__ == "__main__":
    main()
