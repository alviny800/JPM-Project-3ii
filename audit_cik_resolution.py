#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
audit_cik_resolution.py

Resolution-only audit of the EDGAR CIK matcher — NO filing downloads, NO Claude calls.

Runs the exact same name->CIK resolution the Stage-2 pipeline uses (efts full-text
search primary, company_tickers.json fuzzy fallback) across every TARGET in the
analysis universe, and writes a score-sorted table so we can eyeball the low-confidence
tail (generic bank/financial names) and flag any target that needs a manual CIK.

Cost: $0 (efts + one small SEC json download for company_tickers).

Output: cik_resolution_audit.csv (sorted by score ascending — worst matches first).
"""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

import download_ma_edgar_files as dl


def main() -> None:
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


if __name__ == "__main__":
    main()
