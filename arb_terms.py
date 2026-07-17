#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
arb_terms.py  —  DATA LAYER for the election-arb Monte Carlo / backtest framework.

Assembles ONE clean deal table (arb_deals.csv) that every downstream module reads.
Per completed cash-or-stock election deal we resolve the structural terms the MC needs:

  C        cash consideration per share            ($/sh)          -> cash election value
  R        exchange ratio (fixed only)             (acq sh / tgt)  -> stock leg
  P_acq    acquirer price at the election deadline ($/sh)          -> stock election value = R * P_acq
  stock_val= R * P_acq                             ($/sh)
  spread   = C - stock_val                         ($/sh)          -> the deadline election spread
  pi_cash  aggregate CASH proration target         (frac 0-1)      -> the fixed cash pool (from cash_cap)
  f_cash   REALIZED fraction of shares electing cash(frac 0-1)     -> the stochastic outcome we model
  ratio_type  fixed | floating                                     -> floating excluded from spread work

Nothing here is stochastic — this is just the observed, cleaned deal terms.
Reuses deadline_spread.csv (already has C, R, P_acq, spread, ratio_type, realized demand).
"""
from __future__ import annotations
import re
import numpy as np
import pandas as pd


def first_num(s, lo, hi):
    """First number in text within [lo,hi] — skips share counts / years."""
    if pd.isna(s):
        return np.nan
    for tok in re.findall(r"\d+\.?\d*", str(s).replace(",", "")):
        v = float(tok)
        if lo <= v <= hi:
            return v
    return np.nan


def parse_pct_frac(s):
    """A percentage in prose -> fraction in (0,1). Skips share counts. e.g. '50%' -> 0.50"""
    if pd.isna(s):
        return np.nan
    m = re.search(r"(\d{1,3}(?:\.\d+)?)\s*%", str(s).replace(",", ""))
    if m:
        v = float(m.group(1))
        return v / 100.0 if 1 <= v <= 99 else np.nan
    return np.nan


def build_deals() -> pd.DataFrame:
    ds = pd.read_csv("deadline_spread.csv")          # C, R, P_acq, spread, ratio_type, realized demand
    ext = pd.read_csv("ma_edgar_full/llm_field_extractions.csv")
    w = ext[ext.event_id.notna()].pivot_table(index="event_id", columns="field_name",
                                               values="value", aggfunc="first")
    norm = pd.read_csv("normalized_labels.csv")

    d = ds.rename(columns={"cash_val": "C", "ratio": "R", "acq_price_deadline": "P_acq",
                           "spread_deadline": "spread", "realized_cash_share": "f_cash"}).copy()

    # aggregate cash proration target pi_cash: prefer cash_cap %, else 1 - stock_cap %, else 0.50 (the
    # modal 50/50 election structure) with a source flag so coverage is auditable.
    def pi_for(eid):
        if eid in w.index:
            p = parse_pct_frac(w.loc[eid].get("cash_cap"))
            if not np.isnan(p):
                return p, "cash_cap"
            ps = parse_pct_frac(w.loc[eid].get("stock_cap"))
            if not np.isnan(ps):
                return 1 - ps, "stock_cap"
        return 0.50, "default_5050"
    pis = d["event_id"].map(lambda e: pi_for(e))
    d["pi_cash"] = pis.map(lambda t: t[0])
    d["pi_cash_source"] = pis.map(lambda t: t[1])

    # realized election demand as a fraction (from the normalized labels, authoritative)
    fc = norm.set_index("event_id")["pct_elected_cash"].apply(pd.to_numeric, errors="coerce") / 100.0
    d["f_cash"] = d["event_id"].map(fc)

    # deal break flag (for survivorship-aware P&L backtest later)
    brk = w["deal_completion_or_break"] if "deal_completion_or_break" in w.columns else pd.Series(dtype=str)
    d["broke"] = d["event_id"].map(lambda e: bool(re.search(r"break|terminat|withdraw|fail",
                                    str(brk.get(e, "")), re.I)) if e in getattr(brk, "index", []) else False)

    d["stock_val"] = d["R"] * d["P_acq"]
    # keep the analytic columns
    keep = ["event_id", "target_name", "ratio_type", "C", "R", "P_acq", "stock_val", "spread",
            "pi_cash", "pi_cash_source", "f_cash", "broke"]
    d = d[keep]
    d.to_csv("arb_deals.csv", index=False)

    # coverage report
    have_terms = d[["C", "R", "P_acq"]].notna().all(axis=1)
    have_demand = d["f_cash"].notna()
    fixed = d.ratio_type.eq("fixed")
    print(f"[terms] deals: {len(d)}")
    print(f"  full structural terms (C,R,P_acq): {have_terms.sum()}")
    print(f"  fixed-ratio: {fixed.sum()}   floating: {(~fixed).sum()}")
    print(f"  realized demand present: {have_demand.sum()}   (the MC-calibration set)")
    print(f"  MC-ready (terms + pi_cash + demand, fixed): {(have_terms & have_demand & fixed).sum()}")
    print(f"  pi_cash source: " + ", ".join(f"{k}={v}" for k, v in d.pi_cash_source.value_counts().items()))
    return d


if __name__ == "__main__":
    build_deals()
