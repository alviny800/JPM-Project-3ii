#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
deadline_spread.py

Build the DEADLINE-DATE election spread (the theoretically-correct predictor) from data
we already have — no new WRDS pull:
  1. deadline date  = absolute date parsed from Claude's `election_deadline`, else the
     CRSP close date (deadline sits a few days before close).
  2. acquirer price at the deadline = nearest daily CRSP price on/before that date
     (from wrds_market_daily.csv, whose announce-centered window spans most deadlines).
  3. spread_deadline = cash_election_value - exchange_ratio * acquirer_price_deadline.
Then re-fit p_active(spread_deadline) on the clean normalized demand labels.
"""
from __future__ import annotations
import re
import numpy as np
import pandas as pd
from scipy import stats

DATE_RE = re.compile(r"([A-Z][a-z]{2,8}\.?\s+\d{1,2},?\s+\d{4}|\d{1,2}/\d{1,2}/\d{4}|\d{4}-\d{2}-\d{2})")


def parse_date(s):
    if pd.isna(s):
        return pd.NaT
    m = DATE_RE.search(str(s))
    return pd.to_datetime(m.group(1), errors="coerce") if m else pd.NaT


def num_in(s, lo, hi):
    """First number in the text that falls in [lo,hi] — avoids grabbing a share count."""
    if pd.isna(s):
        return np.nan
    for tok in re.findall(r"\d+\.?\d*", str(s).replace(",", "")):
        v = float(tok)
        if lo <= v <= hi:
            return v
    return np.nan


# ---- terms + normalized demand (from the merged panel) ----
panel = pd.read_csv("eda_output/merged_panel.csv")
panel = panel[panel.event_id.notna()].copy()
ext = pd.read_csv("ma_edgar_full/llm_field_extractions.csv")
wide = ext[ext.event_id.notna()].pivot_table(index="event_id", columns="field_name", values="value", aggfunc="first")

# deadline date: absolute from election_deadline; else close-date anchor
dl_abs = wide["election_deadline"].map(parse_date) if "election_deadline" in wide.columns else pd.Series(dtype="datetime64[ns]")
# close-date fallback keyed via target cusip
cd = pd.read_csv("target_close_dates.csv", dtype=str)
close_by_c8 = {r["target_cusip8"]: r["close_date"] for _, r in cd.iterrows() if str(r.get("close_date", "")).strip()}
smap = pd.read_csv("ma_market_wrds/event_security_map.csv")
tgt = smap[smap.side == "target"][["event_id", "cusip"]].dropna()
def c8(v): return re.sub(r"[^0-9A-Za-z]", "", str(v)).upper()[:8]
close_by_event = {r["event_id"]: pd.to_datetime(close_by_c8.get(c8(r["cusip"]), None), errors="coerce")
                  for _, r in tgt.iterrows()}

daily = pd.read_csv("ma_market_wrds/wrds_market_daily.csv")
daily["price_date"] = pd.to_datetime(daily["price_date"], errors="coerce")
acq = daily[daily.side == "acquirer"].dropna(subset=["price_date", "price"])

rows = []
for eid in panel["event_id"]:
    dd = dl_abs.get(eid) if eid in dl_abs.index else pd.NaT
    src = "election_deadline"
    if pd.isna(dd):
        dd = close_by_event.get(eid, pd.NaT)
        src = "close_date"
    if pd.isna(dd):
        rows.append({"event_id": eid, "deadline_date": pd.NaT, "acq_price_deadline": np.nan, "spread_deadline": np.nan, "date_source": "none"})
        continue
    a = acq[acq.event_id == eid]
    onbefore = a[a.price_date <= dd]
    px = onbefore.sort_values("price_date")["price"].iloc[-1] if len(onbefore) else (
        a.iloc[(a.price_date - dd).abs().argmin()]["price"] if len(a) else np.nan)
    rows.append({"event_id": eid, "deadline_date": dd, "acq_price_deadline": px, "date_source": src})

dd_df = pd.DataFrame(rows).merge(panel[["event_id", "target_name", "cash_consideration_per_share",
                                        "exchange_ratio", "realized_cash_share"]], on="event_id", how="left")

# FIXED vs FLOATING exchange ratio. A floating ratio (= cash / VWAP, set at closing) makes
# the stock value equal the cash value BY DESIGN -> spread is structurally ~0 and carries no
# signal. Only FIXED-ratio deals let the acquirer's price movement create a real spread, so
# only they can test p_active(spread). Detect floating language and exclude those.
FLOAT_RE = re.compile(r"vwap|÷|floating|average (trading|closing|share) price|/\s*\w+\s*(10|5|20)[- ]day|collar", re.I)
dd_df["ratio_type"] = dd_df["exchange_ratio"].map(
    lambda s: "floating" if (not pd.isna(s) and FLOAT_RE.search(str(s))) else "fixed")
dd_df["cash_val"] = dd_df["cash_consideration_per_share"].map(lambda s: num_in(s, 1, 2000))
# only trust a parsed ratio for FIXED deals, and keep it in the plausible exchange-ratio range
dd_df["ratio"] = dd_df.apply(
    lambda r: num_in(r["exchange_ratio"], 0.02, 5) if r["ratio_type"] == "fixed" else np.nan, axis=1)
dd_df["stock_val_deadline"] = dd_df["ratio"] * dd_df["acq_price_deadline"]
dd_df["spread_deadline"] = dd_df["cash_val"] - dd_df["stock_val_deadline"]
dd_df.to_csv("deadline_spread.csv", index=False)
print(f"ratio type: fixed={ (dd_df.ratio_type=='fixed').sum() }  floating={ (dd_df.ratio_type=='floating').sum() } "
      f"(floating deals are spread~0 by design, excluded from the fit)")

# coverage
print(f"deals: {len(dd_df)}")
print(f"  deadline date resolved: {dd_df['deadline_date'].notna().sum()} "
      f"(election_deadline={ (dd_df.date_source=='election_deadline').sum() }, close_date={ (dd_df.date_source=='close_date').sum() })")
print(f"  acquirer price at deadline: {dd_df['acq_price_deadline'].notna().sum()}")
print(f"  sane cash & ratio: {(dd_df.cash_val.notna() & dd_df.ratio.notna()).sum()}")
print(f"  spread_deadline computed: {dd_df['spread_deadline'].notna().sum()}")

# ---- re-fit p_active on the DEADLINE spread (clean demand deals) ----
d = dd_df.dropna(subset=["spread_deadline", "realized_cash_share"]).copy()
s = d["spread_deadline"]; y = d["realized_cash_share"]
print(f"\n=== spread_deadline distribution (n={len(d)}): min={s.min():.1f} max={s.max():.1f} median={s.median():.2f} std={s.std():.1f} ===")
m = s.abs() <= 50  # drop any residual mis-parses
print(f"  sane (|spread|<=50): {m.sum()}")
if m.sum() >= 5:
    sl, ic, r, pv, se = stats.linregress(s[m], y[m])
    print(f"\n=== p_active(DEADLINE spread), n={m.sum()} ===")
    print(f"  p_active = {ic:.3f} + {sl:.5f} * spread")
    print(f"  R = {r:.3f}   R² = {r**2:.3f}   p = {pv:.4f}")
    print(f"  vs the announce-date fit earlier: R²=0.000, p=0.95")
