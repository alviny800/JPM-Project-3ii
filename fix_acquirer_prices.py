#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
fix_acquirer_prices.py

Recover missing ACQUIRER daily prices caused by identifier drift (renames like BB&T->Truist,
delistings/'Q' tickers, foreign listings). These acquirers ARE public — the CRSP lookup just
failed on the drifted ticker/name. We resolve PERMNO from the acquirer CUSIP (which does NOT
drift) via the cached crsp.stocknames, respecting the name-validity window, then pull crsp.dsf
prices over each deal's lifecycle and append them to wrds_market_daily.csv (side=acquirer).

Grows the MC-ready set (fixed-ratio deals with a priced/hedgeable acquirer leg). $0 API, WRDS only.
"""
from __future__ import annotations
import re, sys
import datetime as dt
import numpy as np
import pandas as pd
from download_ownership_etf_data import WrdsClient

DAILY = "ma_market_wrds/wrds_market_daily.csv"


def cusip8(v):
    return re.sub(r"[^0-9A-Za-z]", "", str(v)).upper()[:8]


def announce_from_event(eid):
    m = re.search(r"_(\d{8})_", str(eid))
    return pd.to_datetime(m.group(1), format="%Y%m%d", errors="coerce") if m else pd.NaT


def main():
    daily = pd.read_csv(DAILY)
    ana = pd.read_csv("US_election_deals_for_analysis.csv")
    cache = pd.read_csv("stocknames_cache.csv", dtype=str)
    cache["namedt"] = pd.to_datetime(cache["namedt"], errors="coerce")
    cache["nameenddt"] = pd.to_datetime(cache["nameenddt"], errors="coerce")

    # events that have SOME rows but NO acquirer prices
    have_acq = set(daily[daily.side == "acquirer"].event_id.unique())
    missing = sorted(set(daily.event_id.unique()) - have_acq)
    # target name per event (to look up the acquirer CUSIP in the analysis file)
    tgt_name = (daily[daily.side == "target"].dropna(subset=["company_name"])
                .groupby("event_id")["company_name"].first())
    print(f"[fix] events missing acquirer prices: {len(missing)}")

    def acq_cusip_for(eid):
        tn = str(tgt_name.get(eid, ""))
        if not tn:
            return None, None, None
        row = ana[ana["Target Name"].astype(str).str.contains(re.escape(tn[:16]), case=False, na=False)]
        if not len(row):
            return None, None, None
        r = row.iloc[0]
        return r.get("Acquirer cusip"), r.get("Acquirer Name"), r.get("Acquirer Ticker Clean", r.get("Acquirer Ticker"))

    def resolve_permno(c8, on_date):
        m = cache[(cache.ncusip == c8) | (cache.cusip == c8)]
        if not len(m):
            return None, None
        valid = m[(m.namedt <= on_date) & (m.nameenddt >= on_date)]
        pick = valid.iloc[0] if len(valid) else m.sort_values("nameenddt").iloc[-1]
        return int(pick.permno), pick.ticker

    plan = []
    for eid in missing:
        ann = announce_from_event(eid)
        cus, aname, atkr = acq_cusip_for(eid)
        if pd.isna(ann) or not cus or pd.isna(cus):
            continue
        c8 = cusip8(cus)
        permno, tkr = resolve_permno(c8, ann)
        if permno is None:
            print(f"[fix]   {eid[:40]:40s} acq={str(aname)[:22]:22s} cusip8={c8} -> NO PERMNO in cache")
            continue
        plan.append({"event_id": eid, "announce": ann, "permno": permno, "cusip": cus,
                     "symbol": tkr or atkr, "company_name": aname})
    plan = pd.DataFrame(plan)
    print(f"[fix] resolved PERMNO for {len(plan)} / {len(missing)} missing-acquirer events")
    if not len(plan):
        print("[fix] nothing to pull."); return
    print(plan[["event_id", "company_name", "permno", "symbol"]].to_string(index=False))

    print("\n[fix] connecting to WRDS ...")
    client = WrdsClient()
    print("[fix] connected. pulling crsp.dsf prices ...")

    new_rows = []
    for _, r in plan.iterrows():
        start = (r["announce"] - pd.Timedelta(days=30)).date()
        end = (r["announce"] + pd.Timedelta(days=400)).date()
        sql = """
            SELECT permno, date AS price_date, cusip, prc, openprc, bid, ask, bidlo, askhi,
                   vol AS volume, ret, shrout
            FROM crsp.dsf
            WHERE permno = %(permno)s AND date BETWEEN %(start)s AND %(end)s
            ORDER BY date
        """
        try:
            df = client.raw_sql(sql, params={"permno": int(r["permno"]), "start": start, "end": end})
        except Exception as e:
            print(f"[fix]   {r['event_id'][:36]} query failed: {str(e)[:70]}"); continue
        if not len(df):
            print(f"[fix]   {r['event_id'][:36]} permno={r['permno']} -> 0 rows"); continue
        df["price"] = pd.to_numeric(df["prc"], errors="coerce").abs()
        df["open_price"] = pd.to_numeric(df["openprc"], errors="coerce").abs()
        df["event_id"] = r["event_id"]; df["side"] = "acquirer"
        df["company_name"] = r["company_name"]; df["symbol"] = r["symbol"]
        df["announce_date"] = r["announce"].strftime("%m/%d/%Y")
        df["symbol_source"] = "cusip_permno_fix"
        new_rows.append(df)
        print(f"[fix]   {str(r['company_name'])[:24]:24s} permno={r['permno']} -> {len(df)} days "
              f"(${df['price'].dropna().iloc[0]:.2f}..${df['price'].dropna().iloc[-1]:.2f})")
    client.close()

    if not new_rows:
        print("[fix] no prices recovered."); return
    add = pd.concat(new_rows, ignore_index=True)
    # align to existing schema, then append
    daily.to_csv(DAILY.replace(".csv", "_backup.csv"), index=False)
    combined = pd.concat([daily, add], ignore_index=True)
    combined.to_csv(DAILY, index=False)
    print(f"\n[fix] appended {len(add)} acquirer price rows for {add.event_id.nunique()} events "
          f"(backup: {DAILY.replace('.csv','_backup.csv')})")


if __name__ == "__main__":
    main()
