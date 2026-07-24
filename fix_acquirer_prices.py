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

# Explicit acquirer-PERMNO overrides for identifier COLLISIONS (distinct from the drift the pull
# below repairs). Bloomberg back-stamps the acquirer field with the entity's *current* name/ticker/
# CUSIP, so when an acquirer is later itself acquired and renamed, the recorded CUSIP resolves to the
# WRONG PERMNO and the stock leg is priced off the wrong company. Keyed by an event_id substring ->
# the correct (permno, company_name, symbol, note). These events already HAVE acquirer rows, so the
# override REPLACES them rather than appending.
#   Isle of Capri: real acquirer was Eldorado Resorts (ERI, PERMNO 14882, ncusip 28470R10). Eldorado
#   bought old Caesars in 2020 and renamed to "Caesars Entertainment"/CZR/12769G100, so Bloomberg's
#   acquirer CUSIP resolved to the OLD bankrupt Caesars (PERMNO 13267, ~$7.45, data stops 2016-10).
#   The disclosed ratio itself cites "ERI's 30-day VWAP of $14.04" — matching PERMNO 14882 ($13.84 at
#   announce). Correct deadline price (2017-05-01) is $19.15.
#   MTR Gaming: same collision — MTR merged with Eldorado HoldCo in Sept 2014 to FORM Eldorado
#   Resorts (PERMNO 14882, first trades 2014-09-22 ~$4.35, right at close). Bloomberg records the
#   acquirer as the 2020-surviving "Caesars Entertainment"/CZR, so it resolved to old Caesars
#   (PERMNO 13267) again. Election was $6.05 cash OR 1.0 new-Eldorado share; at close Eldorado ~$4.35
#   so cash is the richer side (39% gap, under threshold -> tradeable). Caveat: the new stock barely
#   existed at the election deadline, so the deadline price is the first available Eldorado print.
ACQUIRER_PERMNO_OVERRIDES = {
    "Isle_of_Capri": (14882, "Eldorado Resorts Inc", "ERI",
                      "bbg_surviving_entity_collision:caesars_permno_13267->eldorado_permno_14882"),
    "MTR_Gaming": (14882, "Eldorado Resorts Inc", "ERI",
                   "bbg_surviving_entity_collision:caesars_permno_13267->eldorado_permno_14882"),
}

# TARGET-side PERMNO overrides: the Bloomberg TARGET cusip was wrong/garbage and misresolved to an
# unrelated company, so the entry price M is nonsense and the deal trips the return guard.
#   Sirius International: bbg target cusip 012348108 resolved to Honeywell (PERMNO 10145, symbol HON,
#   ~$160). Real target is Sirius Intl Insurance Group Ltd (PERMNO 18260, ticker SG, cusip G8196D10,
#   valid 2018-11 to 2021-02, ~$8) — the acquirer SiriusPoint took it over 8/2020.
TARGET_PERMNO_OVERRIDES = {
    "Sirius_International": (18260, "Sirius Intl Insurance Group Ltd", "SG",
                            "bad_bbg_target_cusip_012348108->honeywell_permno_10145; real permno=18260"),
}


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
                     "symbol": tkr or atkr, "company_name": aname, "side": "acquirer", "replace": False})

    # explicit collision overrides: correct the acquirer/target PERMNO for events that already have
    # (wrong) rows on that side. Match each override key against the full event-id universe.
    all_events = set(daily.event_id.unique())
    for side, overrides in (("acquirer", ACQUIRER_PERMNO_OVERRIDES), ("target", TARGET_PERMNO_OVERRIDES)):
        for key, (permno, cname, sym, note) in overrides.items():
            for eid in sorted(e for e in all_events if key in str(e)):
                ann = announce_from_event(eid)
                if pd.isna(ann):
                    continue
                plan.append({"event_id": eid, "announce": ann, "permno": int(permno), "cusip": "",
                             "symbol": sym, "company_name": cname, "side": side,
                             "replace": True, "note": note})
                print(f"[fix] override ({side}): {eid[:40]:40s} -> permno={permno} ({cname})")

    plan = pd.DataFrame(plan)
    n_missing = int((~plan.get("replace", False)).sum()) if len(plan) else 0
    print(f"[fix] resolved PERMNO for {n_missing} / {len(missing)} missing-acquirer events "
          f"+ {len(plan) - n_missing} override(s)")
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
        side = r.get("side", "acquirer")
        df["event_id"] = r["event_id"]; df["side"] = side
        df["company_name"] = r["company_name"]; df["symbol"] = r["symbol"]
        df["announce_date"] = r["announce"].strftime("%m/%d/%Y")
        df["symbol_source"] = f"{side}_permno_override" if r.get("replace") else "cusip_permno_fix"
        new_rows.append(df)
        print(f"[fix]   {str(r['company_name'])[:24]:24s} permno={r['permno']} -> {len(df)} days "
              f"(${df['price'].dropna().iloc[0]:.2f}..${df['price'].dropna().iloc[-1]:.2f})")
    client.close()

    if not new_rows:
        print("[fix] no prices recovered."); return
    add = pd.concat(new_rows, ignore_index=True)
    # align to existing schema, then append
    daily.to_csv(DAILY.replace(".csv", "_backup.csv"), index=False)
    # for override events, drop the existing WRONG rows on that side before appending the corrected ones
    replace = plan[plan["replace"]] if "replace" in plan.columns else plan.iloc[0:0]
    if len(replace):
        drop_keys = {f"{e}|{s}" for e, s in zip(replace["event_id"], replace["side"])}
        key_daily = daily["event_id"].astype(str) + "|" + daily["side"].astype(str)
        before = len(daily)
        daily = daily[~key_daily.isin(drop_keys)]
        print(f"[fix] dropped {before - len(daily)} wrong rows for "
              f"{len(drop_keys)} override event/side(s)")
    combined = pd.concat([daily, add], ignore_index=True)
    combined.to_csv(DAILY, index=False)
    print(f"\n[fix] appended {len(add)} acquirer price rows for {add.event_id.nunique()} events "
          f"(backup: {DAILY.replace('.csv','_backup.csv')})")


if __name__ == "__main__":
    main()
