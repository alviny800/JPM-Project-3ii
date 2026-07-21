#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
reextract_unresolved.py

Fast, targeted re-extraction of election DEMAND for deals the normalization pass couldn't
resolve — reads the SAME cached full-document payloads but with the NEW sharp
FIELD_LLM_SYSTEM_PROMPT (elected-vs-received separated). Recovers demand that the loose
first pass mis-reported as allocation. Fixed-ratio deals only by default (the ones that can
move the p_active(spread) fit).

Writes reextracted_labels.csv and (optionally) folds pct_elected_cash into normalized_labels.csv.
"""
from __future__ import annotations
import argparse, json, os, re, time
import numpy as np
import pandas as pd
import download_ma_edgar_files as dl


def run_sync(payloads, api_key, model, max_tokens, max_cost_usd):
    """Synchronous per-deal extraction with real-time cost tracking + a hard cap.
    Returns records shaped like run_anthropic_batch: {'request': {'event_id':..}, 'parsed': {..}}."""
    pin, pout = dl.MODEL_PRICES_USD_PER_MTOK.get(model, (2.0, 10.0))
    spent, records = 0.0, []
    for i, pl in enumerate(payloads):
        eid = pl["event"]["event_id"]
        for attempt in range(4):
            try:
                res = dl.call_anthropic(pl, api_key=api_key, model=model, max_tokens=max_tokens)
                usage = res.get("response", {}).get("usage", {})
                cost = usage.get("input_tokens", 0) / 1e6 * pin + usage.get("output_tokens", 0) / 1e6 * pout
                spent += cost
                records.append({"request": {"event_id": eid}, "parsed": res.get("parsed", {})})
                print(f"[sync] {i+1}/{len(payloads)} {eid[:34]:34s} ${cost:.3f}  running=${spent:.2f}", flush=True)
                break
            except Exception as e:
                wait = 3 * (attempt + 1)
                print(f"[sync] {eid[:30]} attempt {attempt+1} failed: {str(e)[:80]} — retry in {wait}s", flush=True)
                time.sleep(wait)
        else:
            print(f"[sync] {eid[:30]} GAVE UP after retries", flush=True)
        if spent > max_cost_usd:
            print(f"[sync] HARD CAP hit (${spent:.2f} > ${max_cost_usd}) — stopping after {i+1} deals", flush=True)
            break
        time.sleep(0.3)
    print(f"[sync] done: {len(records)} deals, total spend ~${spent:.2f}", flush=True)
    return records


def pct(s):
    if s is None or (isinstance(s, float) and np.isnan(s)):
        return np.nan
    t = str(s).replace(",", "")
    m = re.search(r"(\d+\.?\d*)\s*%", t)
    if m:
        v = float(m.group(1)); return v if 0 <= v <= 100 else np.nan
    m = re.search(r"\b(0?\.\d+)\b", t)
    if m:
        v = float(m.group(1)) * 100; return v if 0 <= v <= 100 else np.nan
    return np.nan


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--ids", default="reextract_ids.txt")
    p.add_argument("--fixed-only", action="store_true", help="restrict to fixed exchange-ratio deals")
    p.add_argument("--payloads", default="ma_edgar_full/llm_field_payloads.jsonl")
    p.add_argument("--extractions", default="ma_edgar_full/llm_field_extractions.csv")
    p.add_argument("--out", default="reextracted_labels.csv")
    p.add_argument("--key-env", default="ANTHROPIC_API_KEY")
    p.add_argument("--max-cost-usd", type=float, default=12.0)
    p.add_argument("--sync", action="store_true", help="synchronous per-deal calls (reliable, full price)")
    p.add_argument("--max-tokens", type=int, default=8000)
    p.add_argument("--merge-into", default="normalized_labels.csv")
    args = p.parse_args()

    ids = [l.strip() for l in open(args.ids) if l.strip()]
    if args.fixed_only:
        d = pd.read_csv(args.extractions)
        w = d[d.event_id.notna()].pivot_table(index="event_id", columns="field_name", values="value", aggfunc="first")
        FLOAT = re.compile(r"vwap|÷|floating|average (trading|closing|share) price|collar", re.I)
        ids = [e for e in ids if e in w.index and not FLOAT.search(str(w.loc[e, "exchange_ratio"]))]
    print(f"[reextract] {len(ids)} deals to re-extract (fixed_only={args.fixed_only})")

    payloads = {json.loads(l)["event"]["event_id"]: json.loads(l)
                for l in open(args.payloads) if l.strip()}
    sharp = dl.FIELD_LLM_SYSTEM_PROMPT
    batch_payloads = []
    for e in ids:
        pl = payloads.get(e)
        if pl is None:
            continue
        pl = dict(pl); pl["system_prompt"] = sharp   # override with the tightened prompt
        batch_payloads.append(pl)

    key = os.environ.get(args.key_env, "")
    if not key:
        raise SystemExit(f"[reextract] no key in ${args.key_env}")
    if args.sync:
        records = run_sync(batch_payloads, api_key=key, model="claude-sonnet-5",
                           max_tokens=args.max_tokens, max_cost_usd=args.max_cost_usd)
    else:
        records = dl.run_anthropic_batch(batch_payloads, api_key=key, model="claude-sonnet-5",
                                         max_tokens=16000, max_cost_usd=args.max_cost_usd, poll_seconds=30)

    rows = []
    for rec in records:
        eid = rec.get("request", {}).get("event_id", "")
        parsed = rec.get("parsed", {})
        fields = parsed.get("fields", {}) if isinstance(parsed, dict) else {}
        def fv(f):
            x = fields.get(f); return (x.get("value") if isinstance(x, dict) else x)
        rows.append({"event_id": eid,
                     "pct_elected_cash": pct(fv("realized_cash_election_demand")),
                     "pct_elected_stock": pct(fv("realized_stock_election_demand")),
                     "cash_prose": str(fv("realized_cash_election_demand"))[:110]})
    out = pd.DataFrame(rows)
    out.to_csv(args.out, index=False)
    rec_n = pd.to_numeric(out["pct_elected_cash"], errors="coerce").notna().sum()
    print(f"[reextract] wrote {args.out}: {len(out)} deals, {rec_n} now have pct_elected_cash")

    # fold recovered demand back into normalized_labels.csv
    if args.merge_into and os.path.exists(args.merge_into):
        norm = pd.read_csv(args.merge_into)
        upd = out.set_index("event_id")["pct_elected_cash"].dropna()
        norm = norm.set_index("event_id")
        n_before = pd.to_numeric(norm["pct_elected_cash"], errors="coerce").notna().sum()
        for e, v in upd.items():
            if e in norm.index:
                norm.loc[e, "pct_elected_cash"] = v
                norm.loc[e, "disclosure_type"] = "reextracted_elected"
        norm.reset_index().to_csv(args.merge_into, index=False)
        n_after = pd.to_numeric(norm["pct_elected_cash"], errors="coerce").notna().sum()
        print(f"[reextract] merged into {args.merge_into}: clean demand {n_before} -> {n_after}")


if __name__ == "__main__":
    main()
