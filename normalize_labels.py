#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
normalize_labels.py

Second, cheap LLM pass that NORMALIZES the heterogeneous realized-election disclosures
into strictly-defined numeric columns. The raw extractor faithfully pulled each issuer's
prose, but issuers report the same thing in incompatible ways — % elected, % received
after proration, raw share counts, aggregate dollars, or just "oversubscribed". A regex
can't tell demand ("elected") from allocation ("received/converted"), and drops dollar-only
cases. This pass sends ONLY the extracted prose (not the 250K-token filings), so it's ~$1-2.

Output: normalized_labels.csv with pct_elected_cash (the model's true target), plus
pct_received_cash, proration factor, dollars, and a disclosure_type flag.
"""
from __future__ import annotations
import argparse, json, os
from pathlib import Path
import pandas as pd
import download_ma_edgar_files as dl

NORM_SYSTEM = """You normalize free-text M&A cash-or-stock election RESULTS into strict numeric fields.
You are given the raw text a prior extractor pulled from SEC filings for ONE deal. Return exactly ONE
JSON object. Use null whenever a value is not determinable from the supplied text — never guess.

Fields to return:
- pct_elected_cash: percent of shares whose holders ELECTED/CHOSE cash (election DEMAND), 0-100, or null
- pct_elected_stock: percent electing stock, 0-100, or null
- pct_received_cash: percent of shares that RECEIVED cash AFTER proration/allocation (an OUTCOME,
  distinct from what was elected), 0-100, or null
- cash_proration_factor: the fill/proration factor applied to cash elections, a fraction 0-1, or null
- aggregate_cash_usd: aggregate cash paid in US dollars if disclosed only as a dollar amount, or null
- shares_base: the total shares base used for percentages if stated, or null
- disclosure_type: one of ["elected_pct","elected_shares_derived","received_or_allocated",
  "dollars_only","proration_factor_only","oversubscribed_qualitative","not_disclosed"]
- basis: "direct" (explicitly stated), "derived" (you computed % from a share count and a base),
  or "not_found"
- notes: one short sentence

CRITICAL RULES:
1. Distinguish ELECTED (what holders chose = demand) from RECEIVED / CONVERTED / ALLOCATED (what they
   got after proration). "converted into cash", "received the cash consideration" = received, NOT elected.
   "elected cash", "made a cash election", "shares electing cash" = elected.
2. If the text gives a raw share COUNT electing cash plus a shares base, DERIVE pct_elected_cash =
   100 * count / base, set basis="derived".
3. If only an aggregate dollar amount is disclosed, fill aggregate_cash_usd and leave the pct_elected
   fields null (disclosure_type="dollars_only").
4. Return JSON only, no prose outside the object."""


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--extractions", default="ma_edgar_full/llm_field_extractions.csv")
    p.add_argument("--out", default="normalized_labels.csv", type=Path)
    p.add_argument("--key-env", default="ANTHROPIC_API_KEY")
    p.add_argument("--model", default="claude-sonnet-5")
    p.add_argument("--max-cost-usd", type=float, default=5.0)
    p.add_argument("--sync", action="store_true", help="Run synchronous per-deal calls instead of a batch (faster to finish when the batch queue is backed up; ~2x cost).")
    args = p.parse_args()

    d = pd.read_csv(args.extractions)
    d = d[d["event_id"].notna()].copy()
    wide = d.pivot_table(index=["event_id", "target_name", "acquirer_name"],
                         columns="field_name", values="value", aggfunc="first").reset_index()
    LAB = ["realized_cash_election_demand", "realized_stock_election_demand",
           "final_proration_results", "preliminary_proration_results"]
    CTX = ["consideration_menu", "proration_formula", "cash_cap", "stock_cap"]

    def nn(v):
        s = "" if pd.isna(v) else str(v)
        return s.strip().lower() not in ("", "nan", "none", "null", "not disclosed", "not found", "n/a")

    payloads = []
    for _, r in wide.iterrows():
        if not any(nn(r.get(f)) for f in LAB):
            continue
        payloads.append({
            "system_prompt": NORM_SYSTEM,
            "task": "normalize_election_results",
            "event": {"event_id": str(r["event_id"])},
            "event_id": str(r["event_id"]),
            "target_name": str(r.get("target_name", "")),
            "acquirer_name": str(r.get("acquirer_name", "")),
            "raw_disclosure": {f: (str(r.get(f)) if nn(r.get(f)) else None) for f in LAB + CTX},
        })
    print(f"[normalize] {len(payloads)} deals with realized-demand prose to normalize")

    key = os.environ.get(args.key_env, "")
    if not key:
        raise SystemExit(f"[normalize] no key in ${args.key_env}")

    if args.sync:
        import time
        records = []
        for i, p in enumerate(payloads):
            rec = None
            for attempt in range(4):
                try:
                    rec = dl.call_anthropic(p, api_key=key, model=args.model, max_tokens=1500)
                    break
                except Exception as e:
                    if attempt == 3:
                        rec = {"parsed": {"parse_error": True, "error": str(e)}, "response": {}}
                    else:
                        time.sleep(2 ** attempt)
            rec["request"] = {"event_id": p.get("event_id", "")}  # track event_id for merge
            records.append(rec)
            time.sleep(0.25)  # gentle pacing to avoid rate limits
            if (i + 1) % 20 == 0:
                print(f"[normalize] sync {i+1}/{len(payloads)}", flush=True)
    else:
        records = dl.run_anthropic_batch(payloads, api_key=key, model=args.model, max_tokens=1500,
                                         max_cost_usd=args.max_cost_usd, poll_seconds=20)

    cols = ["pct_elected_cash", "pct_elected_stock", "pct_received_cash", "cash_proration_factor",
            "aggregate_cash_usd", "shares_base", "disclosure_type", "basis", "notes"]
    rows = []
    for rec in records:
        eid = rec.get("request", {}).get("event_id", "")
        pr = rec.get("parsed", {})
        if isinstance(pr, dict) and not pr.get("parse_error"):
            rows.append({"event_id": eid, **{c: pr.get(c) for c in cols}})
        else:
            rows.append({"event_id": eid, "disclosure_type": "parse_error"})
    out = pd.DataFrame(rows)
    out.to_csv(args.out, index=False)
    print(f"[normalize] wrote {args.out}: {len(out)} rows")
    print(f"[normalize] pct_elected_cash populated: {pd.to_numeric(out['pct_elected_cash'],errors='coerce').notna().sum()}")
    print("[normalize] disclosure_type counts:")
    print(out["disclosure_type"].value_counts().to_string())


if __name__ == "__main__":
    main()
