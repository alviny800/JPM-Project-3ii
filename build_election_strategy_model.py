#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
build_election_strategy_model.py

Merge SEC/Claude, WRDS ownership, and WRDS market outputs into a model panel,
audit variable coverage, and run a conservative v1 election/proration strategy
signal. This script is local-only: it does not call SEC, WRDS, or Claude.
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import math
import re
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

import pandas as pd

from download_ownership_etf_data import clean_str, event_id_from_row, numeric_value
from download_wrds_market_data import parse_exchange_ratio, parse_money
from structural_election_model import run_structural_backtest, write_structural_summary


SEC_TRADE_ENTRY_FIELDS = [
    "consideration_menu",
    "cash_consideration_per_share",
    "stock_consideration_per_share",
    "exchange_ratio",
    "cash_cap",
    "stock_cap",
    "proration_formula",
    "non_election_default_rule",
    "election_deadline",
    "record_date",
]

SEC_LABEL_FIELDS = [
    "preliminary_proration_results",
    "final_proration_results",
    "realized_cash_election_demand",
    "realized_stock_election_demand",
    "deal_completion_or_break",
]

EXTERNAL_FIELDS = [
    "target_price",
    "acquirer_price",
    "deal_spread",
    "volume_liquidity",
    "borrow_cost",
    "borrow_availability",
    "short_interest",
    "passive_ownership",
    "etf_ownership",
    "hedge_fund_ownership",
    "nport_on_loan",
    "fund_lending_policy",
    "lending_control_rule",
]

FIELD_SOURCES = {
    **{f: "sec_claude_trade_entry" for f in SEC_TRADE_ENTRY_FIELDS},
    **{f: "sec_claude_post_election_label" for f in SEC_LABEL_FIELDS},
    "target_price": "wrds_crsp_dsf",
    "acquirer_price": "wrds_crsp_dsf",
    "deal_spread": "computed_from_sec_terms_and_wrds_prices",
    "volume_liquidity": "wrds_crsp_dsf",
    "borrow_cost": "borrow_lending_vendor_or_prime_broker",
    "borrow_availability": "borrow_lending_vendor_or_prime_broker",
    "short_interest": "finra_or_vendor_short_interest",
    "passive_ownership": "wrds_etf_holdings_proxy_or_ownership_feed",
    "etf_ownership": "wrds_crsp_mutual_fund_holdings",
    "hedge_fund_ownership": "wrds_13f_or_ownership_feed_with_manager_taxonomy",
    "nport_on_loan": "sec_edgar_nport_or_structured_nport",
    "fund_lending_policy": "sec_edgar_fund_sai_prospectus",
    "lending_control_rule": "fund_sai_lending_agreement_or_legal_assumption",
}


def now_utc() -> str:
    return dt.datetime.utcnow().replace(microsecond=0).isoformat() + "Z"


def read_optional_csv(path: Optional[str]) -> pd.DataFrame:
    if not path:
        return pd.DataFrame()
    p = Path(path)
    if not p.exists():
        return pd.DataFrame()
    return pd.read_csv(p)


def is_present(value: Any) -> bool:
    if value is None:
        return False
    if isinstance(value, float) and math.isnan(value):
        return False
    text = clean_str(value)
    if not text:
        return False
    return text.lower() not in {"nan", "none", "null", "not_found", "not applicable", "n/a"}


def is_found_basis(basis: Any) -> bool:
    text = clean_str(basis).lower()
    return text not in {"", "not_found", "not applicable", "not_applicable", "n/a"}


def make_event_ids(events: pd.DataFrame) -> pd.DataFrame:
    out = events.copy()
    ids: List[str] = []
    for _, row in out.iterrows():
        idx = int(row.get("orig_row_idx", row.name if row.name is not None else -1))
        ids.append(event_id_from_row(idx, row))
    out["event_id"] = ids
    return out


def wide_llm_terms(llm: pd.DataFrame) -> pd.DataFrame:
    if llm.empty:
        return pd.DataFrame(columns=["event_id"])
    rows: Dict[str, Dict[str, Any]] = {}
    for _, row in llm.iterrows():
        event_id = clean_str(row.get("event_id"))
        field = clean_str(row.get("field_name") or row.get("canonical_field"))
        if not event_id or not field:
            continue
        target = rows.setdefault(event_id, {"event_id": event_id})
        target[field] = row.get("value")
        target[f"{field}__basis"] = row.get("basis")
        target[f"{field}__confidence"] = row.get("confidence")
        target[f"{field}__source_doc_ids"] = row.get("source_doc_ids")
        target[f"{field}__source_filing_dates"] = row.get("source_filing_dates")
        target[f"{field}__notes"] = row.get("notes")
    return pd.DataFrame(rows.values())


def merge_model_panel(
    events: pd.DataFrame,
    llm_terms: pd.DataFrame,
    ownership: pd.DataFrame,
    market: pd.DataFrame,
) -> pd.DataFrame:
    panel = make_event_ids(events)
    if not llm_terms.empty:
        panel = panel.merge(llm_terms, on="event_id", how="left")
    if not ownership.empty:
        keep = [c for c in ownership.columns if c not in set(panel.columns) or c == "event_id"]
        panel = panel.merge(ownership[keep], on="event_id", how="left", suffixes=("", "__ownership"))
    if not market.empty:
        keep = [c for c in market.columns if c not in set(panel.columns) or c == "event_id"]
        panel = panel.merge(market[keep], on="event_id", how="left", suffixes=("", "__market"))
    return panel


def parse_fraction(value: Any) -> Optional[float]:
    text = clean_str(value).lower().replace(",", "")
    if not text:
        return None
    pct = re.search(r"([0-9]+(?:\.[0-9]+)?)\s*%", text)
    if pct:
        return float(pct.group(1)) / 100.0
    frac = re.search(r"\b([0-9]+(?:\.[0-9]+)?)\s*/\s*([0-9]+(?:\.[0-9]+)?)\b", text)
    if frac and float(frac.group(2)) != 0:
        return float(frac.group(1)) / float(frac.group(2))
    decimal = re.search(r"\b0\.[0-9]+\b", text)
    if decimal:
        return float(decimal.group(0))
    half_words = ["one-half", "one half", "half of", "50 percent", "fifty percent"]
    if any(w in text for w in half_words):
        return 0.5
    return None


def detect_election_structure(row: pd.Series) -> Dict[str, Any]:
    menu = clean_str(row.get("consideration_menu")).lower()
    formula = clean_str(row.get("proration_formula")).lower()
    cash_cap = row.get("cash_cap")
    stock_cap = row.get("stock_cap")
    no_election = any(
        phrase in menu
        for phrase in [
            "fixed consideration only",
            "no election menu",
            "no cash election",
            "no shareholder election",
            "all holders receive the same",
        ]
    )
    election_words = any(
        phrase in menu or phrase in formula
        for phrase in ["cash election", "stock election", "mixed election", "non-election", "proration", "elect"]
    )
    has_cap = parse_fraction(cash_cap) is not None or parse_fraction(stock_cap) is not None
    has_proration = is_present(formula) and "not_found" not in formula and "not applicable" not in formula
    is_election = bool(election_words and not no_election)
    is_proration_ready = bool(is_election and has_cap and has_proration)
    reason = ""
    if no_election:
        reason = "fixed_consideration_no_shareholder_election"
    elif not is_election:
        reason = "no_cash_or_stock_election_menu_detected"
    elif not has_cap:
        reason = "missing_cash_or_stock_cap_fraction"
    elif not has_proration:
        reason = "missing_computable_proration_formula"
    return {
        "is_election_deal": is_election,
        "is_proration_model_ready": is_proration_ready,
        "election_structure_reason": reason,
        "cash_cap_fraction": parse_fraction(cash_cap),
        "stock_cap_fraction": parse_fraction(stock_cap),
    }


def default_option(row: pd.Series) -> str:
    text = clean_str(row.get("non_election_default_rule")).lower()
    if not text:
        return "unknown"
    if "cash" in text and "stock" not in text:
        return "cash"
    if "stock" in text and "cash" not in text:
        return "stock"
    if "mixed" in text or ("cash" in text and "stock" in text):
        return "mixed"
    return "unknown"


def build_coverage(panel: pd.DataFrame, args: argparse.Namespace) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []
    all_fields = SEC_TRADE_ENTRY_FIELDS + SEC_LABEL_FIELDS + EXTERNAL_FIELDS
    for _, event in panel.iterrows():
        event_id = event.get("event_id")
        for field in all_fields:
            value: Any = None
            basis: Any = ""
            status = "missing"
            notes = ""
            if field in SEC_TRADE_ENTRY_FIELDS + SEC_LABEL_FIELDS:
                value = event.get(field)
                basis = event.get(f"{field}__basis")
                if is_present(value) and is_found_basis(basis):
                    status = "available"
                elif clean_str(basis).lower() in {"not_applicable", "not applicable", "n/a"}:
                    status = "not_applicable"
                elif clean_str(basis).lower() == "not_found":
                    status = "not_found_in_retrieved_evidence"
            elif field == "target_price":
                value = event.get("target_price")
                status = "available" if pd.notna(value) else "missing"
            elif field == "acquirer_price":
                value = event.get("acquirer_price")
                status = "available" if pd.notna(value) else "missing"
            elif field == "deal_spread":
                value = event.get("deal_spread")
                status = "available" if pd.notna(value) else "missing"
            elif field == "volume_liquidity":
                value = event.get("target_adv20")
                status = "available" if pd.notna(value) else "missing"
            elif field == "passive_ownership":
                value = event.get("passive_control_percent")
                status = "available" if pd.notna(value) else "missing"
            elif field == "etf_ownership":
                value = event.get("etf_ownership_percent")
                status = "available" if pd.notna(value) else "missing"
            elif field == "short_interest":
                value = args.short_interest_constant
                status = "assumed_constant"
                notes = "User simplification: short_interest is treated as a constant in v1."
            elif field in {"borrow_cost", "borrow_availability"}:
                if field == "borrow_cost":
                    value = args.borrow_cost_per_share_constant
                else:
                    value = args.borrow_availability_constant
                status = "assumed_constant"
                notes = "User simplification: borrow/lending data are constants in v1."
            elif field == "hedge_fund_ownership":
                status = "missing_manager_taxonomy_or_13f_classification"
            elif field == "nport_on_loan":
                status = "not_required_under_simplified_passive_assumption"
                notes = "User simplification: passive-held shares are permanently passive; no lending haircut."
            elif field in {"fund_lending_policy", "lending_control_rule"}:
                status = "not_required_under_simplified_passive_assumption"
                notes = "User simplification: ignore lent-share control and recall mechanics in v1."

            rows.append(
                {
                    "event_id": event_id,
                    "field": field,
                    "source": FIELD_SOURCES.get(field, ""),
                    "status": status,
                    "value": value,
                    "basis": basis,
                    "required_for_entry_model": field in SEC_TRADE_ENTRY_FIELDS
                    or field
                    in {
                        "target_price",
                        "acquirer_price",
                        "deal_spread",
                        "volume_liquidity",
                        "passive_ownership",
                        "etf_ownership",
                    },
                    "required_for_live_trade": field
                    in {
                        "borrow_cost",
                        "borrow_availability",
                        "fund_lending_policy",
                        "lending_control_rule",
                    },
                    "label_only": field in SEC_LABEL_FIELDS,
                    "notes": notes,
                }
            )
    return pd.DataFrame(rows)


def prediction_for_event(row: pd.Series, args: argparse.Namespace) -> Dict[str, Any]:
    structure = detect_election_structure(row)
    cash = parse_money(row.get("cash_consideration_per_share"))
    exchange_ratio = parse_exchange_ratio(row.get("exchange_ratio"))
    target_price = numeric_value(row.get("target_price"))
    acquirer_price = numeric_value(row.get("acquirer_price"))
    passive = numeric_value(row.get("passive_control_percent"))
    if passive is None:
        passive = numeric_value(row.get("etf_ownership_percent"))
    if passive is None:
        passive = 0.0
    effective_passive = max(0.0, min(1.0, passive * (1.0 - args.passive_lent_haircut)))

    stock_value = None
    if exchange_ratio is not None and acquirer_price is not None:
        stock_value = exchange_ratio * acquirer_price

    base = {
        "event_id": row.get("event_id"),
        "target_name": row.get("Target Name", row.get("target_name", "")),
        "acquirer_name": row.get("Acquirer Name", row.get("acquirer_name", "")),
        "is_election_deal": structure["is_election_deal"],
        "is_proration_model_ready": structure["is_proration_model_ready"],
        "model_status": "blocked",
        "block_reason": structure["election_structure_reason"],
        "cash_value": cash,
        "stock_value": stock_value,
        "target_price": target_price,
        "acquirer_price": acquirer_price,
        "passive_control_percent": passive,
        "effective_passive_percent": effective_passive,
        "cash_cap_fraction": structure["cash_cap_fraction"],
        "stock_cap_fraction": structure["stock_cap_fraction"],
        "favored_option": "",
        "predicted_stock_demand": None,
        "predicted_cash_demand": None,
        "predicted_stock_fill_rate": None,
        "predicted_cash_fill_rate": None,
        "predicted_ev": None,
        "gross_alpha": None,
        "estimated_trading_cost": None,
        "estimated_borrow_cost": None,
        "net_alpha": None,
        "hedge_ratio_acquirer_short_per_target": None,
        "strategy_signal": "no_trade",
        "strategy_notes": "",
    }

    required_numeric = {
        "cash_consideration_per_share": cash,
        "exchange_ratio": exchange_ratio,
        "target_price": target_price,
        "acquirer_price": acquirer_price,
    }
    missing_numeric = [k for k, v in required_numeric.items() if v is None]
    if missing_numeric:
        base["block_reason"] = "missing_numeric_inputs:" + ",".join(missing_numeric)
        return base
    if not structure["is_proration_model_ready"]:
        return base

    assert cash is not None and stock_value is not None and target_price is not None
    favored = "stock" if stock_value > cash else "cash"
    default = default_option(row)
    active_float = max(0.0, 1.0 - effective_passive)
    active_favored = max(0.0, min(1.0, args.active_favored_propensity))

    if default == favored:
        passive_favored = args.passive_default_propensity
    elif default == "mixed":
        passive_favored = 0.5
    elif default == "unknown":
        passive_favored = args.passive_unknown_favored_propensity
    else:
        passive_favored = 1.0 - args.passive_default_propensity

    favored_demand = active_float * active_favored + effective_passive * passive_favored
    other_demand = max(0.0, 1.0 - favored_demand)
    if favored == "stock":
        stock_demand = favored_demand
        cash_demand = other_demand
    else:
        cash_demand = favored_demand
        stock_demand = other_demand

    stock_cap = structure["stock_cap_fraction"]
    cash_cap = structure["cash_cap_fraction"]
    if stock_cap is None and cash_cap is not None:
        stock_cap = max(0.0, min(1.0, 1.0 - cash_cap))
    if cash_cap is None and stock_cap is not None:
        cash_cap = max(0.0, min(1.0, 1.0 - stock_cap))
    if stock_cap is None or cash_cap is None:
        base["block_reason"] = "missing_cap_fraction"
        return base

    stock_fill = min(1.0, stock_cap / stock_demand) if stock_demand > 0 else 1.0
    cash_fill = min(1.0, cash_cap / cash_demand) if cash_demand > 0 else 1.0
    if favored == "stock":
        ev = stock_fill * stock_value + (1.0 - stock_fill) * cash
        hedge_ratio = stock_fill * (exchange_ratio or 0.0)
    else:
        ev = cash_fill * cash + (1.0 - cash_fill) * stock_value
        hedge_ratio = (1.0 - cash_fill) * (exchange_ratio or 0.0)

    trading_cost = target_price * args.trading_cost_bps / 10000.0
    borrow_cost = float(args.borrow_cost_per_share_constant)
    gross_alpha = ev - target_price
    net_alpha = gross_alpha - trading_cost - borrow_cost
    signal = "paper_trade" if net_alpha >= args.min_net_alpha else "no_trade"
    notes = (
        f"v1 simplified assumptions: borrow_cost_per_share={args.borrow_cost_per_share_constant}; "
        f"borrow_availability={args.borrow_availability_constant}; "
        f"short_interest={args.short_interest_constant}; passive_lent_haircut={args.passive_lent_haircut}."
    )
    if args.borrow_availability_constant <= 0:
        signal = "blocked_no_borrow_availability"

    base.update(
        {
            "model_status": "ok",
            "block_reason": "",
            "favored_option": favored,
            "predicted_stock_demand": stock_demand,
            "predicted_cash_demand": cash_demand,
            "predicted_stock_fill_rate": stock_fill,
            "predicted_cash_fill_rate": cash_fill,
            "predicted_ev": ev,
            "gross_alpha": gross_alpha,
            "estimated_trading_cost": trading_cost,
            "estimated_borrow_cost": borrow_cost,
            "net_alpha": net_alpha,
            "hedge_ratio_acquirer_short_per_target": hedge_ratio,
            "strategy_signal": signal,
            "strategy_notes": notes,
        }
    )
    return base


def build_predictions(panel: pd.DataFrame, args: argparse.Namespace) -> pd.DataFrame:
    return pd.DataFrame([prediction_for_event(row, args) for _, row in panel.iterrows()])


def write_summary(out_dir: Path, coverage: pd.DataFrame, preds: pd.DataFrame) -> None:
    status_counts = coverage.groupby(["field", "status"], dropna=False).size().reset_index(name="events")
    blocked = preds["model_status"].ne("ok").sum() if not preds.empty else 0
    summary = {
        "generated_at": now_utc(),
        "event_count": int(preds.shape[0]),
        "model_ready_count": int(preds["model_status"].eq("ok").sum()) if not preds.empty else 0,
        "blocked_count": int(blocked),
        "strategy_signal_counts": preds["strategy_signal"].value_counts(dropna=False).to_dict() if not preds.empty else {},
        "coverage_status_counts": status_counts.to_dict(orient="records"),
    }
    (out_dir / "model_run_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Build election model panel, coverage report, and risk-aware strategy signal.")
    p.add_argument("--events", required=True, help="candidate_events.csv")
    p.add_argument("--llm-extractions", required=True, help="llm_field_extractions.csv")
    p.add_argument("--ownership-mix", required=True, help="ownership_mix_by_event.csv")
    p.add_argument("--market-features", required=True, help="event_market_features.csv")
    p.add_argument("--output-dir", required=True)
    p.add_argument("--passive-lent-haircut", type=float, default=0.0)
    p.add_argument("--borrow-cost-per-share-constant", type=float, default=0.0)
    p.add_argument("--borrow-availability-constant", type=float, default=1.0)
    p.add_argument("--short-interest-constant", type=float, default=0.0)
    p.add_argument("--active-favored-propensity", type=float, default=0.85)
    p.add_argument("--passive-default-propensity", type=float, default=0.90)
    p.add_argument("--passive-unknown-favored-propensity", type=float, default=0.50)
    p.add_argument("--trading-cost-bps", type=float, default=10.0)
    p.add_argument("--acquirer-trading-cost-bps", type=float, default=None,
                   help="Trading-cost bps for the acquirer short leg. Defaults to --trading-cost-bps.")
    p.add_argument("--use-bid-ask-costs", dest="use_bid_ask_costs", action="store_true", default=True,
                   help="Add available CRSP bid-ask spread proxies to target/acquirer transaction costs.")
    p.add_argument("--no-bid-ask-costs", dest="use_bid_ask_costs", action="store_false",
                   help="Ignore bid-ask spread proxies and use only bps trading-cost assumptions.")
    p.add_argument("--annual-borrow-cost-bps", type=float, default=0.0,
                   help="Annualized borrow cost for the acquirer short leg.")
    p.add_argument("--min-net-alpha", type=float, default=0.0)
    p.add_argument("--min-p05-net-alpha", type=float, default=-1e18,
                   help="Optional Monte Carlo p5 net-alpha floor per target share.")
    p.add_argument("--max-loss-probability", type=float, default=1.0,
                   help="Optional Monte Carlo probability-of-loss ceiling.")
    p.add_argument("--trade-decision-metric", choices=["mean", "p05", "cvar05", "deterministic"],
                   default="mean",
                   help="Risk metric used to choose and gate cash vs stock election.")
    p.add_argument("--rolling-window-events", type=int, default=50,
                   help="Number of prior labeled events used to fit p and q.")
    p.add_argument("--min-fit-events", type=int, default=10,
                   help="Minimum prior labeled events before fitting; otherwise defaults are used.")
    p.add_argument("--p-grid-size", type=int, default=51,
                   help="Grid size for irrational cash-election probability p.")
    p.add_argument("--q-grid-size", type=int, default=51,
                   help="Grid size for rational original ownership share q.")
    p.add_argument("--default-irrational-cash-prob", type=float, default=0.5,
                   help="Fallback p before enough rolling observations exist.")
    p.add_argument("--default-rational-share", type=float, default=0.3,
                   help="Fallback q before enough rolling observations exist.")
    p.add_argument("--trade-notional", type=float, default=1_000_000.0,
                   help="Dollar notional for our target long.")
    p.add_argument("--own-hedge-policy", choices=["dollar_neutral", "conversion_expected", "none"],
                   default="conversion_expected",
                   help="How to size the acquirer short leg for realized P&L.")
    p.add_argument("--holding-period-days", type=int, default=30,
                   help="Fallback holding period for borrow/deal-break cost calculations when dates are unavailable.")
    p.add_argument("--mc-draws", type=int, default=2000,
                   help="Monte Carlo draws per event/election choice. Use 0 to disable.")
    p.add_argument("--mc-seed", type=int, default=1729)
    p.add_argument("--mc-demand-concentration", type=float, default=75.0,
                   help="Beta concentration around predicted cash demand. Higher values mean less demand uncertainty.")
    p.add_argument("--mc-min-beta-alpha", type=float, default=0.25,
                   help="Minimum beta shape parameter used when predicted demand is near zero or one.")
    p.add_argument("--deal-break-prob", type=float, default=0.0,
                   help="Fallback deal-break probability if no event-level column is supplied.")
    p.add_argument("--break-loss-pct", type=float, default=0.30,
                   help="Fallback target loss if the deal breaks and no break-price column is supplied.")
    p.add_argument("--acquirer-break-return-pct", type=float, default=0.0,
                   help="Fallback acquirer return in a deal-break scenario for the short hedge.")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    events = pd.read_csv(args.events)
    llm = wide_llm_terms(pd.read_csv(args.llm_extractions))
    ownership = read_optional_csv(args.ownership_mix)
    market = read_optional_csv(args.market_features)

    panel = merge_model_panel(events, llm, ownership, market)
    for _, row in panel.iterrows():
        structure = detect_election_structure(row)
        for key, value in structure.items():
            panel.loc[panel["event_id"] == row["event_id"], key] = value

    coverage = build_coverage(panel, args)
    heuristic_preds = build_predictions(panel, args)
    preds, rolling_params, backtest_summary = run_structural_backtest(panel, args)

    panel.to_csv(out_dir / "model_input_panel.csv", index=False)
    coverage.to_csv(out_dir / "variable_coverage_report.csv", index=False)
    heuristic_preds.to_csv(out_dir / "election_model_predictions_heuristic.csv", index=False)
    preds.to_csv(out_dir / "election_model_predictions.csv", index=False)
    rolling_params.to_csv(out_dir / "rolling_parameter_estimates.csv", index=False)
    preds.to_csv(out_dir / "election_backtest_trades.csv", index=False)
    write_structural_summary(out_dir / "backtest_summary.json", backtest_summary)
    write_summary(out_dir, coverage, preds)

    print(f"[{now_utc()}] Wrote {out_dir / 'model_input_panel.csv'}")
    print(f"[{now_utc()}] Wrote {out_dir / 'variable_coverage_report.csv'}")
    print(f"[{now_utc()}] Wrote {out_dir / 'election_model_predictions_heuristic.csv'}")
    print(f"[{now_utc()}] Wrote {out_dir / 'election_model_predictions.csv'}")
    print(f"[{now_utc()}] Wrote {out_dir / 'rolling_parameter_estimates.csv'}")
    print(f"[{now_utc()}] Wrote {out_dir / 'election_backtest_trades.csv'}")
    print(f"[{now_utc()}] Wrote {out_dir / 'backtest_summary.json'}")
    print(f"[{now_utc()}] Wrote {out_dir / 'model_run_summary.json'}")


if __name__ == "__main__":
    main()
