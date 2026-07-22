#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Rolling structural election-demand fit and cap-aware backtest.

Model notation used here matches the project note:
- p: irrational investors' cash-election probability.
- q: rational EV-sensitive investors' original share of target ownership.
- passive ownership comes from the ownership pipeline and is treated as known.

For each event, parameters are fit on prior events by event count, not calendar
time.  The model predicts election demand before our trade, then recomputes
cap/proration after adding our own notional position and election instruction.
"""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import math
import random
import re
from typing import Any, Dict, Iterable, List, Optional, Tuple

import pandas as pd

from deal_outcome_model import normalize_outcome_label, outcome_probability_row


EPS = 1e-12
NEG_INF = -1e18


def arg_value(args: argparse.Namespace, name: str, default: Any) -> Any:
    return getattr(args, name, default)


def bounded_prob(value: Optional[float], default: Optional[float] = None) -> Optional[float]:
    if value is None or not math.isfinite(value):
        return default
    return max(0.0, min(1.0, value))


def stable_seed(base_seed: int, *parts: Any) -> int:
    token = "|".join([str(base_seed), *(str(p) for p in parts)])
    return int(hashlib.sha256(token.encode("utf-8")).hexdigest()[:16], 16)


def quantile(values: List[float], q: float) -> Optional[float]:
    clean = sorted(v for v in values if v is not None and math.isfinite(v))
    if not clean:
        return None
    if len(clean) == 1:
        return clean[0]
    q = max(0.0, min(1.0, q))
    pos = q * (len(clean) - 1)
    lo = int(math.floor(pos))
    hi = int(math.ceil(pos))
    if lo == hi:
        return clean[lo]
    weight = pos - lo
    return clean[lo] * (1.0 - weight) + clean[hi] * weight


def mean_or_none(values: List[float]) -> Optional[float]:
    clean = [v for v in values if v is not None and math.isfinite(v)]
    if not clean:
        return None
    return sum(clean) / len(clean)


def parse_date(value: Any) -> Optional[dt.date]:
    text = clean_str(value)
    if not text:
        return None
    try:
        parsed = pd.to_datetime(text, errors="coerce")
        if pd.isna(parsed):
            return None
        return parsed.date()
    except Exception:
        return None


def elapsed_days(start: Any, end: Any, fallback: int) -> int:
    start_date = parse_date(start)
    end_date = parse_date(end)
    if start_date and end_date and end_date >= start_date:
        return max(1, (end_date - start_date).days)
    return max(1, int(fallback))


def clean_str(value: Any) -> str:
    if value is None:
        return ""
    try:
        if pd.isna(value):
            return ""
    except Exception:
        pass
    text = str(value).strip()
    if text.lower() in {"", "nan", "none", "null", "not_found", "not applicable", "n/a"}:
        return ""
    return text


def numeric_value(value: Any) -> Optional[float]:
    if value is None:
        return None
    try:
        if pd.isna(value):
            return None
    except Exception:
        pass
    if isinstance(value, (int, float)):
        return float(value)
    text = clean_str(value).replace(",", "")
    if not text:
        return None
    match = re.search(r"-?\d+(?:\.\d+)?", text)
    if not match:
        return None
    try:
        return float(match.group(0))
    except Exception:
        return None


def parse_money(value: Any) -> Optional[float]:
    number = numeric_value(value)
    if number is None:
        return None
    return number


def parse_exchange_ratio(value: Any) -> Optional[float]:
    text = clean_str(value).lower().replace(",", "")
    if not text:
        return None
    for pattern in [
        r"exchange ratio (?:of )?([0-9]+(?:\.[0-9]+)?)",
        r"([0-9]+(?:\.[0-9]+)?)\s*(?:share|shares)\s+of",
        r"\b([0-9]+(?:\.[0-9]+)?)\s*:\s*1\b",
        r"\b([0-9]+(?:\.[0-9]+)?)\b",
    ]:
        match = re.search(pattern, text)
        if match:
            return float(match.group(1))
    return numeric_value(value)


def parse_fraction(value: Any) -> Optional[float]:
    if value is None:
        return None
    try:
        if pd.isna(value):
            return None
    except Exception:
        pass
    if isinstance(value, (int, float)):
        x = float(value)
        if 0.0 <= x <= 1.0:
            return x
        if 1.0 < x <= 100.0:
            return x / 100.0
        return None
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
    whole = numeric_value(text)
    if whole is not None and 1.0 < whole <= 100.0 and any(w in text for w in ["percent", "pct"]):
        return whole / 100.0
    if any(w in text for w in ["one-half", "one half", "half of", "fifty percent"]):
        return 0.5
    return None


def bounded_share(value: Optional[float]) -> Optional[float]:
    if value is None or not math.isfinite(value):
        return None
    if value > 1.0 and value <= 100.0:
        value = value / 100.0
    return max(0.0, min(1.0, value))


def first_present(row: pd.Series, names: Iterable[str]) -> Any:
    for name in names:
        if name in row.index:
            value = row.get(name)
            if clean_str(value) or numeric_value(value) is not None:
                return value
    return None


def share_from_value(value: Any, denominator: Optional[float] = None) -> Optional[float]:
    text = clean_str(value)
    if not text and numeric_value(value) is None:
        return None
    if "%" in text:
        x = numeric_value(text)
        return bounded_share(x / 100.0 if x is not None else None)
    x = numeric_value(value)
    if x is None:
        return None
    if 0.0 <= x <= 1.0:
        return x
    if 1.0 < x <= 100.0 and any(w in text.lower() for w in ["percent", "pct", "%", "share"]):
        return x / 100.0
    if denominator and denominator > 0:
        return bounded_share(x / denominator)
    if 1.0 < x <= 100.0:
        return x / 100.0
    return None


def proration_factor_from_text(text: str, option: str) -> Optional[float]:
    text_l = clean_str(text).lower()
    if not text_l:
        return None
    option_l = option.lower()
    windows: List[str] = []
    for match in re.finditer(option_l, text_l):
        start = max(0, match.start() - 120)
        end = min(len(text_l), match.end() + 180)
        window = text_l[start:end]
        if "prorat" in window or "election" in window:
            windows.append(window)
    if not windows and option_l in text_l:
        windows = [text_l]
    for window in windows:
        pct = re.search(r"([0-9]+(?:\.[0-9]+)?)\s*%", window)
        if pct:
            return bounded_share(float(pct.group(1)) / 100.0)
        dec = re.search(r"\b0\.[0-9]+\b", window)
        if dec:
            return bounded_share(float(dec.group(0)))
    return None


def observed_from_final_proration(row: pd.Series) -> Tuple[Optional[float], Optional[float], str]:
    text = clean_str(first_present(row, ["final_proration_results", "preliminary_proration_results"]))
    if not text:
        return None, None, ""
    terms = event_terms(row)
    cash_fill = proration_factor_from_text(text, "cash")
    stock_fill = proration_factor_from_text(text, "stock")
    cash = None
    stock = None
    if cash_fill and cash_fill > EPS and terms.get("cash_cap") is not None:
        cash = bounded_share(terms["cash_cap"] / cash_fill)
        if cash is not None:
            stock = max(0.0, 1.0 - cash)
    if stock_fill and stock_fill > EPS and terms.get("stock_cap") is not None:
        stock = bounded_share(terms["stock_cap"] / stock_fill)
        if stock is not None:
            cash = max(0.0, 1.0 - stock)
    source = "final_proration_results_backed_out_from_cap" if cash is not None or stock is not None else ""
    return cash, stock, source


def observed_election_shares(row: pd.Series) -> Tuple[Optional[float], Optional[float], str]:
    shares_out = numeric_value(row.get("target_shares_outstanding"))
    cash = share_from_value(
        first_present(row, [
            "observed_cash_election_share",
            "realized_cash_share",
            "realized_cash_election_demand_num",
            "realized_cash_election_demand",
        ]),
        denominator=shares_out,
    )
    stock = share_from_value(
        first_present(row, [
            "observed_stock_election_share",
            "realized_stock_share",
            "realized_stock_election_demand_num",
            "realized_stock_election_demand",
        ]),
        denominator=shares_out,
    )
    source = ""
    if cash is not None:
        source = "direct_realized_cash_election_demand"
    if stock is not None and not source:
        source = "direct_realized_stock_election_demand"
    if cash is not None and stock is None:
        stock = max(0.0, 1.0 - cash)
    if stock is not None and cash is None:
        cash = max(0.0, 1.0 - stock)
    if cash is None and stock is None:
        cash, stock, source = observed_from_final_proration(row)
    return bounded_share(cash), bounded_share(stock), source


def cap_fraction_from_row(row: pd.Series, names: Iterable[str]) -> Optional[float]:
    raw = first_present(row, names)
    fraction = parse_fraction(raw)
    if fraction is not None:
        return fraction
    cap_shares = numeric_value(raw)
    shares_out = numeric_value(row.get("target_shares_outstanding"))
    if cap_shares is not None and shares_out and shares_out > 0 and cap_shares > 100.0:
        return bounded_share(cap_shares / shares_out)
    return None


def label_quality_weight(source: str) -> float:
    text = clean_str(source).lower()
    if text.startswith("direct_realized_cash") or text.startswith("direct_realized_stock"):
        return 1.0
    if "final_proration_results_backed_out" in text:
        return 0.65
    if "preliminary" in text:
        return 0.45
    return 0.0


def event_terms(row: pd.Series) -> Dict[str, Any]:
    cash = parse_money(first_present(row, ["cash_consideration_per_share_num", "cash_consideration_per_share"]))
    exchange_ratio = parse_exchange_ratio(first_present(row, ["exchange_ratio_num", "exchange_ratio"]))
    entry_target = numeric_value(first_present(row, ["entry_target_price", "target_price"]))
    entry_acquirer = numeric_value(first_present(row, ["entry_acquirer_price", "acquirer_price"]))
    exit_acquirer = numeric_value(first_present(row, ["exit_acquirer_price", "acquirer_price", "entry_acquirer_price"]))
    passive = bounded_share(numeric_value(first_present(row, ["passive_control_percent", "etf_ownership_percent"]))) or 0.0
    cash_cap = cap_fraction_from_row(row, ["cash_cap_fraction", "cash_cap"])
    stock_cap = cap_fraction_from_row(row, ["stock_cap_fraction", "stock_cap"])
    if cash_cap is None and stock_cap is not None:
        cash_cap = max(0.0, 1.0 - stock_cap)
    if stock_cap is None and cash_cap is not None:
        stock_cap = max(0.0, 1.0 - cash_cap)
    stock_value = None
    if exchange_ratio is not None and entry_acquirer is not None:
        stock_value = exchange_ratio * entry_acquirer
    return {
        "cash_value": cash,
        "exchange_ratio": exchange_ratio,
        "entry_target_price": entry_target,
        "entry_acquirer_price": entry_acquirer,
        "exit_acquirer_price": exit_acquirer,
        "entry_rule_date": first_present(row, ["entry_rule_date", "election_deadline"]),
        "exit_result_date": first_present(row, ["exit_result_date", "closing_date", "deal_completion_date"]),
        "stock_value": stock_value,
        "passive_share": passive,
        "cash_cap": bounded_share(cash_cap),
        "stock_cap": bounded_share(stock_cap),
        "default_rule": default_rule(row),
        "target_bid_ask_spread_pct": bounded_share(numeric_value(first_present(row, ["target_bid_ask_spread_pct", "target_spread_pct"]))),
        "acquirer_bid_ask_spread_pct": bounded_share(numeric_value(first_present(row, ["acquirer_bid_ask_spread_pct", "acquirer_spread_pct"]))),
        "short_volume_ratio": bounded_share(numeric_value(first_present(row, ["short_volume_ratio", "short_volume_ratio_20d_avg"]))),
        "deal_break_probability": bounded_share(numeric_value(first_present(row, ["deal_break_probability", "deal_break_prob", "p_break"]))),
        "deal_completed_probability": bounded_share(numeric_value(first_present(row, ["deal_completed_probability", "p_completed"]))),
        "deal_terminated_probability": bounded_share(numeric_value(first_present(row, ["deal_terminated_probability", "p_terminated"]))),
        "deal_withdrawn_probability": bounded_share(numeric_value(first_present(row, ["deal_withdrawn_probability", "p_withdrawn"]))),
        "deal_outcome_model_source": first_present(row, ["deal_outcome_model_source"]),
        "actual_deal_outcome": normalize_outcome_label(row) or clean_str(first_present(row, ["actual_deal_outcome"])),
        "break_price": numeric_value(first_present(row, ["break_price", "standalone_target_price", "unaffected_target_price"])),
        "terminated_break_price": numeric_value(first_present(row, ["terminated_break_price"])),
        "withdrawn_break_price": numeric_value(first_present(row, ["withdrawn_break_price"])),
    }


def default_rule(row: pd.Series) -> str:
    text = clean_str(first_present(row, ["non_election_default_rule", "default_rule"])).lower()
    if not text:
        return "mixed"
    if "cash" in text and "stock" not in text:
        return "cash"
    if "stock" in text and "cash" not in text:
        return "stock"
    if "mixed" in text or ("cash" in text and "stock" in text):
        return "mixed"
    return "mixed"


def passive_cash_fraction(default: str) -> float:
    if default == "cash":
        return 1.0
    if default == "stock":
        return 0.0
    return 0.5


def rational_action(cash_ev: float, stock_ev: float) -> str:
    if cash_ev < 0.0 and stock_ev < 0.0:
        return "exit"
    return "stock" if stock_ev >= cash_ev else "cash"


def predict_votes(row: pd.Series, p: float, q: float) -> Dict[str, Any]:
    terms = event_terms(row)
    missing = [
        key for key in ["cash_value", "stock_value", "entry_target_price", "cash_cap", "stock_cap"]
        if terms.get(key) is None
    ]
    base = {
        "model_status": "ok" if not missing else "blocked",
        "block_reason": "" if not missing else "missing_inputs:" + ",".join(missing),
        **terms,
        "p_hat": p,
        "q_hat": q,
    }
    if missing:
        return base

    cash_ev = terms["cash_value"] - terms["entry_target_price"]
    stock_ev = terms["stock_value"] - terms["entry_target_price"]
    action = rational_action(cash_ev, stock_ev)
    passive = terms["passive_share"]
    rational = max(0.0, min(q, 1.0 - passive))
    irrational = max(0.0, 1.0 - passive - rational)

    if action == "exit":
        base_non_rational = passive + irrational
        if base_non_rational > EPS:
            passive = passive + rational * passive / base_non_rational
            irrational = irrational + rational * irrational / base_non_rational
        else:
            irrational = 1.0
        rational = 0.0

    cash_demand = irrational * p + passive * passive_cash_fraction(terms["default_rule"])
    stock_demand = irrational * (1.0 - p) + passive * (1.0 - passive_cash_fraction(terms["default_rule"]))
    if action == "cash":
        cash_demand += rational
    elif action == "stock":
        stock_demand += rational

    total = cash_demand + stock_demand
    if total > EPS:
        cash_demand /= total
        stock_demand /= total

    base.update({
        "cash_ev_uncapped_per_share": cash_ev,
        "stock_ev_uncapped_per_share": stock_ev,
        "rational_action": action,
        "effective_rational_share": rational,
        "effective_irrational_share": irrational,
        "effective_passive_share": passive,
        "predicted_cash_demand_share": cash_demand,
        "predicted_stock_demand_share": stock_demand,
    })
    base.update(fill_rates(cash_demand, stock_demand, terms["cash_cap"], terms["stock_cap"], prefix="predicted_"))
    return base


def fill_rates(
    cash_demand: float,
    stock_demand: float,
    cash_cap: Optional[float],
    stock_cap: Optional[float],
    prefix: str = "",
) -> Dict[str, Optional[float]]:
    if cash_cap is None or stock_cap is None:
        return {f"{prefix}cash_fill_rate": None, f"{prefix}stock_fill_rate": None}
    cash_fill = 1.0 if cash_demand <= EPS else min(1.0, cash_cap / cash_demand)
    stock_fill = 1.0 if stock_demand <= EPS else min(1.0, stock_cap / stock_demand)
    return {f"{prefix}cash_fill_rate": cash_fill, f"{prefix}stock_fill_rate": stock_fill}


def receipt_mix(choice: str, cash_fill: Optional[float], stock_fill: Optional[float]) -> Tuple[Optional[float], Optional[float]]:
    if cash_fill is None or stock_fill is None:
        return None, None
    if choice == "cash":
        cash_fraction = cash_fill
        stock_fraction = 1.0 - cash_fill
    else:
        stock_fraction = stock_fill
        cash_fraction = 1.0 - stock_fill
    return cash_fraction, stock_fraction


def ev_for_choice(terms: Dict[str, Any], choice: str, cash_fill: float, stock_fill: float) -> Optional[float]:
    cash_fraction, stock_fraction = receipt_mix(choice, cash_fill, stock_fill)
    if cash_fraction is None or terms.get("cash_value") is None or terms.get("stock_value") is None:
        return None
    value = cash_fraction * terms["cash_value"] + stock_fraction * terms["stock_value"]
    return value - terms["entry_target_price"]


def hedge_shares_per_target(terms: Dict[str, Any], expected_stock_fraction: float, args: argparse.Namespace) -> float:
    entry_target = terms.get("entry_target_price")
    entry_acq = terms.get("entry_acquirer_price")
    ratio = terms.get("exchange_ratio") or 0.0
    policy = arg_value(args, "own_hedge_policy", "conversion_expected")
    if policy == "dollar_neutral":
        if entry_target is None or entry_acq is None or entry_acq <= 0:
            return 0.0
        return max(0.0, entry_target / entry_acq)
    if policy == "conversion_expected":
        return max(0.0, expected_stock_fraction * ratio)
    return 0.0


def holding_days_for_terms(terms: Dict[str, Any], args: argparse.Namespace) -> int:
    fallback = int(arg_value(args, "holding_period_days", 30))
    return elapsed_days(terms.get("entry_rule_date"), terms.get("exit_result_date"), fallback)


def trading_cost_per_target_share(
    terms: Dict[str, Any],
    expected_stock_fraction: float,
    args: argparse.Namespace,
) -> Tuple[float, float, float]:
    entry_target = terms.get("entry_target_price") or 0.0
    entry_acq = terms.get("entry_acquirer_price") or 0.0
    short_ratio = hedge_shares_per_target(terms, expected_stock_fraction, args)
    target_bps = float(arg_value(args, "trading_cost_bps", 0.0))
    acq_bps = arg_value(args, "acquirer_trading_cost_bps", None)
    if acq_bps is None:
        acq_bps = target_bps
    base_cost = entry_target * target_bps / 10000.0
    base_cost += short_ratio * entry_acq * float(acq_bps) / 10000.0

    bid_ask_cost = 0.0
    if bool(arg_value(args, "use_bid_ask_costs", True)):
        target_spread = terms.get("target_bid_ask_spread_pct")
        acq_spread = terms.get("acquirer_bid_ask_spread_pct")
        if target_spread is not None:
            bid_ask_cost += 0.5 * entry_target * float(target_spread)
        if acq_spread is not None:
            # Short entry plus cover is roughly two half-spreads.
            bid_ask_cost += short_ratio * entry_acq * float(acq_spread)
    return base_cost + bid_ask_cost, base_cost, bid_ask_cost


def borrow_cost_per_target_share(
    terms: Dict[str, Any],
    expected_stock_fraction: float,
    args: argparse.Namespace,
) -> float:
    entry_acq = terms.get("entry_acquirer_price") or 0.0
    short_ratio = hedge_shares_per_target(terms, expected_stock_fraction, args)
    annual_bps = float(arg_value(args, "annual_borrow_cost_bps", 0.0))
    days = holding_days_for_terms(terms, args)
    dynamic = short_ratio * entry_acq * annual_bps / 10000.0 * days / 365.0
    return dynamic + float(arg_value(args, "borrow_cost_per_share_constant", 0.0))


def net_alpha_for_choice(
    terms: Dict[str, Any],
    gross_alpha: Optional[float],
    expected_stock_fraction: float,
    args: argparse.Namespace,
) -> Dict[str, Optional[float]]:
    if gross_alpha is None:
        return {
            "net_alpha_per_share": None,
            "trading_cost_per_share": None,
            "borrow_cost_per_share": None,
            "total_cost_per_share": None,
            "hedge_ratio_acquirer_short_per_target": None,
        }
    trading_cost, fixed_cost, bid_ask_cost = trading_cost_per_target_share(terms, expected_stock_fraction, args)
    borrow_cost = borrow_cost_per_target_share(terms, expected_stock_fraction, args)
    total_cost = trading_cost + borrow_cost
    return {
        "net_alpha_per_share": gross_alpha - total_cost,
        "trading_cost_per_share": trading_cost,
        "fixed_trading_cost_per_share": fixed_cost,
        "bid_ask_cost_per_share": bid_ask_cost,
        "borrow_cost_per_share": borrow_cost,
        "total_cost_per_share": total_cost,
        "hedge_ratio_acquirer_short_per_target": hedge_shares_per_target(terms, expected_stock_fraction, args),
    }


def deal_outcome_probabilities(row: pd.Series, terms: Dict[str, Any], args: argparse.Namespace) -> Dict[str, float]:
    completed = terms.get("deal_completed_probability")
    terminated = terms.get("deal_terminated_probability")
    withdrawn = terms.get("deal_withdrawn_probability")

    if completed is None and terminated is None and withdrawn is None:
        p_break = terms.get("deal_break_probability")
        if p_break is None:
            p_break = float(arg_value(args, "deal_break_prob", 0.0))
        p_break = bounded_prob(float(p_break), 0.0) or 0.0
        withdrawn_share = bounded_prob(float(arg_value(args, "withdrawn_share_of_break_prob", 0.35)), 0.35) or 0.35
        completed = 1.0 - p_break
        withdrawn = p_break * withdrawn_share
        terminated = p_break - withdrawn

    probs = {
        "completed": bounded_prob(completed, 0.0) or 0.0,
        "terminated": bounded_prob(terminated, 0.0) or 0.0,
        "withdrawn": bounded_prob(withdrawn, 0.0) or 0.0,
    }
    total = sum(probs.values())
    if total <= EPS:
        return {"completed": 1.0, "terminated": 0.0, "withdrawn": 0.0}
    return {k: v / total for k, v in probs.items()}


def break_alpha_for_choice(
    terms: Dict[str, Any],
    expected_stock_fraction: float,
    args: argparse.Namespace,
    outcome: str = "terminated",
) -> Optional[float]:
    entry_target = terms.get("entry_target_price")
    entry_acq = terms.get("entry_acquirer_price")
    if entry_target is None or entry_acq is None:
        return None
    if outcome == "withdrawn":
        break_price = terms.get("withdrawn_break_price")
        loss_pct = float(arg_value(args, "withdrawn_break_loss_pct", arg_value(args, "break_loss_pct", 0.30)))
    else:
        break_price = terms.get("terminated_break_price")
        loss_pct = float(arg_value(args, "terminated_break_loss_pct", arg_value(args, "break_loss_pct", 0.30)))
    if break_price is None:
        break_price = terms.get("break_price")
    if break_price is None:
        break_price = entry_target * (1.0 - loss_pct)
    acq_break_price = entry_acq * (1.0 + float(arg_value(args, "acquirer_break_return_pct", 0.0)))
    short_ratio = hedge_shares_per_target(terms, expected_stock_fraction, args)
    gross = (break_price - entry_target) + short_ratio * (entry_acq - acq_break_price)
    costs = net_alpha_for_choice(terms, gross, expected_stock_fraction, args)
    return costs["net_alpha_per_share"]


def state_adjusted_net_alpha_for_choice(
    row: pd.Series,
    terms: Dict[str, Any],
    close_net_alpha: Optional[float],
    expected_stock_fraction: float,
    args: argparse.Namespace,
) -> Dict[str, Optional[float]]:
    if close_net_alpha is None:
        return {
            "state_adjusted_net_alpha_per_share": None,
            "deal_completed_probability": None,
            "deal_terminated_probability": None,
            "deal_withdrawn_probability": None,
            "terminated_net_alpha_per_share": None,
            "withdrawn_net_alpha_per_share": None,
        }
    probs = deal_outcome_probabilities(row, terms, args)
    terminated_alpha = break_alpha_for_choice(terms, expected_stock_fraction, args, outcome="terminated")
    withdrawn_alpha = break_alpha_for_choice(terms, expected_stock_fraction, args, outcome="withdrawn")
    terminated_component = 0.0 if terminated_alpha is None else probs["terminated"] * terminated_alpha
    withdrawn_component = 0.0 if withdrawn_alpha is None else probs["withdrawn"] * withdrawn_alpha
    adjusted = probs["completed"] * close_net_alpha + terminated_component + withdrawn_component
    return {
        "state_adjusted_net_alpha_per_share": adjusted,
        "deal_completed_probability": probs["completed"],
        "deal_terminated_probability": probs["terminated"],
        "deal_withdrawn_probability": probs["withdrawn"],
        "terminated_net_alpha_per_share": terminated_alpha,
        "withdrawn_net_alpha_per_share": withdrawn_alpha,
    }


def summarize_draws(values: List[float], prefix: str) -> Dict[str, Optional[float]]:
    p05 = quantile(values, 0.05)
    out: Dict[str, Optional[float]] = {
        f"{prefix}_mean": mean_or_none(values),
        f"{prefix}_p05": p05,
        f"{prefix}_p50": quantile(values, 0.50),
        f"{prefix}_p95": quantile(values, 0.95),
    }
    if p05 is None:
        out[f"{prefix}_cvar05"] = None
    else:
        tail = [v for v in values if v <= p05]
        out[f"{prefix}_cvar05"] = mean_or_none(tail)
    return out


def simulate_choice_distribution(
    row: pd.Series,
    terms: Dict[str, Any],
    pred: Dict[str, Any],
    choice: str,
    own_share: float,
    deterministic_expected_stock_fraction: float,
    args: argparse.Namespace,
) -> Dict[str, Any]:
    draws = int(arg_value(args, "mc_draws", 0))
    if draws <= 0:
        return {}

    mean_cash = bounded_prob(pred.get("predicted_cash_demand_share"), 0.5) or 0.5
    concentration = max(2.0, float(arg_value(args, "mc_demand_concentration", 75.0)))
    min_alpha = max(0.01, float(arg_value(args, "mc_min_beta_alpha", 0.25)))
    a = max(min_alpha, mean_cash * concentration)
    b = max(min_alpha, (1.0 - mean_cash) * concentration)
    outcome_probs = deal_outcome_probabilities(row, terms, args)
    terminated_alpha = break_alpha_for_choice(terms, deterministic_expected_stock_fraction, args, outcome="terminated")
    withdrawn_alpha = break_alpha_for_choice(terms, deterministic_expected_stock_fraction, args, outcome="withdrawn")

    rng = random.Random(stable_seed(int(arg_value(args, "mc_seed", 1729)), row.get("event_id", ""), choice))
    net_alphas: List[float] = []
    gross_alphas: List[float] = []
    cash_fills: List[float] = []
    stock_fills: List[float] = []
    cash_demands: List[float] = []
    stock_demands: List[float] = []
    expected_cash_fractions: List[float] = []
    expected_stock_fractions: List[float] = []

    for _ in range(draws):
        outcome_draw = rng.random()
        if outcome_draw < outcome_probs["terminated"] and terminated_alpha is not None:
            net_alphas.append(float(terminated_alpha))
            continue
        if outcome_draw < outcome_probs["terminated"] + outcome_probs["withdrawn"] and withdrawn_alpha is not None:
            net_alphas.append(float(withdrawn_alpha))
            continue

        cash_sample = rng.betavariate(a, b)
        stock_sample = 1.0 - cash_sample
        cash_d, stock_d = demands_with_own_vote(cash_sample, stock_sample, own_share, choice)
        fills = fill_rates(cash_d, stock_d, terms["cash_cap"], terms["stock_cap"], prefix="")
        gross = ev_for_choice(terms, choice, fills["cash_fill_rate"], fills["stock_fill_rate"])
        cash_fraction, stock_fraction = receipt_mix(choice, fills["cash_fill_rate"], fills["stock_fill_rate"])
        if gross is None or cash_fraction is None or stock_fraction is None:
            continue
        costs = net_alpha_for_choice(terms, gross, stock_fraction, args)
        net = costs["net_alpha_per_share"]
        if net is None:
            continue
        net_alphas.append(float(net))
        gross_alphas.append(float(gross))
        cash_fills.append(float(fills["cash_fill_rate"]))
        stock_fills.append(float(fills["stock_fill_rate"]))
        cash_demands.append(float(cash_d))
        stock_demands.append(float(stock_d))
        expected_cash_fractions.append(float(cash_fraction))
        expected_stock_fractions.append(float(stock_fraction))

    out: Dict[str, Any] = {
        "mc_draws": draws,
        "mc_effective_draws": len(net_alphas),
        "mc_cash_demand_mean_input": mean_cash,
        "mc_demand_concentration": concentration,
        "mc_deal_completed_probability": outcome_probs["completed"],
        "mc_deal_terminated_probability": outcome_probs["terminated"],
        "mc_deal_withdrawn_probability": outcome_probs["withdrawn"],
        "mc_deal_break_probability": outcome_probs["terminated"] + outcome_probs["withdrawn"],
        "mc_terminated_net_alpha_per_share": terminated_alpha,
        "mc_withdrawn_net_alpha_per_share": withdrawn_alpha,
        "mc_loss_probability": (
            sum(1 for v in net_alphas if v < 0.0) / len(net_alphas) if net_alphas else None
        ),
    }
    out.update(summarize_draws(net_alphas, "mc_net_alpha_per_share"))
    out.update(summarize_draws(gross_alphas, "mc_gross_alpha_per_share"))
    out.update(summarize_draws(cash_fills, "mc_cash_fill_rate"))
    out.update(summarize_draws(stock_fills, "mc_stock_fill_rate"))
    out.update(summarize_draws(cash_demands, "mc_cash_demand_share"))
    out.update(summarize_draws(stock_demands, "mc_stock_demand_share"))
    out.update(summarize_draws(expected_cash_fractions, "mc_expected_cash_fraction"))
    out.update(summarize_draws(expected_stock_fractions, "mc_expected_stock_fraction"))
    return out


def own_vote_share(row: pd.Series, notional: float, entry_target_price: Optional[float]) -> float:
    shares_out = numeric_value(row.get("target_shares_outstanding"))
    if shares_out is None or shares_out <= 0 or entry_target_price is None or entry_target_price <= 0:
        return 0.0
    return max(0.0, min(0.25, (notional / entry_target_price) / shares_out))


def demands_with_own_vote(cash_demand: float, stock_demand: float, own_share: float, choice: str) -> Tuple[float, float]:
    own_share = max(0.0, min(1.0, own_share))
    cash = cash_demand * (1.0 - own_share)
    stock = stock_demand * (1.0 - own_share)
    if choice == "cash":
        cash += own_share
    else:
        stock += own_share
    total = cash + stock
    if total > EPS:
        return cash / total, stock / total
    return cash, stock


def fit_p_q(
    rows: List[pd.Series],
    p_grid_size: int,
    q_grid_size: int,
    default_p: float,
    default_q: float,
) -> Dict[str, Any]:
    usable: List[Tuple[pd.Series, float, float, str]] = []
    for row in rows:
        y, _, source = observed_election_shares(row)
        if y is not None:
            pred0 = predict_votes(row, default_p, default_q)
            if pred0.get("model_status") == "ok":
                weight = label_quality_weight(source)
                if weight > 0:
                    usable.append((row, y, weight, source))
    if not usable:
        return {
            "p_hat": default_p,
            "q_hat": default_q,
            "fit_n": 0,
            "fit_effective_n": 0.0,
            "fit_sse": None,
            "fit_loglike": None,
        }

    best = {"p_hat": default_p, "q_hat": default_q, "fit_sse": float("inf")}
    p_den = max(1, p_grid_size - 1)
    q_den = max(1, q_grid_size - 1)
    for ip in range(p_grid_size):
        p = ip / p_den
        for iq in range(q_grid_size):
            q = iq / q_den
            sse = 0.0
            ok = 0
            total_weight = 0.0
            for row, y, weight, _ in usable:
                pred = predict_votes(row, p, q)
                mu = pred.get("predicted_cash_demand_share")
                if mu is None:
                    continue
                sse += weight * (float(y) - float(mu)) ** 2
                total_weight += weight
                ok += 1
            if ok and sse < best["fit_sse"]:
                best = {"p_hat": p, "q_hat": q, "fit_sse": sse}
    n = len(usable)
    effective_n = sum(weight for _, _, weight, _ in usable)
    sigma2 = max(best["fit_sse"] / max(EPS, effective_n), 0.01 ** 2)
    loglike = -0.5 * effective_n * (math.log(2.0 * math.pi * sigma2) + 1.0)
    best.update({"fit_n": n, "fit_effective_n": effective_n, "fit_loglike": loglike})
    return best


def realized_pnl(
    row: pd.Series,
    terms: Dict[str, Any],
    choice: str,
    cash_fill: float,
    stock_fill: float,
    expected_stock_fraction: float,
    args: argparse.Namespace,
) -> Optional[float]:
    entry_target = terms.get("entry_target_price")
    entry_acq = terms.get("entry_acquirer_price")
    exit_acq = terms.get("exit_acquirer_price")
    ratio = terms.get("exchange_ratio")
    cash_value = terms.get("cash_value")
    if any(x is None for x in [entry_target, entry_acq, exit_acq, ratio, cash_value]):
        return None
    if entry_target <= 0 or entry_acq <= 0:
        return None
    cash_fraction, stock_fraction = receipt_mix(choice, cash_fill, stock_fill)
    if cash_fraction is None:
        return None
    target_shares = args.trade_notional / entry_target
    if args.own_hedge_policy == "dollar_neutral":
        short_acquirer_shares = args.trade_notional / entry_acq
    elif args.own_hedge_policy == "conversion_expected":
        short_acquirer_shares = target_shares * expected_stock_fraction * ratio
    else:
        short_acquirer_shares = 0.0
    cash_received = target_shares * cash_fraction * cash_value
    acq_shares_received = target_shares * stock_fraction * ratio
    short_pnl = short_acquirer_shares * (entry_acq - exit_acq)
    gross_pnl = cash_received + acq_shares_received * exit_acq - args.trade_notional + short_pnl
    per_share_cost = net_alpha_for_choice(terms, 0.0, expected_stock_fraction, args)["total_cost_per_share"] or 0.0
    return gross_pnl - target_shares * per_share_cost


def realized_break_pnl(
    row: pd.Series,
    terms: Dict[str, Any],
    outcome: str,
    expected_stock_fraction: float,
    args: argparse.Namespace,
) -> Optional[float]:
    entry_target = terms.get("entry_target_price")
    if entry_target is None or entry_target <= 0:
        return None
    alpha = break_alpha_for_choice(terms, expected_stock_fraction, args, outcome=outcome)
    if alpha is None:
        return None
    target_shares = args.trade_notional / entry_target
    return target_shares * alpha


def choice_decision_value(choice_row: Dict[str, Any], args: argparse.Namespace) -> Optional[float]:
    metric = str(arg_value(args, "trade_decision_metric", "mean")).lower()
    fallback = choice_row.get("state_adjusted_net_alpha_per_share")
    if fallback is None:
        fallback = choice_row.get("deterministic_net_alpha_per_share")
    if metric == "deterministic":
        return fallback
    if metric == "p05":
        value = choice_row.get("mc_net_alpha_per_share_p05")
        return value if value is not None else fallback
    if metric == "cvar05":
        value = choice_row.get("mc_net_alpha_per_share_cvar05")
        return value if value is not None else fallback
    value = choice_row.get("mc_net_alpha_per_share_mean")
    return value if value is not None else fallback


def choose_trade_and_backtest(row: pd.Series, pred: Dict[str, Any], args: argparse.Namespace) -> Dict[str, Any]:
    if pred.get("model_status") != "ok":
        return {
            "strategy_signal": "no_trade",
            "trade_reason": pred.get("block_reason", "blocked"),
        }
    terms = pred
    own = own_vote_share(row, args.trade_notional, terms.get("entry_target_price"))
    choices: Dict[str, Dict[str, Any]] = {}
    for choice in ["cash", "stock"]:
        cash_d, stock_d = demands_with_own_vote(
            pred["predicted_cash_demand_share"],
            pred["predicted_stock_demand_share"],
            own,
            choice,
        )
        fills = fill_rates(cash_d, stock_d, terms["cash_cap"], terms["stock_cap"], prefix="")
        gross_alpha = ev_for_choice(terms, choice, fills["cash_fill_rate"], fills["stock_fill_rate"])
        cash_fraction, stock_fraction = receipt_mix(choice, fills["cash_fill_rate"], fills["stock_fill_rate"])
        expected_stock_fraction = stock_fraction or 0.0
        costs = net_alpha_for_choice(terms, gross_alpha, expected_stock_fraction, args)
        state_adjusted = state_adjusted_net_alpha_for_choice(
            row,
            terms,
            costs["net_alpha_per_share"],
            expected_stock_fraction,
            args,
        )
        choices[choice] = {
            "cash_demand_with_own": cash_d,
            "stock_demand_with_own": stock_d,
            "cash_fill": fills["cash_fill_rate"],
            "stock_fill": fills["stock_fill_rate"],
            # Backward-compatible column name: this is alpha over entry target price.
            "cap_adjusted_ev_per_share": gross_alpha,
            "deterministic_gross_alpha_per_share": gross_alpha,
            "deterministic_net_alpha_per_share": costs["net_alpha_per_share"],
            "deterministic_trading_cost_per_share": costs["trading_cost_per_share"],
            "deterministic_fixed_trading_cost_per_share": costs["fixed_trading_cost_per_share"],
            "deterministic_bid_ask_cost_per_share": costs["bid_ask_cost_per_share"],
            "deterministic_borrow_cost_per_share": costs["borrow_cost_per_share"],
            "deterministic_total_cost_per_share": costs["total_cost_per_share"],
            "hedge_ratio_acquirer_short_per_target": costs["hedge_ratio_acquirer_short_per_target"],
            **state_adjusted,
            "expected_cash_fraction": cash_fraction,
            "expected_stock_fraction": stock_fraction,
        }
        mc = simulate_choice_distribution(
            row=row,
            terms=terms,
            pred=pred,
            choice=choice,
            own_share=own,
            deterministic_expected_stock_fraction=expected_stock_fraction,
            args=args,
        )
        choices[choice].update(mc)

    best_choice = max(
        choices,
        key=lambda c: choice_decision_value(choices[c], args)
        if choice_decision_value(choices[c], args) is not None
        else NEG_INF,
    )
    best_ev = choices[best_choice]["cap_adjusted_ev_per_share"]
    best_net = choices[best_choice].get("deterministic_net_alpha_per_share")
    best_decision_value = choice_decision_value(choices[best_choice], args)
    best_p05 = choices[best_choice].get("mc_net_alpha_per_share_p05")
    best_cvar05 = choices[best_choice].get("mc_net_alpha_per_share_cvar05")
    best_loss_probability = choices[best_choice].get("mc_loss_probability")
    min_p05 = float(arg_value(args, "min_p05_net_alpha", NEG_INF))
    max_loss_prob = float(arg_value(args, "max_loss_probability", 1.0))
    if pred["cash_ev_uncapped_per_share"] < 0.0 and pred["stock_ev_uncapped_per_share"] < 0.0:
        signal = "no_trade"
        reason = "both_uncapped_evs_negative_rational_exit"
    elif best_decision_value is None or best_decision_value < args.min_net_alpha:
        signal = "no_trade"
        reason = "risk_adjusted_alpha_below_threshold"
    elif best_p05 is not None and best_p05 < min_p05:
        signal = "no_trade"
        reason = "mc_p05_alpha_below_threshold"
    elif best_loss_probability is not None and best_loss_probability > max_loss_prob:
        signal = "no_trade"
        reason = "mc_loss_probability_above_threshold"
    else:
        signal = f"trade_{best_choice}_election"
        reason = "risk_adjusted_alpha_positive"

    actual_deal_outcome = clean_str(terms.get("actual_deal_outcome")).lower()
    outcome_label_available = actual_deal_outcome in {"completed", "terminated", "withdrawn"}
    is_break_outcome = actual_deal_outcome in {"terminated", "withdrawn"}
    obs_cash, obs_stock, obs_source = observed_election_shares(row)
    realized_election_label_available = bool(obs_source and obs_cash is not None and obs_stock is not None)
    realized_label_available = bool(is_break_outcome or realized_election_label_available)
    actual_source = obs_source or "predicted_proxy_no_realized_label"
    if is_break_outcome:
        actual_source = f"deal_outcome_{actual_deal_outcome}_break_scenario"
    if obs_cash is None or obs_stock is None:
        obs_cash = pred["predicted_cash_demand_share"]
        obs_stock = pred["predicted_stock_demand_share"]

    pnl_by_choice: Dict[str, Dict[str, Any]] = {}
    for choice in ["cash", "stock"]:
        expected_stock_fraction = choices[choice]["expected_stock_fraction"] or 0.0
        if is_break_outcome:
            cash_d = None
            stock_d = None
            fills = {"cash_fill_rate": None, "stock_fill_rate": None}
            pnl = realized_break_pnl(row, terms, actual_deal_outcome, expected_stock_fraction, args)
        else:
            cash_d, stock_d = demands_with_own_vote(obs_cash, obs_stock, own if signal.startswith("trade") else 0.0, choice)
            fills = fill_rates(cash_d, stock_d, terms["cash_cap"], terms["stock_cap"], prefix="")
            pnl = realized_pnl(row, terms, choice, fills["cash_fill_rate"], fills["stock_fill_rate"], expected_stock_fraction, args)
        pnl_by_choice[choice] = {
            "actual_cash_demand_with_own": cash_d,
            "actual_stock_demand_with_own": stock_d,
            "actual_cash_fill": fills["cash_fill_rate"],
            "actual_stock_fill": fills["stock_fill_rate"],
            "pnl": pnl,
            "realized_pnl": pnl,
        }

    executed_choice = best_choice if signal.startswith("trade") else ""
    executed_pnl_or_proxy = pnl_by_choice[best_choice]["pnl"] if executed_choice else None
    best_pnl_choice = max(
        pnl_by_choice,
        key=lambda c: pnl_by_choice[c]["pnl"] if pnl_by_choice[c]["pnl"] is not None else NEG_INF,
    )
    best_pnl = pnl_by_choice[best_pnl_choice]["pnl"]
    executed_realized_pnl = executed_pnl_or_proxy if realized_label_available else None
    executed_proxy_pnl = executed_pnl_or_proxy if not realized_label_available else None
    best_realized_pnl = best_pnl if realized_label_available else None
    best_proxy_pnl = best_pnl if not realized_label_available else None
    missed = bool(realized_label_available and (not signal.startswith("trade")) and best_realized_pnl is not None and best_realized_pnl > 0.0)
    loss = bool(realized_label_available and signal.startswith("trade") and executed_realized_pnl is not None and executed_realized_pnl < 0.0)
    proxy_missed = bool((not realized_label_available) and (not signal.startswith("trade")) and best_proxy_pnl is not None and best_proxy_pnl > 0.0)
    proxy_loss = bool((not realized_label_available) and signal.startswith("trade") and executed_proxy_pnl is not None and executed_proxy_pnl < 0.0)

    out = {
        "strategy_signal": signal,
        "trade_reason": reason,
        "own_vote_share": own,
        "chosen_election": executed_choice,
        "expected_ev_per_share": best_ev,
        "expected_net_alpha_per_share": best_net,
        "trade_decision_metric": str(arg_value(args, "trade_decision_metric", "mean")),
        "trade_decision_value": best_decision_value,
        "trade_decision_p05_net_alpha": best_p05,
        "trade_decision_cvar05_net_alpha": best_cvar05,
        "trade_decision_loss_probability": best_loss_probability,
        "expected_pnl_notional": best_decision_value * args.trade_notional / terms["entry_target_price"]
        if best_decision_value is not None and terms["entry_target_price"]
        else None,
        "realized_label_available": realized_label_available,
        "realized_election_label_available": realized_election_label_available,
        "deal_outcome_label_available": outcome_label_available,
        "actual_deal_outcome": actual_deal_outcome,
        "realized_label_quality_weight": label_quality_weight(obs_source),
        "realized_label_source": actual_source,
        "observed_cash_demand_share": obs_cash,
        "observed_stock_demand_share": obs_stock,
        "executed_realized_pnl": executed_realized_pnl,
        "executed_proxy_pnl": executed_proxy_pnl,
        "best_realized_choice": best_pnl_choice if realized_label_available else "",
        "best_realized_pnl": best_realized_pnl,
        "best_proxy_choice": best_pnl_choice if not realized_label_available else "",
        "best_proxy_pnl": best_proxy_pnl,
        "missed_arbitrage": missed,
        "loss_trade": loss,
        "proxy_missed_arbitrage": proxy_missed,
        "proxy_loss_trade": proxy_loss,
    }
    for choice, vals in choices.items():
        out.update({f"{choice}_{k}": v for k, v in vals.items()})
    for choice, vals in pnl_by_choice.items():
        out.update({f"{choice}_{k}": v for k, v in vals.items()})
    return out


def sort_panel(panel: pd.DataFrame) -> pd.DataFrame:
    out = panel.copy()
    date_col = next((c for c in ["Announce Date Parsed", "announce_date", "Announce Date"] if c in out.columns), None)
    if date_col:
        out["_sort_date"] = pd.to_datetime(out[date_col], errors="coerce")
    else:
        out["_sort_date"] = pd.NaT
    out["_sort_pos"] = range(len(out))
    return out.sort_values(["_sort_date", "_sort_pos"], na_position="last").reset_index(drop=True)


def run_structural_backtest(panel: pd.DataFrame, args: argparse.Namespace) -> Tuple[pd.DataFrame, pd.DataFrame, Dict[str, Any]]:
    ordered = sort_panel(panel)
    parameter_rows: List[Dict[str, Any]] = []
    prediction_rows: List[Dict[str, Any]] = []
    historical_rows: List[pd.Series] = []
    historical_outcome_rows: List[pd.Series] = []

    for i, row in ordered.iterrows():
        train_rows = historical_rows[-args.rolling_window_events:] if args.rolling_window_events > 0 else historical_rows
        outcome_train_rows = (
            historical_outcome_rows[-arg_value(args, "outcome_rolling_window_events", 250):]
            if int(arg_value(args, "outcome_rolling_window_events", 250)) > 0
            else historical_outcome_rows
        )
        fit = fit_p_q(
            train_rows if len(train_rows) >= args.min_fit_events else [],
            args.p_grid_size,
            args.q_grid_size,
            args.default_irrational_cash_prob,
            args.default_rational_share,
        )
        pred = predict_votes(row, fit["p_hat"], fit["q_hat"])
        outcome = {}
        if bool(arg_value(args, "enable_deal_outcome_model", True)):
            outcome = outcome_probability_row(
                row=row,
                train_rows=outcome_train_rows,
                min_fit_events=int(arg_value(args, "outcome_min_fit_events", 25)),
                default_completed_prob=float(arg_value(args, "default_completed_prob", 0.90)),
                default_terminated_prob=float(arg_value(args, "default_terminated_prob", 0.07)),
                default_withdrawn_prob=float(arg_value(args, "default_withdrawn_prob", 0.03)),
            )
            pred.update(outcome)
        trade = choose_trade_and_backtest(row, pred, args)
        event_id = row.get("event_id", f"row_{i}")
        parameter_rows.append({
            "event_id": event_id,
            "rolling_event_index": i,
            **fit,
            **outcome,
        })
        prediction_rows.append({
            "event_id": event_id,
            "target_name": row.get("Target Name", row.get("target_name", "")),
            "acquirer_name": row.get("Acquirer Name", row.get("acquirer_name", "")),
            "announce_date": row.get("Announce Date", row.get("announce_date", "")),
            **pred,
            **trade,
        })
        obs_cash, _, _ = observed_election_shares(row)
        if obs_cash is not None:
            historical_rows.append(row)
        if normalize_outcome_label(row) in {"completed", "terminated", "withdrawn"}:
            historical_outcome_rows.append(row)

    predictions = pd.DataFrame(prediction_rows)
    parameters = pd.DataFrame(parameter_rows)
    traded = predictions["strategy_signal"].astype(str).str.startswith("trade") if not predictions.empty else pd.Series(dtype=bool)
    label_available = predictions.get("realized_label_available", pd.Series(dtype=bool)).fillna(False).astype(bool) if not predictions.empty else pd.Series(dtype=bool)
    executed_realized_pnl = pd.to_numeric(predictions.get("executed_realized_pnl"), errors="coerce") if not predictions.empty else pd.Series(dtype=float)
    executed_proxy_pnl = pd.to_numeric(predictions.get("executed_proxy_pnl"), errors="coerce") if not predictions.empty else pd.Series(dtype=float)
    realized_trades = traded & label_available if not predictions.empty else pd.Series(dtype=bool)
    proxy_trades = traded & ~label_available if not predictions.empty else pd.Series(dtype=bool)
    summary = {
        "event_count": int(len(predictions)),
        "fit_event_count": int(parameters["fit_n"].gt(0).sum()) if not parameters.empty and "fit_n" in parameters else 0,
        "labeled_event_count": int(label_available.sum()) if not predictions.empty else 0,
        "trade_count": int(traded.sum()) if not predictions.empty else 0,
        "realized_label_trade_count": int(realized_trades.sum()) if not predictions.empty else 0,
        "proxy_trade_count": int(proxy_trades.sum()) if not predictions.empty else 0,
        "missed_arbitrage_count": int(predictions.get("missed_arbitrage", pd.Series(dtype=bool)).fillna(False).sum()) if not predictions.empty else 0,
        "loss_trade_count": int(predictions.get("loss_trade", pd.Series(dtype=bool)).fillna(False).sum()) if not predictions.empty else 0,
        "proxy_missed_arbitrage_count": int(predictions.get("proxy_missed_arbitrage", pd.Series(dtype=bool)).fillna(False).sum()) if not predictions.empty else 0,
        "proxy_loss_trade_count": int(predictions.get("proxy_loss_trade", pd.Series(dtype=bool)).fillna(False).sum()) if not predictions.empty else 0,
        "average_realized_pnl_per_labeled_trade": float(executed_realized_pnl[realized_trades].mean()) if not predictions.empty and realized_trades.any() else None,
        "total_realized_pnl_labeled_trades": float(executed_realized_pnl[realized_trades].sum()) if not predictions.empty and realized_trades.any() else 0.0,
        "average_proxy_pnl_per_proxy_trade": float(executed_proxy_pnl[proxy_trades].mean()) if not predictions.empty and proxy_trades.any() else None,
        "total_proxy_pnl_proxy_trades": float(executed_proxy_pnl[proxy_trades].sum()) if not predictions.empty and proxy_trades.any() else 0.0,
        "trade_notional": float(args.trade_notional),
        "borrow_cost_assumption": {
            "borrow_cost_per_share_constant": float(arg_value(args, "borrow_cost_per_share_constant", 0.0)),
            "annual_borrow_cost_bps": float(arg_value(args, "annual_borrow_cost_bps", 0.0)),
        },
        "hedge_policy": args.own_hedge_policy,
        "trade_decision_metric": str(arg_value(args, "trade_decision_metric", "mean")),
        "mc_draws": int(arg_value(args, "mc_draws", 0)),
        "deal_outcome_model_enabled": bool(arg_value(args, "enable_deal_outcome_model", True)),
    }
    return predictions, parameters, summary


def write_structural_summary(path: Any, summary: Dict[str, Any]) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
