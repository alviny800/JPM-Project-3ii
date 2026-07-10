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
import json
import math
import re
from typing import Any, Dict, Iterable, List, Optional, Tuple

import pandas as pd


EPS = 1e-12


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
        source = "realized_cash_election_demand"
    if stock is not None and not source:
        source = "realized_stock_election_demand"
    if cash is not None and stock is None:
        stock = max(0.0, 1.0 - cash)
    if stock is not None and cash is None:
        cash = max(0.0, 1.0 - stock)
    if cash is None and stock is None:
        cash, stock, source = observed_from_final_proration(row)
    return bounded_share(cash), bounded_share(stock), source


def event_terms(row: pd.Series) -> Dict[str, Any]:
    cash = parse_money(first_present(row, ["cash_consideration_per_share_num", "cash_consideration_per_share"]))
    exchange_ratio = parse_exchange_ratio(first_present(row, ["exchange_ratio_num", "exchange_ratio"]))
    entry_target = numeric_value(first_present(row, ["entry_target_price", "target_price"]))
    entry_acquirer = numeric_value(first_present(row, ["entry_acquirer_price", "acquirer_price"]))
    exit_acquirer = numeric_value(first_present(row, ["exit_acquirer_price", "acquirer_price", "entry_acquirer_price"]))
    passive = bounded_share(numeric_value(first_present(row, ["passive_control_percent", "etf_ownership_percent"]))) or 0.0
    cash_cap = parse_fraction(first_present(row, ["cash_cap_fraction", "cash_cap"]))
    stock_cap = parse_fraction(first_present(row, ["stock_cap_fraction", "stock_cap"]))
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
        "stock_value": stock_value,
        "passive_share": passive,
        "cash_cap": bounded_share(cash_cap),
        "stock_cap": bounded_share(stock_cap),
        "default_rule": default_rule(row),
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
    usable: List[Tuple[pd.Series, float]] = []
    for row in rows:
        y, _, _ = observed_election_shares(row)
        if y is not None:
            pred0 = predict_votes(row, default_p, default_q)
            if pred0.get("model_status") == "ok":
                usable.append((row, y))
    if not usable:
        return {"p_hat": default_p, "q_hat": default_q, "fit_n": 0, "fit_sse": None, "fit_loglike": None}

    best = {"p_hat": default_p, "q_hat": default_q, "fit_sse": float("inf")}
    p_den = max(1, p_grid_size - 1)
    q_den = max(1, q_grid_size - 1)
    for ip in range(p_grid_size):
        p = ip / p_den
        for iq in range(q_grid_size):
            q = iq / q_den
            sse = 0.0
            ok = 0
            for row, y in usable:
                pred = predict_votes(row, p, q)
                mu = pred.get("predicted_cash_demand_share")
                if mu is None:
                    continue
                sse += (float(y) - float(mu)) ** 2
                ok += 1
            if ok and sse < best["fit_sse"]:
                best = {"p_hat": p, "q_hat": q, "fit_sse": sse}
    n = len(usable)
    sigma2 = max(best["fit_sse"] / max(1, n), 0.01 ** 2)
    loglike = -0.5 * n * (math.log(2.0 * math.pi * sigma2) + 1.0)
    best.update({"fit_n": n, "fit_loglike": loglike})
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
    return cash_received + acq_shares_received * exit_acq - args.trade_notional + short_pnl


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
        ev = ev_for_choice(terms, choice, fills["cash_fill_rate"], fills["stock_fill_rate"])
        cash_fraction, stock_fraction = receipt_mix(choice, fills["cash_fill_rate"], fills["stock_fill_rate"])
        choices[choice] = {
            "cash_demand_with_own": cash_d,
            "stock_demand_with_own": stock_d,
            "cash_fill": fills["cash_fill_rate"],
            "stock_fill": fills["stock_fill_rate"],
            "cap_adjusted_ev_per_share": ev,
            "expected_cash_fraction": cash_fraction,
            "expected_stock_fraction": stock_fraction,
        }

    best_choice = max(choices, key=lambda c: choices[c]["cap_adjusted_ev_per_share"] if choices[c]["cap_adjusted_ev_per_share"] is not None else -1e9)
    best_ev = choices[best_choice]["cap_adjusted_ev_per_share"]
    if pred["cash_ev_uncapped_per_share"] < 0.0 and pred["stock_ev_uncapped_per_share"] < 0.0:
        signal = "no_trade"
        reason = "both_uncapped_evs_negative_rational_exit"
    elif best_ev is None or best_ev < args.min_net_alpha:
        signal = "no_trade"
        reason = "cap_adjusted_ev_below_threshold"
    else:
        signal = f"trade_{best_choice}_election"
        reason = "cap_adjusted_ev_positive"

    obs_cash, obs_stock, obs_source = observed_election_shares(row)
    actual_source = obs_source or "predicted_proxy_no_realized_label"
    if obs_cash is None or obs_stock is None:
        obs_cash = pred["predicted_cash_demand_share"]
        obs_stock = pred["predicted_stock_demand_share"]

    realized_by_choice: Dict[str, Dict[str, Any]] = {}
    for choice in ["cash", "stock"]:
        cash_d, stock_d = demands_with_own_vote(obs_cash, obs_stock, own if signal.startswith("trade") else 0.0, choice)
        fills = fill_rates(cash_d, stock_d, terms["cash_cap"], terms["stock_cap"], prefix="")
        expected_stock_fraction = choices[choice]["expected_stock_fraction"] or 0.0
        pnl = realized_pnl(row, terms, choice, fills["cash_fill_rate"], fills["stock_fill_rate"], expected_stock_fraction, args)
        realized_by_choice[choice] = {
            "actual_cash_demand_with_own": cash_d,
            "actual_stock_demand_with_own": stock_d,
            "actual_cash_fill": fills["cash_fill_rate"],
            "actual_stock_fill": fills["stock_fill_rate"],
            "realized_pnl": pnl,
        }

    executed_choice = best_choice if signal.startswith("trade") else ""
    executed_pnl = realized_by_choice[best_choice]["realized_pnl"] if executed_choice else None
    best_realized_choice = max(
        realized_by_choice,
        key=lambda c: realized_by_choice[c]["realized_pnl"] if realized_by_choice[c]["realized_pnl"] is not None else -1e18,
    )
    best_realized_pnl = realized_by_choice[best_realized_choice]["realized_pnl"]
    missed = bool((not signal.startswith("trade")) and best_realized_pnl is not None and best_realized_pnl > 0.0)
    loss = bool(signal.startswith("trade") and executed_pnl is not None and executed_pnl < 0.0)

    out = {
        "strategy_signal": signal,
        "trade_reason": reason,
        "own_vote_share": own,
        "chosen_election": executed_choice,
        "expected_ev_per_share": best_ev,
        "expected_pnl_notional": best_ev * args.trade_notional / terms["entry_target_price"] if best_ev is not None and terms["entry_target_price"] else None,
        "realized_label_source": actual_source,
        "observed_cash_demand_share": obs_cash,
        "observed_stock_demand_share": obs_stock,
        "executed_realized_pnl": executed_pnl,
        "best_realized_choice": best_realized_choice,
        "best_realized_pnl": best_realized_pnl,
        "missed_arbitrage": missed,
        "loss_trade": loss,
    }
    for choice, vals in choices.items():
        out.update({f"{choice}_{k}": v for k, v in vals.items()})
    for choice, vals in realized_by_choice.items():
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

    for i, row in ordered.iterrows():
        train_rows = historical_rows[-args.rolling_window_events:] if args.rolling_window_events > 0 else historical_rows
        fit = fit_p_q(
            train_rows if len(train_rows) >= args.min_fit_events else [],
            args.p_grid_size,
            args.q_grid_size,
            args.default_irrational_cash_prob,
            args.default_rational_share,
        )
        pred = predict_votes(row, fit["p_hat"], fit["q_hat"])
        trade = choose_trade_and_backtest(row, pred, args)
        event_id = row.get("event_id", f"row_{i}")
        parameter_rows.append({
            "event_id": event_id,
            "rolling_event_index": i,
            **fit,
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

    predictions = pd.DataFrame(prediction_rows)
    parameters = pd.DataFrame(parameter_rows)
    traded = predictions["strategy_signal"].astype(str).str.startswith("trade") if not predictions.empty else pd.Series(dtype=bool)
    executed_pnl = pd.to_numeric(predictions.get("executed_realized_pnl"), errors="coerce") if not predictions.empty else pd.Series(dtype=float)
    summary = {
        "event_count": int(len(predictions)),
        "fit_event_count": int(parameters["fit_n"].gt(0).sum()) if not parameters.empty and "fit_n" in parameters else 0,
        "trade_count": int(traded.sum()) if not predictions.empty else 0,
        "missed_arbitrage_count": int(predictions.get("missed_arbitrage", pd.Series(dtype=bool)).fillna(False).sum()) if not predictions.empty else 0,
        "loss_trade_count": int(predictions.get("loss_trade", pd.Series(dtype=bool)).fillna(False).sum()) if not predictions.empty else 0,
        "average_pnl_per_trade": float(executed_pnl[traded].mean()) if not predictions.empty and traded.any() else None,
        "total_pnl": float(executed_pnl[traded].sum()) if not predictions.empty and traded.any() else 0.0,
        "trade_notional": float(args.trade_notional),
        "borrow_cost_assumption": 0.0,
        "hedge_policy": args.own_hedge_policy,
    }
    return predictions, parameters, summary


def write_structural_summary(path: Any, summary: Dict[str, Any]) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
