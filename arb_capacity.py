#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Capacity and election-impact overlay for election-arb signals.

The payoff engine prices a marginal share.  This module asks how many shares can
actually be sourced without changing the election outcome enough to destroy the
edge.  It is deliberately transparent: when true holder/borrow data are absent,
the output labels the behavioral assumptions rather than pretending they were
observed.
"""
from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any, Dict, Iterable, List, Optional

import numpy as np
import pandas as pd


@dataclass
class CapacityConfig:
    panel_path: str = "eda_output/merged_panel.csv"
    holder_model: str = "rolling_structural"
    holder_rolling_window_events: int = 50
    holder_min_fit_events: int = 10
    holder_p_grid_size: int = 11
    holder_q_grid_size: int = 11
    default_irrational_cash_prob: float = 0.50
    default_rational_share: float = 0.30
    build_days: float = 10.0
    max_adv_participation: float = 0.20
    max_position_pct_shares: float = 0.05
    positive_holder_share_of_active: float = 0.35
    noisy_sell_fraction: float = 0.50
    noisy_buy_fraction: float = 0.50
    noisy_election_cash_prob: float = 0.50
    positive_sell_min: float = 0.02
    positive_sell_max: float = 0.85
    positive_sell_width: float = 0.03
    positive_no_sell_edge: float = 0.03
    passive_sell_fraction: float = 0.01
    passive_lendable_fraction: float = 0.20
    noisy_lendable_fraction: float = 0.10
    capacity_grid_points: int = 51


def as_float(value: Any) -> float:
    try:
        x = float(value)
    except Exception:
        return math.nan
    return x if math.isfinite(x) else math.nan


def clamp01(value: float) -> float:
    if not math.isfinite(value):
        return math.nan
    return max(0.0, min(1.0, float(value)))


def normalize_fraction(value: Any) -> float:
    x = as_float(value)
    if not math.isfinite(x):
        return math.nan
    if x > 1.0 and x <= 100.0:
        x = x / 100.0
    return clamp01(x)


def build_rolling_holder_estimates(panel: pd.DataFrame, cfg: CapacityConfig) -> pd.DataFrame:
    """Estimate event-level holder composition from prior realized election events.

    This is the old structural model wired into the current capacity layer:
    p_hat = irrational/noisy holder cash-election probability.
    q_hat = rational, EV-sensitive holder share of total target ownership.
    Fits are rolling and leave the current event out by construction.
    """
    if panel.empty or "event_id" not in panel.columns:
        return pd.DataFrame()
    try:
        from structural_election_model import fit_p_q, observed_election_shares, predict_votes, sort_panel
    except Exception:
        return pd.DataFrame()

    ordered = sort_panel(panel)
    historical_rows: List[pd.Series] = []
    rows: List[Dict[str, Any]] = []
    p_grid = max(3, int(cfg.holder_p_grid_size))
    q_grid = max(3, int(cfg.holder_q_grid_size))
    for i, row in ordered.iterrows():
        train_rows = historical_rows[-cfg.holder_rolling_window_events:] if cfg.holder_rolling_window_events > 0 else historical_rows
        fit = fit_p_q(
            train_rows if len(train_rows) >= cfg.holder_min_fit_events else [],
            p_grid,
            q_grid,
            cfg.default_irrational_cash_prob,
            cfg.default_rational_share,
        )
        pred = predict_votes(row, fit["p_hat"], fit["q_hat"])
        passive = normalize_fraction(pred.get("passive_share", row.get("passive_control_percent", math.nan)))
        passive = 0.0 if not math.isfinite(passive) else passive
        active = max(0.0, 1.0 - passive)
        q_hat = clamp01(as_float(fit.get("q_hat", math.nan)))
        p_hat = clamp01(as_float(fit.get("p_hat", math.nan)))
        positive = min(active, q_hat) if math.isfinite(q_hat) else math.nan
        noisy = max(0.0, active - positive) if math.isfinite(positive) else math.nan
        rows.append({
            "event_id": row.get("event_id"),
            "holder_model_source": "rolling_structural_fit" if int(fit.get("fit_n", 0) or 0) > 0 else "rolling_structural_default",
            "holder_rolling_event_index": i,
            "holder_fit_n": int(fit.get("fit_n", 0) or 0),
            "holder_fit_sse": fit.get("fit_sse", math.nan),
            "holder_fit_loglike": fit.get("fit_loglike", math.nan),
            "holder_p_hat": p_hat,
            "holder_q_hat": q_hat,
            "holder_positive_share": positive,
            "holder_positive_share_of_active": positive / active if active > 0 and math.isfinite(positive) else math.nan,
            "holder_noisy_share": noisy,
            "holder_noisy_cash_prob": p_hat,
            "holder_predicted_cash_demand_share": pred.get("predicted_cash_demand_share", math.nan),
            "holder_predicted_stock_demand_share": pred.get("predicted_stock_demand_share", math.nan),
            "holder_rational_action": pred.get("rational_action", ""),
            "holder_structural_status": pred.get("model_status", ""),
            "holder_structural_block_reason": pred.get("block_reason", ""),
        })
        obs_cash, _, _ = observed_election_shares(row)
        if obs_cash is not None:
            historical_rows.append(row)
    return pd.DataFrame(rows)


def load_capacity_table(path: str, cfg: Optional[CapacityConfig] = None) -> pd.DataFrame:
    if not path:
        return pd.DataFrame()
    try:
        panel = pd.read_csv(path)
    except FileNotFoundError:
        return pd.DataFrame()
    if "event_id" not in panel.columns:
        return pd.DataFrame()
    keep = [
        "event_id",
        "target_shares_outstanding",
        "target_adv20",
        "target_dollar_adv20",
        "target_market_cap",
        "passive_control_percent",
        "etf_ownership_percent",
        "active_share",
        "passive_pct",
        "default_rule",
        "non_election_default_rule",
        "short_volume_ratio_20d_avg",
        "announce_date",
        "active_cash_election_rate",
        "realized_cash_share",
        "pct_elected_cash",
        "spread",
        "deal_spread",
        "cash_election_value",
        "stock_election_value",
    ]
    keep = [c for c in keep if c in panel.columns]
    out = panel[keep].drop_duplicates("event_id")
    if cfg is not None and str(cfg.holder_model).lower() == "rolling_structural":
        estimates = build_rolling_holder_estimates(panel, cfg)
        if not estimates.empty:
            out = out.merge(estimates, on="event_id", how="left")
    return out.set_index("event_id")


def capacity_row(event_id: str, table: pd.DataFrame) -> pd.Series:
    if table.empty or event_id not in table.index:
        return pd.Series(dtype=object)
    row = table.loc[event_id]
    if isinstance(row, pd.DataFrame):
        row = row.iloc[0]
    return row


def passive_share_from_row(row: pd.Series) -> float:
    passive = normalize_fraction(row.get("passive_control_percent", math.nan))
    if not math.isfinite(passive):
        passive = normalize_fraction(row.get("passive_pct", math.nan))
    if not math.isfinite(passive):
        passive = normalize_fraction(row.get("etf_ownership_percent", math.nan))
    return 0.0 if not math.isfinite(passive) else passive


def default_cash_probability(row: pd.Series, pi_cash: float) -> float:
    text = str(row.get("default_rule", row.get("non_election_default_rule", ""))).strip().lower()
    if "cash" in text and "stock" not in text:
        return 1.0
    if "stock" in text and "cash" not in text:
        return 0.0
    if "mixed" in text or ("cash" in text and "stock" in text):
        return clamp01(pi_cash)
    return clamp01(pi_cash) if math.isfinite(pi_cash) else 0.5


def capacity_inputs(row: pd.Series, entry_price: float, cfg: CapacityConfig) -> Dict[str, Any]:
    shares = as_float(row.get("target_shares_outstanding", math.nan))
    adv = as_float(row.get("target_adv20", math.nan))
    dollar_adv = as_float(row.get("target_dollar_adv20", math.nan))
    if (not math.isfinite(adv) or adv <= 0) and math.isfinite(dollar_adv) and entry_price > 0:
        adv = dollar_adv / entry_price

    passive = passive_share_from_row(row)
    active = max(0.0, 1.0 - passive)
    rolling_positive = normalize_fraction(row.get("holder_positive_share", math.nan))
    if math.isfinite(rolling_positive):
        positive = min(active, rolling_positive)
        holder_model_source = str(row.get("holder_model_source", "rolling_structural_fit"))
    else:
        positive = active * clamp01(cfg.positive_holder_share_of_active)
        holder_model_source = "static_positive_holder_share_of_active"
    noisy = max(0.0, active - positive)
    noisy_cash_prob = normalize_fraction(row.get("holder_noisy_cash_prob", math.nan))
    if not math.isfinite(noisy_cash_prob):
        noisy_cash_prob = cfg.noisy_election_cash_prob
    adv_pct = adv / shares if math.isfinite(adv) and math.isfinite(shares) and shares > 0 else math.nan
    return {
        "shares_outstanding": shares,
        "adv20": adv,
        "dollar_adv20": dollar_adv,
        "passive_share": passive,
        "active_share": active,
        "positive_holder_share": positive,
        "positive_holder_share_of_active": positive / active if active > 0 else math.nan,
        "noisy_holder_share": noisy,
        "noisy_cash_prob": noisy_cash_prob,
        "adv_pct_shares": adv_pct,
        "holder_model_source": holder_model_source,
        "holder_fit_n": as_float(row.get("holder_fit_n", math.nan)),
        "holder_p_hat": as_float(row.get("holder_p_hat", math.nan)),
        "holder_q_hat": as_float(row.get("holder_q_hat", math.nan)),
        "holder_predicted_cash_demand_share": as_float(row.get("holder_predicted_cash_demand_share", math.nan)),
        "holder_rational_action": str(row.get("holder_rational_action", "")),
    }


def positive_sell_rate(edge_return: float, cfg: CapacityConfig) -> float:
    """EV-sensitive holders sell little when edge is positive, a lot when it is negative."""
    edge = 0.0 if not math.isfinite(edge_return) else float(edge_return)
    if edge >= cfg.positive_no_sell_edge:
        return 0.0
    if edge >= 0:
        scale = 1.0 - edge / max(cfg.positive_no_sell_edge, 1e-6)
        return max(0.0, cfg.positive_sell_min * scale)
    stress = min(1.0, -edge / max(cfg.positive_sell_width, 1e-6))
    return cfg.positive_sell_min + (cfg.positive_sell_max - cfg.positive_sell_min) * stress


def _finite_capacity_inputs(inputs: Dict[str, Any]) -> bool:
    shares = inputs.get("shares_outstanding", math.nan)
    adv = inputs.get("adv20", math.nan)
    return math.isfinite(shares) and shares > 0 and math.isfinite(adv) and adv > 0


def _cap_summary(pools: Iterable[Dict[str, float]], adv_cap: float, position_cap: float) -> Dict[str, Any]:
    source_cap = sum(max(0.0, p.get("capacity_pct", 0.0)) for p in pools)
    caps = {
        "source_supply": source_cap,
        "adv_participation": adv_cap,
        "position_limit": position_cap,
    }
    max_pct = min(caps.values())
    binding = min(caps, key=caps.get)
    return {"max_ownership": max(0.0, max_pct), "binding_constraint": binding, "source_cap": source_cap}


def long_flow(row: pd.Series, entry_price: float, pi_cash: float, elect: str,
              edge_return: float, cfg: CapacityConfig) -> Dict[str, Any]:
    inputs = capacity_inputs(row, entry_price, cfg)
    if not _finite_capacity_inputs(inputs):
        return {"status": "missing_shares_or_adv", **inputs, "max_ownership": math.nan, "pools": []}

    passive_cash = default_cash_probability(row, pi_cash)
    own_cash = 1.0 if str(elect).upper() == "CASH" else 0.0
    pos_sell = positive_sell_rate(edge_return, cfg)
    adv_cap = inputs["adv_pct_shares"] * cfg.build_days * cfg.max_adv_participation
    pools = [
        {
            "name": "noisy_sellers",
            "capacity_pct": min(inputs["noisy_holder_share"], adv_cap * cfg.noisy_sell_fraction),
            "cash_prob": inputs["noisy_cash_prob"],
        },
        {
            "name": "positive_holders",
            "capacity_pct": inputs["positive_holder_share"] * pos_sell,
            "cash_prob": own_cash,
        },
        {
            "name": "passive_inactive",
            "capacity_pct": inputs["passive_share"] * cfg.passive_sell_fraction,
            "cash_prob": passive_cash,
        },
    ]
    if edge_return < 0:
        pools = [pools[1], pools[0], pools[2]]
    cap = _cap_summary(pools, adv_cap, cfg.max_position_pct_shares)
    return {
        "status": "ok",
        **inputs,
        **cap,
        "pools": pools,
        "own_cash_prob": own_cash,
        "positive_sell_rate": pos_sell,
        "passive_cash_prob": passive_cash,
        "flow_model": "long_target_buy_from_noisy_then_ev_sensitive_then_passive",
    }


def reverse_flow(row: pd.Series, entry_price: float, pi_cash: float,
                 reverse_return: float, cfg: CapacityConfig) -> Dict[str, Any]:
    inputs = capacity_inputs(row, entry_price, cfg)
    if not _finite_capacity_inputs(inputs):
        return {"status": "missing_shares_or_adv", **inputs, "max_ownership": math.nan, "pools": []}

    passive_cash = default_cash_probability(row, pi_cash)
    adv_cap = inputs["adv_pct_shares"] * cfg.build_days * cfg.max_adv_participation
    borrow_pools = [
        {
            "name": "passive_inactive_lenders",
            "capacity_pct": inputs["passive_share"] * cfg.passive_lendable_fraction,
            "cash_prob": passive_cash,
        },
        {
            "name": "noisy_inactive_lenders",
            "capacity_pct": inputs["noisy_holder_share"] * cfg.noisy_lendable_fraction,
            "cash_prob": inputs["noisy_cash_prob"],
        },
    ]
    borrow_cap = sum(p["capacity_pct"] for p in borrow_pools)
    noisy_buyer_cap = min(inputs["noisy_holder_share"], adv_cap * cfg.noisy_buy_fraction)
    cap = _cap_summary(
        [{"name": "borrowable_inactive", "capacity_pct": borrow_cap, "cash_prob": passive_cash}],
        adv_cap,
        cfg.max_position_pct_shares,
    )
    if noisy_buyer_cap < cap["max_ownership"]:
        cap["max_ownership"] = noisy_buyer_cap
        cap["binding_constraint"] = "noisy_buyer_demand"

    lender_alloc = allocate_sources(borrow_pools, cap["max_ownership"])
    buyer_cash = inputs["noisy_cash_prob"]
    return {
        "status": "ok",
        **inputs,
        **cap,
        "pools": borrow_pools,
        "borrow_cap": borrow_cap,
        "noisy_buyer_cap": noisy_buyer_cap,
        "lender_cash_prob": lender_alloc["cash_prob"],
        "buyer_cash_prob": buyer_cash,
        "election_cash_shift": cap["max_ownership"] * (buyer_cash - lender_alloc["cash_prob"]),
        "flow_model": "short_target_borrow_from_inactive_sell_to_noisy_buyers",
        "positive_sell_rate": positive_sell_rate(reverse_return, cfg),
        "passive_cash_prob": passive_cash,
    }


def allocate_sources(pools: Iterable[Dict[str, float]], ownership: float) -> Dict[str, Any]:
    q = max(0.0, 0.0 if not math.isfinite(ownership) else float(ownership))
    if q <= 0:
        return {"cash_prob": math.nan, "source_mix": "", "unfilled_pct": 0.0}
    remaining = q
    cash = 0.0
    taken: List[str] = []
    for pool in pools:
        cap = max(0.0, pool.get("capacity_pct", 0.0))
        take = min(remaining, cap)
        if take <= 0:
            continue
        cash += take * pool.get("cash_prob", 0.5)
        taken.append(f"{pool.get('name', 'unknown')}:{take / q:.2f}")
        remaining -= take
        if remaining <= 1e-12:
            break
    filled = max(q - remaining, 0.0)
    return {
        "cash_prob": cash / filled if filled > 0 else math.nan,
        "source_mix": "|".join(taken),
        "unfilled_pct": max(0.0, remaining),
    }
