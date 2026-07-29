#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
arb_signal.py  —  TRADE DECISION LAYER for cash-or-stock election arbitrage.

Turns the Monte Carlo payoff model into an actionable trade per deal. For each deal it decides:

  ENTRY     buy the target at its post-announcement market price M
  ELECT     choose the side (cash / stock) with the higher EXPECTED consideration under the
            demand model — the election is committed up front, before the crowd is known
  HEDGE     short the acquirer stock leg you expect to receive, so the deal's value is locked
            at today's prices (removes acquirer price risk -> you don't need the close price)
  REVERSE   short the target when the expected edge is negative. This side has NO election
            right, so its completion liability is the passive blended settlement, not the
            optimal elected payoff.
  VALUE     V = E[ elected-side consideration ] from the MC, valued at entry prices
  SIGNAL    completion payoff distribution + completed/terminated/withdrawn outcome overlay
            -> expected return, p5, CVaR, loss probability, and size

This is a forward-looking generator: point it at a live deal's terms+prices and it emits the
trade. Run on our historical deals it also prints the REALIZED outcome next to each signal.

The demand model is fit LEAVE-ONE-OUT per deal (a Beta that never saw the deal it prices), so
the reported signal skill is fully out-of-sample with respect to the demand distribution.

Caveats (flagged, not hidden): if no event-level outcome-probability file is supplied, the
completed/terminated/withdrawn probabilities are scenario defaults, not a trained model. Hedge
assumes the expected stock fraction (residual hedging error ignored).
"""
from __future__ import annotations
import argparse
from dataclasses import dataclass
import json
from pathlib import Path
import zlib
import numpy as np
import pandas as pd
from arb_capacity import (
    CapacityConfig,
    allocate_sources,
    capacity_row,
    load_capacity_table,
    long_flow,
    reverse_flow,
)
from arb_mc import DemandModel, apply_outcome_overlay, prorate
from arb_outcome import OutcomeDefaults, load_outcome_probability_table, normalize_outcome_label, outcome_probabilities_for_event

RECOVERY_LAG, ENTRY_LAG = 12, 5                    # pre-announce recovery days; post-announce entry days
DEFAULT_P_COMPLETED = 0.88
DEFAULT_P_TERMINATED = 0.07
DEFAULT_P_WITHDRAWN = 0.05
HURDLE = 0.005                                     # 0.5% risk-adjusted return to enter
MIN_P05_RETURN = -0.25
MAX_LOSS_PROBABILITY = 0.50
TERMINATED_LOSS_FRAC = 0.25
WITHDRAWN_LOSS_FRAC = 0.35
NDRAW = 20000


@dataclass
class SignalConfig:
    deals_path: str = "arb_deals.csv"
    market_daily_path: str = "ma_market_wrds/wrds_market_daily.csv"
    capacity_panel_path: str = "eda_output/merged_panel.csv"
    output_path: str = "arb_signals.csv"
    summary_output_path: str = "arb_strategy_summary.json"
    outcome_probs_path: str = ""
    n_draws: int = NDRAW
    hurdle: float = HURDLE
    decision_metric: str = "mean"
    min_p05_return: float = MIN_P05_RETURN
    max_loss_probability: float = MAX_LOSS_PROBABILITY
    default_completed_prob: float = DEFAULT_P_COMPLETED
    default_terminated_prob: float = DEFAULT_P_TERMINATED
    default_withdrawn_prob: float = DEFAULT_P_WITHDRAWN
    withdrawn_share_of_break_prob: float = 0.35
    terminated_loss_frac: float = TERMINATED_LOSS_FRAC
    withdrawn_loss_frac: float = WITHDRAWN_LOSS_FRAC
    holder_model: str = "rolling_structural"
    holder_rolling_window_events: int = 50
    holder_min_fit_events: int = 10
    holder_p_grid_size: int = 11
    holder_q_grid_size: int = 11
    default_irrational_cash_prob: float = 0.50
    default_rational_share: float = 0.30
    capacity_build_days: float = 10.0
    capacity_max_adv_participation: float = 0.20
    capacity_max_position_pct_shares: float = 0.05
    positive_holder_share_of_active: float = 0.35
    positive_no_sell_edge: float = 0.03
    capacity_grid_points: int = 51


def price_on(df, date, mode="onbefore"):
    df = df.dropna(subset=["price_date", "price"]).sort_values("price_date")
    if mode == "onbefore":
        s = df[df.price_date <= date]
        return s["price"].iloc[-1] if len(s) else np.nan
    s = df[df.price_date >= date]
    return s["price"].iloc[0] if len(s) else np.nan


def state_break_values(M, recovery, terminated_loss_frac, withdrawn_loss_frac):
    """Return scenario values for terminated and withdrawn states."""
    recovery_loss = max(M - recovery, 0.0) if np.isfinite(recovery) else 0.0
    terminated_loss = max(recovery_loss, terminated_loss_frac * M)
    withdrawn_loss = max(recovery_loss, withdrawn_loss_frac * M)
    return M - terminated_loss, M - withdrawn_loss


def summarize_returns(ret_dist):
    p05 = float(np.percentile(ret_dist, 5))
    tail = ret_dist[ret_dist <= p05]
    cvar05 = float(tail.mean()) if len(tail) else p05
    return {
        "mean": float(ret_dist.mean()),
        "p05": p05,
        "p50": float(np.percentile(ret_dist, 50)),
        "p95": float(np.percentile(ret_dist, 95)),
        "cvar05": cvar05,
        "loss_probability": float((ret_dist < 0).mean()),
    }


def decision_value(stats, metric):
    metric = str(metric).lower()
    if metric == "p05":
        return stats["p05"]
    if metric == "cvar05":
        return stats["cvar05"]
    return stats["mean"]


def gate_trade(stats, metric, cfg):
    decision = decision_value(stats, metric)
    p05 = stats["p05"]
    cvar05 = stats["cvar05"]
    loss_probability = stats["loss_probability"]
    if decision > cfg.hurdle and p05 >= cfg.min_p05_return and loss_probability <= cfg.max_loss_probability:
        risk_scale = max(-cvar05, -p05, 1e-3)
        return True, float(np.clip(stats["mean"] / risk_scale, 0, 3)), "risk_adjusted_return_positive"
    if decision <= cfg.hurdle:
        return False, 0.0, "risk_adjusted_return_below_hurdle"
    if p05 < cfg.min_p05_return:
        return False, 0.0, "p05_below_floor"
    return False, 0.0, "loss_probability_above_ceiling"


def apply_overlay_with_uniforms(complete_values, probs, terminated_value, withdrawn_value, uniforms):
    complete_values = np.asarray(complete_values, float)
    u = np.asarray(uniforms, float)
    p_completed = float(probs["completed"])
    p_terminated = float(probs["terminated"])
    return np.where(
        u < p_completed,
        complete_values,
        np.where(u < p_completed + p_terminated, terminated_value, withdrawn_value),
    )


def capacity_config_from_signal_config(cfg):
    return CapacityConfig(
        panel_path=cfg.capacity_panel_path,
        holder_model=cfg.holder_model,
        holder_rolling_window_events=cfg.holder_rolling_window_events,
        holder_min_fit_events=cfg.holder_min_fit_events,
        holder_p_grid_size=cfg.holder_p_grid_size,
        holder_q_grid_size=cfg.holder_q_grid_size,
        default_irrational_cash_prob=cfg.default_irrational_cash_prob,
        default_rational_share=cfg.default_rational_share,
        build_days=cfg.capacity_build_days,
        max_adv_participation=cfg.capacity_max_adv_participation,
        max_position_pct_shares=cfg.capacity_max_position_pct_shares,
        positive_holder_share_of_active=cfg.positive_holder_share_of_active,
        positive_no_sell_edge=cfg.positive_no_sell_edge,
        capacity_grid_points=max(5, int(getattr(cfg, "capacity_grid_points", 51))),
    )


def empty_capacity(status="not_evaluated"):
    return {
        "capacity_status": status,
        "capacity_flow_model": "",
        "capacity_binding_constraint": "",
        "capacity_source_mix": "",
        "capacity_raw_max_binding_constraint": "",
        "capacity_raw_max_notional": np.nan,
        "capacity_raw_max_pct_shares_outstanding": np.nan,
        "capacity_raw_max_shares": np.nan,
        "capacity_max_binding_constraint": "",
        "capacity_max_source_mix": "",
        "capacity_max_seller_cash_prob": np.nan,
        "capacity_max_own_cash_prob": np.nan,
        "capacity_max_lender_cash_prob": np.nan,
        "capacity_max_buyer_cash_prob": np.nan,
        "capacity_max_shares": np.nan,
        "capacity_max_notional": np.nan,
        "capacity_max_pct_shares_outstanding": np.nan,
        "capacity_max_E_return_%": np.nan,
        "capacity_max_downside_5%_%": np.nan,
        "capacity_max_loss_probability_%": np.nan,
        "capacity_max_expected_pnl": np.nan,
        "capacity_max_cash_demand_shift_pctpts": np.nan,
        "capacity_max_impacted_mean_cash_election_%": np.nan,
        "capacity_max_baseline_realized_return_%": np.nan,
        "capacity_max_baseline_realized_pnl": np.nan,
        "capacity_max_self_impact_realized_return_%": np.nan,
        "capacity_max_self_impact_realized_pnl": np.nan,
        "capacity_optimal_binding_constraint": "",
        "capacity_optimal_source_mix": "",
        "capacity_optimal_seller_cash_prob": np.nan,
        "capacity_optimal_own_cash_prob": np.nan,
        "capacity_optimal_lender_cash_prob": np.nan,
        "capacity_optimal_buyer_cash_prob": np.nan,
        "capacity_optimal_shares": np.nan,
        "capacity_optimal_notional": np.nan,
        "capacity_optimal_pct_shares_outstanding": np.nan,
        "capacity_optimal_E_return_%": np.nan,
        "capacity_optimal_downside_5%_%": np.nan,
        "capacity_optimal_loss_probability_%": np.nan,
        "capacity_optimal_expected_pnl": np.nan,
        "capacity_optimal_cash_demand_shift_pctpts": np.nan,
        "capacity_optimal_impacted_mean_cash_election_%": np.nan,
        "capacity_optimal_baseline_realized_return_%": np.nan,
        "capacity_optimal_baseline_realized_pnl": np.nan,
        "capacity_optimal_self_impact_realized_return_%": np.nan,
        "capacity_optimal_self_impact_realized_pnl": np.nan,
        "capacity_shares": np.nan,
        "capacity_notional": np.nan,
        "capacity_pct_shares_outstanding": np.nan,
        "capacity_max_raw_pct_shares_outstanding": np.nan,
        "capacity_adv20": np.nan,
        "capacity_shares_outstanding": np.nan,
        "capacity_passive_share_%": np.nan,
        "capacity_positive_holder_share_%": np.nan,
        "capacity_positive_holder_share_of_active_%": np.nan,
        "capacity_noisy_holder_share_%": np.nan,
        "capacity_noisy_cash_prob": np.nan,
        "capacity_holder_model_source": "",
        "capacity_holder_fit_n": np.nan,
        "capacity_holder_p_hat": np.nan,
        "capacity_holder_q_hat": np.nan,
        "capacity_holder_predicted_cash_demand_%": np.nan,
        "capacity_holder_rational_action": "",
        "capacity_positive_sell_rate_%": np.nan,
        "capacity_seller_cash_prob": np.nan,
        "capacity_buyer_cash_prob": np.nan,
        "capacity_lender_cash_prob": np.nan,
        "capacity_own_cash_prob": np.nan,
        "capacity_cash_demand_shift_pctpts": np.nan,
        "capacity_impacted_mean_cash_election_%": np.nan,
        "capacity_election_impact_used_in_payoff": False,
        "capacity_adjusted_E_return_%": np.nan,
        "capacity_adjusted_downside_5%_%": np.nan,
        "capacity_adjusted_loss_probability_%": np.nan,
        "capacity_alpha_decay_%": np.nan,
        "capacity_adjusted_realized_return_%": np.nan,
        "capacity_baseline_realized_return_%": np.nan,
        "capacity_baseline_realized_pnl": np.nan,
        "capacity_self_impact_realized_return_%": np.nan,
        "capacity_self_impact_realized_pnl": np.nan,
        "self_impact_can_eliminate_arbitrage": False,
        "self_impact_break_even_pct_shares_outstanding": np.nan,
        "self_impact_proof_note": "",
    }


def rng_for_event(event_id, salt=0):
    seed = (zlib.crc32(str(event_id).encode("utf-8")) + int(salt)) & 0xFFFFFFFF
    return np.random.default_rng(seed)


def _long_q_candidate(flow, d, M, stock_val, f, elect, probs, terminated_value,
                      withdrawn_value, uniforms, q):
    alloc = allocate_sources(flow.get("pools", []), float(q))
    seller_cash = alloc["cash_prob"]
    own_cash = flow.get("own_cash_prob", 1.0 if elect == "CASH" else 0.0)
    if not np.isfinite(seller_cash) or alloc["unfilled_pct"] > 1e-9:
        return None
    shift = float(q) * (own_cash - seller_cash)
    f_impacted = np.clip(f + shift, 1e-6, 1 - 1e-6)
    cash_h, stock_h, _, _ = prorate(f_impacted, d.pi_cash, d.C, stock_val)
    held = cash_h if elect == "CASH" else stock_h
    realized_val = apply_overlay_with_uniforms(held, probs, terminated_value, withdrawn_value, uniforms)
    stats = summarize_returns((realized_val - M) / M)
    return {
        "q": float(q),
        "alloc": alloc,
        "stats": stats,
        "shift": shift,
        "impacted_mean": float(f_impacted.mean()),
        "seller_cash": seller_cash,
        "own_cash": own_cash,
    }


def _noisy_self_impact_proof(d, M, stock_val, f, elect, probs, terminated_value,
                             withdrawn_value, uniforms, cfg):
    own_cash = 1.0 if str(elect).upper() == "CASH" else 0.0
    seller_cash = 0.5
    for q in np.linspace(0.0, 1.0, 201):
        shift = float(q) * (own_cash - seller_cash)
        f_impacted = np.clip(f + shift, 1e-6, 1 - 1e-6)
        cash_h, stock_h, _, _ = prorate(f_impacted, d.pi_cash, d.C, stock_val)
        held = cash_h if elect == "CASH" else stock_h
        realized_val = apply_overlay_with_uniforms(held, probs, terminated_value, withdrawn_value, uniforms)
        stats = summarize_returns((realized_val - M) / M)
        ok, _, reason = gate_trade(stats, cfg.decision_metric, cfg)
        if not ok:
            return {
                "can_eliminate": True,
                "break_even_pct": float(q * 100),
                "note": f"pure_noisy_counterparty_self_impact_hits_{reason}",
            }
    return {
        "can_eliminate": False,
        "break_even_pct": np.nan,
        "note": "not_within_100pct_target_shares_under_current_terms_and_gates",
    }


def _write_capacity_choice(out, prefix, cand, shares, M, binding):
    if cand is None or shares <= 0:
        return
    q = cand["q"]
    notional = q * shares * M
    stats = cand["stats"]
    out.update({
        f"capacity_{prefix}_binding_constraint": binding,
        f"capacity_{prefix}_source_mix": cand["alloc"]["source_mix"],
        f"capacity_{prefix}_seller_cash_prob": cand.get("seller_cash", np.nan),
        f"capacity_{prefix}_own_cash_prob": cand.get("own_cash", np.nan),
        f"capacity_{prefix}_shares": q * shares,
        f"capacity_{prefix}_notional": notional,
        f"capacity_{prefix}_pct_shares_outstanding": q * 100,
        f"capacity_{prefix}_E_return_%": stats["mean"] * 100,
        f"capacity_{prefix}_downside_5%_%": stats["p05"] * 100,
        f"capacity_{prefix}_loss_probability_%": stats["loss_probability"] * 100,
        f"capacity_{prefix}_expected_pnl": notional * stats["mean"],
        f"capacity_{prefix}_cash_demand_shift_pctpts": cand["shift"] * 100,
        f"capacity_{prefix}_impacted_mean_cash_election_%": cand["impacted_mean"] * 100,
    })


def _alias_optimal_capacity(out, base_stats):
    out.update({
        "capacity_binding_constraint": out.get("capacity_optimal_binding_constraint", ""),
        "capacity_source_mix": out.get("capacity_optimal_source_mix", ""),
        "capacity_shares": out.get("capacity_optimal_shares", np.nan),
        "capacity_notional": out.get("capacity_optimal_notional", np.nan),
        "capacity_pct_shares_outstanding": out.get("capacity_optimal_pct_shares_outstanding", np.nan),
        "capacity_cash_demand_shift_pctpts": out.get("capacity_optimal_cash_demand_shift_pctpts", np.nan),
        "capacity_impacted_mean_cash_election_%": out.get("capacity_optimal_impacted_mean_cash_election_%", np.nan),
        "capacity_adjusted_E_return_%": out.get("capacity_optimal_E_return_%", np.nan),
        "capacity_adjusted_downside_5%_%": out.get("capacity_optimal_downside_5%_%", np.nan),
        "capacity_adjusted_loss_probability_%": out.get("capacity_optimal_loss_probability_%", np.nan),
    })
    opt_e = out.get("capacity_optimal_E_return_%", np.nan)
    out["capacity_alpha_decay_%"] = (
        base_stats["mean"] * 100 - opt_e if np.isfinite(opt_e) else np.nan
    )


def evaluate_long_capacity(cap_row, d, M, stock_val, f, elect, probs, terminated_value,
                           withdrawn_value, base_stats, cfg, rng):
    cap_cfg = capacity_config_from_signal_config(cfg)
    flow = long_flow(cap_row, M, d.pi_cash, elect, base_stats["mean"], cap_cfg)
    out = empty_capacity(flow.get("status", "missing_capacity_inputs"))
    raw_q = flow.get("max_ownership", np.nan)
    shares = flow.get("shares_outstanding", np.nan)
    out.update({
        "capacity_flow_model": flow.get("flow_model", ""),
        "capacity_raw_max_binding_constraint": flow.get("binding_constraint", ""),
        "capacity_raw_max_pct_shares_outstanding": raw_q * 100 if np.isfinite(raw_q) else np.nan,
        "capacity_raw_max_notional": raw_q * shares * M
        if np.isfinite(raw_q) and np.isfinite(shares) else np.nan,
        "capacity_raw_max_shares": raw_q * shares if np.isfinite(raw_q) and np.isfinite(shares) else np.nan,
        "capacity_max_raw_pct_shares_outstanding": raw_q * 100 if np.isfinite(raw_q) else np.nan,
        "capacity_adv20": flow.get("adv20", np.nan),
        "capacity_shares_outstanding": shares,
        "capacity_passive_share_%": flow.get("passive_share", np.nan) * 100
        if np.isfinite(flow.get("passive_share", np.nan)) else np.nan,
        "capacity_positive_holder_share_%": flow.get("positive_holder_share", np.nan) * 100
        if np.isfinite(flow.get("positive_holder_share", np.nan)) else np.nan,
        "capacity_positive_holder_share_of_active_%": flow.get("positive_holder_share_of_active", np.nan) * 100
        if np.isfinite(flow.get("positive_holder_share_of_active", np.nan)) else np.nan,
        "capacity_noisy_holder_share_%": flow.get("noisy_holder_share", np.nan) * 100
        if np.isfinite(flow.get("noisy_holder_share", np.nan)) else np.nan,
        "capacity_noisy_cash_prob": flow.get("noisy_cash_prob", np.nan),
        "capacity_holder_model_source": flow.get("holder_model_source", ""),
        "capacity_holder_fit_n": flow.get("holder_fit_n", np.nan),
        "capacity_holder_p_hat": flow.get("holder_p_hat", np.nan),
        "capacity_holder_q_hat": flow.get("holder_q_hat", np.nan),
        "capacity_holder_predicted_cash_demand_%": flow.get("holder_predicted_cash_demand_share", np.nan) * 100
        if np.isfinite(flow.get("holder_predicted_cash_demand_share", np.nan)) else np.nan,
        "capacity_holder_rational_action": flow.get("holder_rational_action", ""),
        "capacity_positive_sell_rate_%": flow.get("positive_sell_rate", np.nan) * 100
        if np.isfinite(flow.get("positive_sell_rate", np.nan)) else np.nan,
        "capacity_own_cash_prob": flow.get("own_cash_prob", np.nan),
        "capacity_election_impact_used_in_payoff": True,
    })
    if flow.get("status") != "ok" or not np.isfinite(raw_q) or raw_q <= 0:
        return out

    uniforms = rng.random(len(f))
    proof = _noisy_self_impact_proof(
        d, M, stock_val, f, elect, probs, terminated_value, withdrawn_value, uniforms, cfg
    )
    out.update({
        "self_impact_can_eliminate_arbitrage": proof["can_eliminate"],
        "self_impact_break_even_pct_shares_outstanding": proof["break_even_pct"],
        "self_impact_proof_note": proof["note"],
    })

    max_cand = None
    optimal_cand = None
    optimal_pnl = -np.inf
    fail_reason = ""

    for q in np.linspace(raw_q / cap_cfg.capacity_grid_points, raw_q, cap_cfg.capacity_grid_points):
        cand = _long_q_candidate(
            flow, d, M, stock_val, f, elect, probs, terminated_value, withdrawn_value, uniforms, float(q)
        )
        if cand is None:
            fail_reason = "source_allocation_unfilled"
            continue
        ok, _, reason = gate_trade(cand["stats"], cfg.decision_metric, cfg)
        if ok:
            max_cand = cand
            pnl = cand["q"] * shares * M * cand["stats"]["mean"]
            if pnl > optimal_pnl:
                optimal_pnl = pnl
                optimal_cand = cand
        else:
            fail_reason = reason

    if max_cand is None or optimal_cand is None:
        out.update({
            "capacity_status": "zero_capacity_after_self_impact",
            "capacity_binding_constraint": fail_reason or "self_impact_gate",
        })
        return out

    max_binding = flow.get("binding_constraint", "")
    if max_cand["q"] < raw_q * (1.0 - 1e-6):
        max_binding = "self_impact_" + (fail_reason or "gate")
    opt_binding = max_binding if abs(optimal_cand["q"] - max_cand["q"]) < 1e-12 else "profit_optimal_before_max"
    out["capacity_status"] = "ok"
    _write_capacity_choice(out, "max", max_cand, shares, M, max_binding)
    _write_capacity_choice(out, "optimal", optimal_cand, shares, M, opt_binding)
    out["capacity_seller_cash_prob"] = optimal_cand["seller_cash"]
    out["capacity_own_cash_prob"] = optimal_cand["own_cash"]
    _alias_optimal_capacity(out, base_stats)
    return out


def evaluate_reverse_capacity(cap_row, d, M, f, reverse_stats, cfg):
    cap_cfg = capacity_config_from_signal_config(cfg)
    flow = reverse_flow(cap_row, M, d.pi_cash, reverse_stats["mean"], cap_cfg)
    out = empty_capacity(flow.get("status", "missing_capacity_inputs"))
    out.update({
        "capacity_flow_model": flow.get("flow_model", ""),
        "capacity_binding_constraint": flow.get("binding_constraint", ""),
        "capacity_max_raw_pct_shares_outstanding": flow.get("max_ownership", np.nan) * 100
        if np.isfinite(flow.get("max_ownership", np.nan)) else np.nan,
        "capacity_adv20": flow.get("adv20", np.nan),
        "capacity_shares_outstanding": flow.get("shares_outstanding", np.nan),
        "capacity_passive_share_%": flow.get("passive_share", np.nan) * 100
        if np.isfinite(flow.get("passive_share", np.nan)) else np.nan,
        "capacity_positive_holder_share_%": flow.get("positive_holder_share", np.nan) * 100
        if np.isfinite(flow.get("positive_holder_share", np.nan)) else np.nan,
        "capacity_positive_holder_share_of_active_%": flow.get("positive_holder_share_of_active", np.nan) * 100
        if np.isfinite(flow.get("positive_holder_share_of_active", np.nan)) else np.nan,
        "capacity_noisy_holder_share_%": flow.get("noisy_holder_share", np.nan) * 100
        if np.isfinite(flow.get("noisy_holder_share", np.nan)) else np.nan,
        "capacity_noisy_cash_prob": flow.get("noisy_cash_prob", np.nan),
        "capacity_holder_model_source": flow.get("holder_model_source", ""),
        "capacity_holder_fit_n": flow.get("holder_fit_n", np.nan),
        "capacity_holder_p_hat": flow.get("holder_p_hat", np.nan),
        "capacity_holder_q_hat": flow.get("holder_q_hat", np.nan),
        "capacity_holder_predicted_cash_demand_%": flow.get("holder_predicted_cash_demand_share", np.nan) * 100
        if np.isfinite(flow.get("holder_predicted_cash_demand_share", np.nan)) else np.nan,
        "capacity_holder_rational_action": flow.get("holder_rational_action", ""),
        "capacity_positive_sell_rate_%": flow.get("positive_sell_rate", np.nan) * 100
        if np.isfinite(flow.get("positive_sell_rate", np.nan)) else np.nan,
        "capacity_buyer_cash_prob": flow.get("buyer_cash_prob", np.nan),
        "capacity_lender_cash_prob": flow.get("lender_cash_prob", np.nan),
        "capacity_election_impact_used_in_payoff": False,
    })
    max_q = flow.get("max_ownership", np.nan)
    shares = flow.get("shares_outstanding", np.nan)
    out.update({
        "capacity_raw_max_binding_constraint": flow.get("binding_constraint", ""),
        "capacity_raw_max_pct_shares_outstanding": max_q * 100 if np.isfinite(max_q) else np.nan,
        "capacity_raw_max_notional": max_q * shares * M
        if np.isfinite(max_q) and np.isfinite(shares) else np.nan,
        "capacity_raw_max_shares": max_q * shares if np.isfinite(max_q) and np.isfinite(shares) else np.nan,
    })
    if flow.get("status") != "ok" or not np.isfinite(max_q) or max_q <= 0:
        return out

    alloc = allocate_sources(flow.get("pools", []), max_q)
    shift = flow.get("election_cash_shift", np.nan)
    stats = reverse_stats
    notional = max_q * shares * M
    out.update({
        "capacity_status": "ok",
        "capacity_max_binding_constraint": flow.get("binding_constraint", ""),
        "capacity_max_source_mix": alloc["source_mix"],
        "capacity_max_lender_cash_prob": flow.get("lender_cash_prob", np.nan),
        "capacity_max_buyer_cash_prob": flow.get("buyer_cash_prob", np.nan),
        "capacity_max_shares": max_q * shares,
        "capacity_max_notional": notional,
        "capacity_max_pct_shares_outstanding": max_q * 100,
        "capacity_max_E_return_%": stats["mean"] * 100,
        "capacity_max_downside_5%_%": stats["p05"] * 100,
        "capacity_max_loss_probability_%": stats["loss_probability"] * 100,
        "capacity_max_expected_pnl": notional * stats["mean"],
        "capacity_max_cash_demand_shift_pctpts": shift * 100 if np.isfinite(shift) else np.nan,
        "capacity_max_impacted_mean_cash_election_%": (float(np.mean(f)) + shift) * 100
        if np.isfinite(shift) and len(f) else np.nan,
        "capacity_optimal_binding_constraint": flow.get("binding_constraint", ""),
        "capacity_optimal_source_mix": alloc["source_mix"],
        "capacity_optimal_lender_cash_prob": flow.get("lender_cash_prob", np.nan),
        "capacity_optimal_buyer_cash_prob": flow.get("buyer_cash_prob", np.nan),
        "capacity_optimal_shares": max_q * shares,
        "capacity_optimal_notional": notional,
        "capacity_optimal_pct_shares_outstanding": max_q * 100,
        "capacity_optimal_E_return_%": stats["mean"] * 100,
        "capacity_optimal_downside_5%_%": stats["p05"] * 100,
        "capacity_optimal_loss_probability_%": stats["loss_probability"] * 100,
        "capacity_optimal_expected_pnl": notional * stats["mean"],
        "capacity_optimal_cash_demand_shift_pctpts": shift * 100 if np.isfinite(shift) else np.nan,
        "capacity_optimal_impacted_mean_cash_election_%": (float(np.mean(f)) + shift) * 100
        if np.isfinite(shift) and len(f) else np.nan,
        "capacity_buyer_cash_prob": flow.get("buyer_cash_prob", np.nan),
        "capacity_lender_cash_prob": flow.get("lender_cash_prob", np.nan),
        "self_impact_can_eliminate_arbitrage": False,
        "self_impact_break_even_pct_shares_outstanding": np.nan,
        "self_impact_proof_note": "not_possible_for_reverse_under_fixed_pool_passive_blended_settlement",
    })
    _alias_optimal_capacity(out, reverse_stats)
    out["capacity_alpha_decay_%"] = 0.0
    return out


def build_signals(config=None):
    cfg = config or SignalConfig()
    arb = pd.read_csv(cfg.deals_path)
    ready = arb.dropna(subset=["C", "R", "P_acq", "pi_cash"])
    ready = ready[ready.ratio_type == "fixed"].copy()

    daily = pd.read_csv(cfg.market_daily_path)
    daily["price_date"] = pd.to_datetime(daily["price_date"], errors="coerce")
    daily["announce_date"] = pd.to_datetime(daily["announce_date"], errors="coerce")

    # demand values keyed by event, so each deal can be priced LEAVE-ONE-OUT (a demand model
    # that never saw the deal it is pricing -> the signal skill is fully out-of-sample)
    fc_by_event = arb.dropna(subset=["f_cash"]).set_index("event_id")["f_cash"]
    outcome_table = load_outcome_probability_table(cfg.outcome_probs_path)
    cap_cfg = capacity_config_from_signal_config(cfg)
    capacity_table = load_capacity_table(cfg.capacity_panel_path, cap_cfg)
    outcome_defaults = OutcomeDefaults(
        completed=cfg.default_completed_prob,
        terminated=cfg.default_terminated_prob,
        withdrawn=cfg.default_withdrawn_prob,
        withdrawn_share_of_break=cfg.withdrawn_share_of_break_prob,
    )
    rng = np.random.default_rng(11)

    rows = []
    for _, d in ready.iterrows():
        tg = daily[(daily.event_id == d.event_id) & (daily.side == "target")]
        aq = daily[(daily.event_id == d.event_id) & (daily.side == "acquirer")]
        if tg.empty or aq.empty:
            continue
        ann = tg["announce_date"].dropna().iloc[0] if tg["announce_date"].notna().any() else np.nan
        if pd.isna(ann):
            continue
        M = price_on(tg, ann + pd.Timedelta(days=ENTRY_LAG), "onafter")     # enter after announcement
        recovery = price_on(tg, ann - pd.Timedelta(days=RECOVERY_LAG), "onbefore")  # pre-deal level
        Pacq = price_on(aq, ann + pd.Timedelta(days=ENTRY_LAG), "onafter")
        if not np.isfinite(M) or not np.isfinite(Pacq):
            continue
        # fit the demand Beta on every deal EXCEPT this one, then sample -> out-of-sample pricing
        loo = fc_by_event.drop(d.event_id, errors="ignore").values
        f = DemandModel(loo).draw(cfg.n_draws, rng=rng)
        stock_val = d.R * Pacq
        cash_h, stock_h, blended, _ = prorate(f, d.pi_cash, d.C, stock_val)

        # ex-ante election: pick the side with higher EXPECTED value
        E_cash, E_stock = cash_h.mean(), stock_h.mean()
        elect = "CASH" if E_cash >= E_stock else "STOCK"
        V = max(E_cash, E_stock)
        held = cash_h if elect == "CASH" else stock_h              # payoff distribution given the election
        # expected stock fraction received (what to hedge)
        cash_fill = np.minimum(1.0, d.pi_cash / np.clip(f, 1e-6, 1 - 1e-6))
        stock_fill = np.minimum(1.0, (1 - d.pi_cash) / np.clip(1 - f, 1e-6, 1 - 1e-6))
        stock_frac = (1 - cash_fill).mean() if elect == "CASH" else stock_fill.mean()
        hedge_ratio = d.R * stock_frac                            # acquirer shares to short per target share

        arb_spread = V - M
        arb_ret = arb_spread / M
        outcome = outcome_probabilities_for_event(d, outcome_table, outcome_defaults)
        terminated_value, withdrawn_value = state_break_values(
            M,
            recovery,
            cfg.terminated_loss_frac,
            cfg.withdrawn_loss_frac,
        )
        long_realized_val, normalized_probs = apply_outcome_overlay(
            held,
            entry_value=M,
            p_completed=outcome["p_completed"],
            p_terminated=outcome["p_terminated"],
            p_withdrawn=outcome["p_withdrawn"],
            terminated_value=terminated_value,
            withdrawn_value=withdrawn_value,
            rng=rng,
        )
        long_ret_dist = (long_realized_val - M) / M
        long_risk = summarize_returns(long_ret_dist)
        long_decision = decision_value(long_risk, cfg.decision_metric)
        long_ok, long_size, long_reason = gate_trade(long_risk, cfg.decision_metric, cfg)
        long_e_ret = long_risk["mean"]
        long_risk_adjusted_fair_value = M * (1.0 + long_e_ret)
        terminated_return = (terminated_value - M) / M
        withdrawn_return = (withdrawn_value - M) / M

        # Reverse trade: short target.  The short seller has no election right,
        # so completion-state liability is the passive/blended settlement, not
        # the optimal elected-side payoff.
        passive_settlement = float(blended)
        reverse_liability, _ = apply_outcome_overlay(
            np.full_like(held, passive_settlement, dtype=float),
            entry_value=M,
            p_completed=outcome["p_completed"],
            p_terminated=outcome["p_terminated"],
            p_withdrawn=outcome["p_withdrawn"],
            terminated_value=terminated_value,
            withdrawn_value=withdrawn_value,
            rng=rng,
        )
        reverse_ret_dist = (M - reverse_liability) / M
        reverse_risk = summarize_returns(reverse_ret_dist)
        reverse_decision = decision_value(reverse_risk, cfg.decision_metric)
        reverse_ok, reverse_size, reverse_reason = gate_trade(reverse_risk, cfg.decision_metric, cfg)
        reverse_e_ret = reverse_risk["mean"]
        reverse_completion_return = (M - passive_settlement) / M
        reverse_risk_adjusted_settlement_value = M * (1.0 - reverse_e_ret)
        reverse_hedge_ratio = d.R * (1.0 - d.pi_cash)

        # realized (historical) outcome given the actual election demand
        actual_outcome = normalize_outcome_label(d)
        actual_f_cash = pd.to_numeric(pd.Series([d.get("f_cash", np.nan)]), errors="coerce").iloc[0]
        if actual_outcome == "terminated":
            long_realized_ret = terminated_return
            reverse_realized_ret = -terminated_return
            realized_source = "deal_outcome_label"
        elif actual_outcome == "withdrawn":
            long_realized_ret = withdrawn_return
            reverse_realized_ret = -withdrawn_return
            realized_source = "deal_outcome_label"
        elif actual_outcome == "completed" or np.isfinite(actual_f_cash):
            reverse_realized_ret = reverse_completion_return
            if np.isfinite(actual_f_cash):
                cr, sr, _, _ = prorate(np.array([actual_f_cash]), d.pi_cash, d.C, stock_val)
                long_realized_value = (cr if elect == "CASH" else sr)[0]
                long_realized_ret = (long_realized_value - M) / M
                realized_source = "realized_f_cash"
            else:
                long_realized_ret = np.nan
                realized_source = "completed_status_no_realized_f_cash"
            if not actual_outcome:
                actual_outcome = "completed"
        else:
            long_realized_ret = np.nan
            reverse_realized_ret = np.nan
            realized_source = "missing_realized_outcome"

        # data-quality guard: a real post-announcement merger-arb spread lives in ~[-30%, +30%].
        # anything outside that is almost certainly a bad entry price or misparsed term -> REVIEW,
        # never a tradeable signal. keeps the blotter honest by construction.
        data_quality_return = max(abs(arb_ret), abs(reverse_completion_return))
        # also flag an implausible cash-vs-stock TERM gap (misparsed cash value / ratio / acquirer
        # identity or price): the return check above misses these when the misparse still lands the
        # return inside +/-30% (e.g. REVERSE trades that settle at blended value). Check the gap at
        # BOTH the entry and deadline acquirer prices — a real election is structured near cash/stock
        # parity, so a >50% gap at entry is a misparse, and a >50% gap at the deadline is either a
        # misparse (wrong acquirer/price) or basis risk large enough to warrant REVIEW.
        entry_gap = abs(d.C - stock_val) / max(min(abs(d.C), abs(stock_val)), 1e-9)
        deadline_sv = d.R * d.P_acq
        deadline_gap = abs(d.C - deadline_sv) / max(min(abs(d.C), abs(deadline_sv)), 1e-9)
        term_gap = max(entry_gap, deadline_gap)
        if data_quality_return > 0.30 or term_gap > 0.50:
            signal, size, signal_reason = "REVIEW", 0.0, (
                "implausible_cash_vs_stock_term_gap" if term_gap > 0.50
                else "abs_completion_or_election_return_above_30pct")
            selected = "REVIEW"
            selected_risk = long_risk if long_decision >= reverse_decision else reverse_risk
            selected_decision = max(long_decision, reverse_decision)
            selected_e_ret = selected_risk["mean"]
            selected_fair_value = V if long_decision >= reverse_decision else passive_settlement
            selected_risk_adjusted_value = (
                long_risk_adjusted_fair_value
                if long_decision >= reverse_decision
                else reverse_risk_adjusted_settlement_value
            )
            selected_hedge_ratio = hedge_ratio if long_decision >= reverse_decision else reverse_hedge_ratio
            hedge_side = "SHORT_ACQUIRER" if long_decision >= reverse_decision else "LONG_ACQUIRER"
            realized_ret = long_realized_ret if long_decision >= reverse_decision else reverse_realized_ret
        elif long_ok or reverse_ok:
            choose_reverse = reverse_ok and (not long_ok or reverse_decision > long_decision)
            if choose_reverse:
                signal, size, signal_reason = "REVERSE", reverse_size, reverse_reason
                selected = "SHORT_TARGET_PASSIVE"
                selected_risk = reverse_risk
                selected_decision = reverse_decision
                selected_e_ret = reverse_e_ret
                selected_fair_value = passive_settlement
                selected_risk_adjusted_value = reverse_risk_adjusted_settlement_value
                selected_hedge_ratio = reverse_hedge_ratio
                hedge_side = "LONG_ACQUIRER"
                realized_ret = reverse_realized_ret
            else:
                signal, size, signal_reason = "ENTER", long_size, long_reason
                selected = "LONG_TARGET_ELECTION"
                selected_risk = long_risk
                selected_decision = long_decision
                selected_e_ret = long_e_ret
                selected_fair_value = V
                selected_risk_adjusted_value = long_risk_adjusted_fair_value
                selected_hedge_ratio = hedge_ratio
                hedge_side = "SHORT_ACQUIRER"
                realized_ret = long_realized_ret
        elif long_reason == "p05_below_floor" or reverse_reason == "p05_below_floor":
            signal, size, signal_reason = "PASS_P05", 0.0, "best_candidate_p05_below_floor"
            selected = "PASS"
            selected_risk = long_risk if long_decision >= reverse_decision else reverse_risk
            selected_decision = max(long_decision, reverse_decision)
            selected_e_ret = selected_risk["mean"]
            selected_fair_value = V if long_decision >= reverse_decision else passive_settlement
            selected_risk_adjusted_value = (
                long_risk_adjusted_fair_value
                if long_decision >= reverse_decision
                else reverse_risk_adjusted_settlement_value
            )
            selected_hedge_ratio = hedge_ratio if long_decision >= reverse_decision else reverse_hedge_ratio
            hedge_side = "SHORT_ACQUIRER" if long_decision >= reverse_decision else "LONG_ACQUIRER"
            realized_ret = long_realized_ret if long_decision >= reverse_decision else reverse_realized_ret
        elif long_reason == "loss_probability_above_ceiling" or reverse_reason == "loss_probability_above_ceiling":
            signal, size, signal_reason = "PASS_LOSS_PROB", 0.0, "best_candidate_loss_probability_above_ceiling"
            selected = "PASS"
            selected_risk = long_risk if long_decision >= reverse_decision else reverse_risk
            selected_decision = max(long_decision, reverse_decision)
            selected_e_ret = selected_risk["mean"]
            selected_fair_value = V if long_decision >= reverse_decision else passive_settlement
            selected_risk_adjusted_value = (
                long_risk_adjusted_fair_value
                if long_decision >= reverse_decision
                else reverse_risk_adjusted_settlement_value
            )
            selected_hedge_ratio = hedge_ratio if long_decision >= reverse_decision else reverse_hedge_ratio
            hedge_side = "SHORT_ACQUIRER" if long_decision >= reverse_decision else "LONG_ACQUIRER"
            realized_ret = long_realized_ret if long_decision >= reverse_decision else reverse_realized_ret
        else:
            signal, size, signal_reason = "PASS", 0.0, "both_directions_below_hurdle"
            selected = "PASS"
            selected_risk = long_risk if long_decision >= reverse_decision else reverse_risk
            selected_decision = max(long_decision, reverse_decision)
            selected_e_ret = selected_risk["mean"]
            selected_fair_value = V if long_decision >= reverse_decision else passive_settlement
            selected_risk_adjusted_value = (
                long_risk_adjusted_fair_value
                if long_decision >= reverse_decision
                else reverse_risk_adjusted_settlement_value
            )
            selected_hedge_ratio = hedge_ratio if long_decision >= reverse_decision else reverse_hedge_ratio
            hedge_side = "SHORT_ACQUIRER" if long_decision >= reverse_decision else "LONG_ACQUIRER"
            realized_ret = long_realized_ret if long_decision >= reverse_decision else reverse_realized_ret

        cap = empty_capacity()
        cap_source = capacity_row(str(d.event_id), capacity_table)
        if signal == "ENTER":
            cap = evaluate_long_capacity(
                cap_source,
                d,
                M,
                stock_val,
                f,
                elect,
                normalized_probs,
                terminated_value,
                withdrawn_value,
                long_risk,
                cfg,
                rng_for_event(d.event_id, 991),
            )
            for prefix in ["max", "optimal"]:
                notional = cap.get(f"capacity_{prefix}_notional", np.nan)
                q = cap.get(f"capacity_{prefix}_pct_shares_outstanding", np.nan) / 100.0
                seller_cash = cap.get(f"capacity_{prefix}_seller_cash_prob", np.nan)
                own_cash = cap.get(f"capacity_{prefix}_own_cash_prob", np.nan)
                baseline_ret = np.nan
                self_impact_ret = np.nan
                if actual_outcome == "terminated":
                    baseline_ret = terminated_return
                    self_impact_ret = terminated_return
                elif actual_outcome == "withdrawn":
                    baseline_ret = withdrawn_return
                    self_impact_ret = withdrawn_return
                elif np.isfinite(actual_f_cash):
                    baseline_ret = long_realized_ret
                    if np.isfinite(q) and np.isfinite(seller_cash) and np.isfinite(own_cash):
                        f_actual = np.clip(actual_f_cash + q * (own_cash - seller_cash), 1e-6, 1 - 1e-6)
                        cr, sr, _, _ = prorate(np.array([f_actual]), d.pi_cash, d.C, stock_val)
                        cap_realized_value = (cr if elect == "CASH" else sr)[0]
                        self_impact_ret = (cap_realized_value - M) / M
                if np.isfinite(baseline_ret):
                    cap[f"capacity_{prefix}_baseline_realized_return_%"] = baseline_ret * 100
                    cap[f"capacity_{prefix}_baseline_realized_pnl"] = (
                        notional * baseline_ret if np.isfinite(notional) else np.nan
                    )
                if np.isfinite(self_impact_ret):
                    cap[f"capacity_{prefix}_self_impact_realized_return_%"] = self_impact_ret * 100
                    cap[f"capacity_{prefix}_self_impact_realized_pnl"] = (
                        notional * self_impact_ret if np.isfinite(notional) else np.nan
                    )
            cap["capacity_baseline_realized_return_%"] = cap.get("capacity_optimal_baseline_realized_return_%", np.nan)
            cap["capacity_baseline_realized_pnl"] = cap.get("capacity_optimal_baseline_realized_pnl", np.nan)
            cap["capacity_self_impact_realized_return_%"] = cap.get("capacity_optimal_self_impact_realized_return_%", np.nan)
            cap["capacity_self_impact_realized_pnl"] = cap.get("capacity_optimal_self_impact_realized_pnl", np.nan)
            cap["capacity_adjusted_realized_return_%"] = cap["capacity_self_impact_realized_return_%"]
        elif signal == "REVERSE":
            cap = evaluate_reverse_capacity(cap_source, d, M, f, reverse_risk, cfg)
            for prefix in ["max", "optimal"]:
                notional = cap.get(f"capacity_{prefix}_notional", np.nan)
                if np.isfinite(reverse_realized_ret):
                    cap[f"capacity_{prefix}_baseline_realized_return_%"] = reverse_realized_ret * 100
                    cap[f"capacity_{prefix}_baseline_realized_pnl"] = (
                        notional * reverse_realized_ret if np.isfinite(notional) else np.nan
                    )
                    cap[f"capacity_{prefix}_self_impact_realized_return_%"] = reverse_realized_ret * 100
                    cap[f"capacity_{prefix}_self_impact_realized_pnl"] = (
                        notional * reverse_realized_ret if np.isfinite(notional) else np.nan
                    )
            cap["capacity_baseline_realized_return_%"] = cap.get("capacity_optimal_baseline_realized_return_%", np.nan)
            cap["capacity_baseline_realized_pnl"] = cap.get("capacity_optimal_baseline_realized_pnl", np.nan)
            cap["capacity_self_impact_realized_return_%"] = cap.get("capacity_optimal_self_impact_realized_return_%", np.nan)
            cap["capacity_self_impact_realized_pnl"] = cap.get("capacity_optimal_self_impact_realized_pnl", np.nan)
            cap["capacity_adjusted_realized_return_%"] = cap["capacity_self_impact_realized_return_%"]

        rows.append({"event_id": d.event_id, "target": str(d.target_name)[:26], "signal": signal,
                     "signal_reason": signal_reason,
                     "selected_strategy": selected,
                     "elect": elect, "M": round(M, 2), "fair_value": round(selected_fair_value, 2),
                     "risk_adjusted_fair_value": round(selected_risk_adjusted_value, 2),
                     "arb_return_%": round(arb_ret * 100, 2), "E_return_%": round(selected_e_ret * 100, 2),
                     "decision_metric": cfg.decision_metric, "decision_value_%": round(selected_decision * 100, 2),
                     "downside_5%_%": round(selected_risk["p05"] * 100, 2),
                     "cvar_5%_%": round(selected_risk["cvar05"] * 100, 2),
                     "loss_probability_%": round(selected_risk["loss_probability"] * 100, 2),
                     "long_fair_value": round(V, 2),
                     "long_E_return_%": round(long_e_ret * 100, 2),
                     "long_downside_5%_%": round(long_risk["p05"] * 100, 2),
                     "long_loss_probability_%": round(long_risk["loss_probability"] * 100, 2),
                     "reverse_settlement_value": round(passive_settlement, 2),
                     "reverse_completion_return_%": round(reverse_completion_return * 100, 2),
                     "reverse_E_return_%": round(reverse_e_ret * 100, 2),
                     "reverse_downside_5%_%": round(reverse_risk["p05"] * 100, 2),
                     "reverse_loss_probability_%": round(reverse_risk["loss_probability"] * 100, 2),
                     "p_completed": round(normalized_probs["completed"], 4),
                     "p_terminated": round(normalized_probs["terminated"], 4),
                     "p_withdrawn": round(normalized_probs["withdrawn"], 4),
                     "outcome_probability_source": outcome["outcome_probability_source"],
                     "terminated_return_%": round(terminated_return * 100, 2),
                     "withdrawn_return_%": round(withdrawn_return * 100, 2),
                     "terminated_value": round(terminated_value, 2),
                     "withdrawn_value": round(withdrawn_value, 2),
                     "hedge_side": hedge_side,
                     "hedge_ratio": round(selected_hedge_ratio, 3),
                     "long_hedge_ratio_short_acquirer": round(hedge_ratio, 3),
                     "reverse_hedge_ratio_long_acquirer": round(reverse_hedge_ratio, 3),
                     "size_x": round(size, 2),
                     "actual_outcome": actual_outcome,
                     "realized_return_source": realized_source,
                     "realized_return_%": round(realized_ret * 100, 2) if np.isfinite(realized_ret) else np.nan,
                     **{
                         k: (round(v, 4) if isinstance(v, (int, float, np.floating)) and np.isfinite(v) else v)
                         for k, v in cap.items()
                     }})
    out = pd.DataFrame(rows)
    if not out.empty:
        out = out.sort_values("E_return_%", ascending=False)
    out.to_csv(cfg.output_path, index=False)
    # also emit a compact, presentation-ready view. The full file above is the audit-trail
    # appendix (all 100+ diagnostic columns); this is the ~13-column view a human reads. Pure
    # projection, so it can never disagree with the full blotter. Headline return is E_return_%
    # = the SELECTED direction (long for ENTER / reverse for REVERSE), NOT arb_return_% (which is
    # always the long-side view and reads negative on REVERSE trades).
    clean_cols = {
        "target": "target", "signal": "signal", "elect": "elect", "M": "entry",
        "fair_value": "fair_val", "E_return_%": "exp_ret%", "downside_5%_%": "p05%",
        "loss_probability_%": "loss_prob%", "hedge_side": "hedge", "hedge_ratio": "hedge_ratio",
        "size_x": "size", "capacity_notional": "notional_$", "realized_return_%": "realized%",
    }
    present = [c for c in clean_cols if c in out.columns]
    clean_path = cfg.output_path.replace(".csv", "_clean.csv")
    out[present].rename(columns=clean_cols).to_csv(clean_path, index=False)
    return out


def _numeric(df, col):
    if col not in df.columns:
        return pd.Series(np.nan, index=df.index, dtype=float)
    return pd.to_numeric(df[col], errors="coerce")


def _finite_mean(series):
    s = pd.to_numeric(series, errors="coerce").dropna()
    return float(s.mean()) if len(s) else np.nan


def _finite_median(series):
    s = pd.to_numeric(series, errors="coerce").dropna()
    return float(s.median()) if len(s) else np.nan


def _finite_sum(series):
    s = pd.to_numeric(series, errors="coerce").dropna()
    return float(s.sum()) if len(s) else np.nan


def _finite_corr(left, right):
    pair = pd.DataFrame({"left": pd.to_numeric(left, errors="coerce"),
                         "right": pd.to_numeric(right, errors="coerce")}).dropna()
    if len(pair) <= 2:
        return np.nan
    return float(np.corrcoef(pair["left"], pair["right"])[0, 1])


def _notional_weighted_return_pct(df, pnl_col, notional_col):
    pnl = _numeric(df, pnl_col)
    notional = _numeric(df, notional_col)
    ok = np.isfinite(pnl) & np.isfinite(notional) & (notional > 0)
    if not ok.any():
        return np.nan
    return float(pnl[ok].sum() / notional[ok].sum() * 100.0)


def _expected_capacity_block(trades, prefix):
    notional_col = f"capacity_{prefix}_notional"
    pnl_col = f"capacity_{prefix}_expected_pnl"
    ret_col = f"capacity_{prefix}_E_return_%"
    pct_col = f"capacity_{prefix}_pct_shares_outstanding"
    notional = _numeric(trades, notional_col)
    pnl = _numeric(trades, pnl_col)
    ok = np.isfinite(notional) & (notional > 0)
    return {
        "trade_count": int(ok.sum()),
        "total_notional": float(notional[ok].sum()) if ok.any() else np.nan,
        "median_notional": _finite_median(notional[ok]),
        "median_pct_shares_outstanding": _finite_median(_numeric(trades, pct_col)[ok]),
        "total_expected_pnl": _finite_sum(pnl[ok]),
        "unweighted_expected_return_%": _finite_mean(_numeric(trades, ret_col)[ok]),
        "notional_weighted_expected_return_%": _notional_weighted_return_pct(
            trades.loc[ok], pnl_col, notional_col
        ),
    }


def _realized_capacity_block(trades, prefix, mode):
    notional_col = f"capacity_{prefix}_notional"
    ret_col = f"capacity_{prefix}_{mode}_realized_return_%"
    pnl_col = f"capacity_{prefix}_{mode}_realized_pnl"
    ret = _numeric(trades, ret_col)
    pnl = _numeric(trades, pnl_col)
    notional = _numeric(trades, notional_col)
    ok = np.isfinite(ret) & np.isfinite(pnl) & np.isfinite(notional) & (notional > 0)
    losing = ok & (ret <= 0)
    return {
        "covered_trade_count": int(ok.sum()),
        "total_notional_covered": float(notional[ok].sum()) if ok.any() else np.nan,
        "total_realized_pnl": float(pnl[ok].sum()) if ok.any() else np.nan,
        "return_on_deployed_notional_%": float(pnl[ok].sum() / notional[ok].sum() * 100.0)
        if ok.any() and notional[ok].sum() > 0 else np.nan,
        "unweighted_realized_return_%": _finite_mean(ret[ok]),
        "hit_rate_%": float((ret[ok] > 0).mean() * 100.0) if ok.any() else np.nan,
        "wrong_trade_count": int(losing.sum()),
        "wrong_trade_pnl": float(pnl[losing].sum()) if losing.any() else 0.0,
        "correlation_expected_vs_realized": _finite_corr(
            _numeric(trades, f"capacity_{prefix}_E_return_%")[ok], ret[ok]
        ),
    }


def _json_ready(value):
    if isinstance(value, dict):
        return {str(k): _json_ready(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_json_ready(v) for v in value]
    if isinstance(value, tuple):
        return [_json_ready(v) for v in value]
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating, float)):
        x = float(value)
        return x if np.isfinite(x) else None
    if not isinstance(value, (str, bytes, bool)):
        try:
            if pd.isna(value):
                return None
        except Exception:
            pass
    return value


def _risk_adjusted_block(trades):
    """Cross-sectional (per-trade) risk-adjusted performance on REALIZED returns of the traded book.
    Reports equal-weighted mean/median/std, cross-sectional Sharpe (mean/std) and Sortino
    (mean/downside-std), split by ENTER vs REVERSE, plus an ex-largest-winner sensitivity.
    CAVEAT: the universe is completed-only, so there are no losers -> these are UPPER BOUNDS, not
    tradeable Sharpe ratios (see future work: terminated-deal backtest)."""
    def _stats(x):
        x = pd.to_numeric(x, errors="coerce").dropna()
        if len(x) < 1:
            return {"n": 0}
        std = float(x.std())
        downside = x[x < x.mean()]
        dstd = float(downside.std()) if len(downside) > 1 else float("nan")
        return {
            "n": int(len(x)),
            "mean_%": round(float(x.mean()), 2),
            "median_%": round(float(x.median()), 2),
            "std_%": round(std, 2),
            "min_%": round(float(x.min()), 2),
            "max_%": round(float(x.max()), 2),
            "cross_sectional_sharpe": round(float(x.mean() / std), 2) if std > 0 else None,
            "sortino": round(float(x.mean() / dstd), 2) if np.isfinite(dstd) and dstd > 0 else None,
        }
    sig = trades["signal"].astype(str)
    r = _numeric(trades, "realized_return_%").dropna()
    ex = r.drop(r.idxmax()) if len(r) > 1 else r          # drop the single biggest realized winner
    return {
        "note": ("cross-sectional (per-trade) on realized returns; SURVIVORSHIP-BIASED "
                 "(completed-only universe, no losers) -> upper bound, not a tradeable Sharpe"),
        "all_trades": _stats(r),
        "enter": _stats(_numeric(trades[sig.eq("ENTER")], "realized_return_%")),
        "reverse": _stats(_numeric(trades[sig.eq("REVERSE")], "realized_return_%")),
        "all_ex_largest_winner": _stats(ex),
    }


def _render_risk_png(rap, out_path="arb_output/risk_performance.png"):
    """Render the risk_adjusted_performance block as a deck-ready table PNG. Lazy matplotlib import
    so arb_signal.py stays importable without a plotting backend."""
    order = [("all_trades", "All trades"), ("enter", "ENTER"),
             ("reverse", "REVERSE"), ("all_ex_largest_winner", "Ex-top winner")]
    cols = [("n", "n"), ("mean_%", "Mean %"), ("median_%", "Median %"),
            ("std_%", "Std"), ("cross_sectional_sharpe", "Sharpe"), ("sortino", "Sortino")]

    def _fmt(v):
        if v is None or (isinstance(v, float) and not np.isfinite(v)):
            return "—"
        return f"{v:g}"

    labels, rows = [], []
    for k, lbl in order:
        s = rap.get(k) if isinstance(rap.get(k), dict) else None
        if not s or not s.get("n"):
            continue
        labels.append(lbl)
        rows.append([_fmt(s.get(c)) for c, _ in cols])
    if not rows:
        return
    try:
        import os
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception:
        return
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    INK = "#003057"
    fig, ax = plt.subplots(figsize=(7.4, 1.1 + 0.5 * len(rows)))
    ax.axis("off")
    tbl = ax.table(cellText=rows, rowLabels=labels,
                   colLabels=[c[1] for c in cols], loc="center", cellLoc="center")
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(11)
    tbl.scale(1, 1.7)
    for j in range(len(cols)):
        tbl[0, j].set_facecolor(INK)
        tbl[0, j].set_text_props(color="white", weight="bold")
    ax.set_title("Risk-adjusted performance — realized trades (cross-sectional)",
                 fontsize=13, weight="bold", color=INK, pad=16)
    fig.text(0.5, 0.03, "Survivorship-biased (completed-only universe, no losers) "
             "→ upper bound, not a tradeable Sharpe",
             ha="center", fontsize=8, style="italic", color="#434545")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def summarize_strategy(out, output_path="arb_strategy_summary.json"):
    """Summarize ENTER and REVERSE as one combined arbitrage book."""
    if out is None or out.empty or "signal" not in out.columns:
        summary = {
            "priced_signal_count": 0,
            "note": "no_signal_rows",
        }
    else:
        signal = out["signal"].astype(str)
        trades = out[signal.isin(["ENTER", "REVERSE"])].copy()
        pass_rows = out[signal.str.startswith("PASS", na=False)].copy()
        review_rows = out[signal.eq("REVIEW")].copy()
        counts = signal.value_counts().to_dict()

        raw_notional = _numeric(trades, "capacity_raw_max_notional")
        raw_pct = _numeric(trades, "capacity_raw_max_pct_shares_outstanding")
        cap_ok = trades["capacity_status"].eq("ok") if "capacity_status" in trades else pd.Series(False, index=trades.index)
        missed_ret = _numeric(pass_rows, "realized_return_%")
        review_ret = _numeric(review_rows, "realized_return_%")
        optimal_self_ret = _numeric(trades, "capacity_optimal_self_impact_realized_return_%")
        optimal_base_ret = _numeric(trades, "capacity_optimal_baseline_realized_return_%")

        summary = {
            "priced_signal_count": int(len(out)),
            "trade_count": int(len(trades)),
            "enter_count": int(counts.get("ENTER", 0)),
            "reverse_count": int(counts.get("REVERSE", 0)),
            "review_count": int(counts.get("REVIEW", 0)),
            "pass_count": int(sum(v for k, v in counts.items() if str(k).startswith("PASS"))),
            "signal_counts": {str(k): int(v) for k, v in counts.items()},
            "capacity_ok_count": int(cap_ok.sum()),
            "capacity_missing_count": int(len(trades) - cap_ok.sum()),
            "holder_model": {
                "source_counts": trades.get("capacity_holder_model_source", pd.Series(dtype=object)).value_counts(dropna=False).to_dict(),
                "median_fit_n": _finite_median(_numeric(trades, "capacity_holder_fit_n")),
                "median_p_hat": _finite_median(_numeric(trades, "capacity_holder_p_hat")),
                "median_q_hat": _finite_median(_numeric(trades, "capacity_holder_q_hat")),
                "median_positive_share_of_active_%": _finite_median(
                    _numeric(trades, "capacity_positive_holder_share_of_active_%")
                ),
                "median_noisy_cash_prob": _finite_median(_numeric(trades, "capacity_noisy_cash_prob")),
            },
            "raw_maximum": {
                "total_notional": _finite_sum(raw_notional),
                "median_notional": _finite_median(raw_notional),
                "median_pct_shares_outstanding": _finite_median(raw_pct),
            },
            "risk_gated_maximum": _expected_capacity_block(trades, "max"),
            "profit_optimal": _expected_capacity_block(trades, "optimal"),
            "realized_baseline_accept_historical_election": {
                "maximum": _realized_capacity_block(trades, "max", "baseline"),
                "optimal": _realized_capacity_block(trades, "optimal", "baseline"),
            },
            "realized_self_impact_election": {
                "maximum": _realized_capacity_block(trades, "max", "self_impact"),
                "optimal": _realized_capacity_block(trades, "optimal", "self_impact"),
            },
            "trade_quality": {
                "wrong_trade_count_baseline_optimal": int((optimal_base_ret.dropna() <= 0).sum()),
                "wrong_trade_count_self_impact_optimal": int((optimal_self_ret.dropna() <= 0).sum()),
                "missed_profitable_pass_count_marginal_proxy": int((missed_ret.dropna() > 0).sum()),
                "missed_profitable_pass_mean_return_%": _finite_mean(missed_ret[missed_ret > 0]),
                "review_profitable_count_marginal_proxy": int((review_ret.dropna() > 0).sum()),
                "review_profitable_mean_return_%": _finite_mean(review_ret[review_ret > 0]),
                "marginal_proxy_note": (
                    "PASS/REVIEW rows have no executed capacity; missed/review profitability uses the "
                    "selected marginal realized return already printed in arb_signals.csv."
                ),
            },
            "risk_adjusted_performance": _risk_adjusted_block(trades),
            "proof": {
                "long_self_impact_can_eliminate_count": int(
                    out.get("self_impact_can_eliminate_arbitrage", pd.Series(False, index=out.index)).fillna(False).sum()
                ),
                "long_self_impact_break_even_median_pct_shares": _finite_median(
                    _numeric(out[out["signal"].eq("ENTER")], "self_impact_break_even_pct_shares_outstanding")
                ),
                "long_self_impact_break_even_at_or_inside_optimal_count": int((
                    _numeric(trades[trades["signal"].eq("ENTER")], "self_impact_break_even_pct_shares_outstanding")
                    <= _numeric(trades[trades["signal"].eq("ENTER")], "capacity_optimal_pct_shares_outstanding")
                ).sum()),
                "reverse_self_impact_note": (
                    "Under fixed-pool passive blended settlement, reverse payoff is not destroyed by "
                    "its election-demand shift because the short side has no election right and blended "
                    "completion liability is demand-independent."
                ),
            },
            "notes": {
                "maximum_definition": (
                    "capacity_raw_max_* is flow/ADV/position supply. capacity_max_* is the largest "
                    "tradable size that still passes the risk gates after our self-impact. "
                    "capacity_optimal_* maximizes expected dollar PnL over the same feasible grid."
                ),
                "baseline_realized_method": (
                    "Sizes and entry capacity include our market impact, but final settlement accepts "
                    "the historical election result."
                ),
                "self_impact_realized_method": (
                    "Final completed-state election demand is shifted by q*(our election cash "
                    "probability - counterparty cash probability) for ENTER. REVERSE keeps passive "
                    "blended liability because it has no election right."
                ),
            },
        }

        if len(trades):
            for side in ["ENTER", "REVERSE"]:
                side_rows = trades[trades["signal"].eq(side)]
                summary[f"{side.lower()}_profit_optimal"] = _expected_capacity_block(side_rows, "optimal")
                summary[f"{side.lower()}_realized_self_impact_optimal"] = _realized_capacity_block(
                    side_rows, "optimal", "self_impact"
                )

    summary = _json_ready(summary)
    if output_path:
        path = Path(output_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
        csv_path = path.with_suffix(".csv")
        pd.json_normalize(summary, sep=".").to_csv(csv_path, index=False)
        _render_risk_png(summary.get("risk_adjusted_performance", {}))
    return summary


def parse_args():
    p = argparse.ArgumentParser(description="Build risk-aware cash-or-stock election-arb signals.")
    p.add_argument("--deals", default="arb_deals.csv", help="Input deal terms table from arb_terms.py")
    p.add_argument("--market-daily", default="ma_market_wrds/wrds_market_daily.csv",
                   help="WRDS daily market file used for entry/recovery prices")
    p.add_argument("--capacity-panel", default="eda_output/merged_panel.csv",
                   help="Deal panel with target shares, ADV, and passive ownership for capacity sizing")
    p.add_argument("--out", default="arb_signals.csv", help="Output signal blotter")
    p.add_argument("--summary-out", default="arb_strategy_summary.json",
                   help="Combined long/reverse strategy summary output")
    p.add_argument("--outcome-probs", default="",
                   help="Optional event_id-level completed/terminated/withdrawn probability CSV")
    p.add_argument("--n-draws", type=int, default=NDRAW)
    p.add_argument("--hurdle", type=float, default=HURDLE)
    p.add_argument("--decision-metric", choices=["mean", "p05", "cvar05"], default="mean")
    p.add_argument("--min-p05-return", type=float, default=MIN_P05_RETURN)
    p.add_argument("--max-loss-probability", type=float, default=MAX_LOSS_PROBABILITY)
    p.add_argument("--default-completed-prob", type=float, default=DEFAULT_P_COMPLETED)
    p.add_argument("--default-terminated-prob", type=float, default=DEFAULT_P_TERMINATED)
    p.add_argument("--default-withdrawn-prob", type=float, default=DEFAULT_P_WITHDRAWN)
    p.add_argument("--withdrawn-share-of-break-prob", type=float, default=0.35,
                   help="Split used when only aggregate p_break is available")
    p.add_argument("--terminated-loss-frac", type=float, default=TERMINATED_LOSS_FRAC)
    p.add_argument("--withdrawn-loss-frac", type=float, default=WITHDRAWN_LOSS_FRAC)
    p.add_argument("--holder-model", choices=["rolling_structural", "static"], default="rolling_structural",
                   help="Estimate positive/noisy holder mix from prior events or use static fallback")
    p.add_argument("--holder-rolling-window-events", type=int, default=50,
                   help="Prior labeled events used for rolling structural holder fit")
    p.add_argument("--holder-min-fit-events", type=int, default=10,
                   help="Minimum prior labeled events before rolling holder fit is used")
    p.add_argument("--holder-p-grid-size", type=int, default=11,
                   help="Grid size for noisy/irrational cash-election probability p")
    p.add_argument("--holder-q-grid-size", type=int, default=11,
                   help="Grid size for EV-sensitive rational ownership share q")
    p.add_argument("--default-irrational-cash-prob", type=float, default=0.50,
                   help="Fallback p before enough rolling observations exist")
    p.add_argument("--default-rational-share", type=float, default=0.30,
                   help="Fallback q before enough rolling observations exist")
    p.add_argument("--capacity-build-days", type=float, default=10.0)
    p.add_argument("--capacity-max-adv-participation", type=float, default=0.20)
    p.add_argument("--capacity-max-position-pct-shares", type=float, default=0.05)
    p.add_argument("--positive-holder-share-of-active", type=float, default=0.35)
    p.add_argument("--positive-no-sell-edge", type=float, default=0.03,
                   help="Positive holders are assumed not to sell once long EV exceeds this return")
    p.add_argument("--capacity-grid-points", type=int, default=51)
    return p.parse_args()


def config_from_args(args):
    return SignalConfig(
        deals_path=args.deals,
        market_daily_path=args.market_daily,
        capacity_panel_path=args.capacity_panel,
        output_path=args.out,
        summary_output_path=args.summary_out,
        outcome_probs_path=args.outcome_probs,
        n_draws=args.n_draws,
        hurdle=args.hurdle,
        decision_metric=args.decision_metric,
        min_p05_return=args.min_p05_return,
        max_loss_probability=args.max_loss_probability,
        default_completed_prob=args.default_completed_prob,
        default_terminated_prob=args.default_terminated_prob,
        default_withdrawn_prob=args.default_withdrawn_prob,
        withdrawn_share_of_break_prob=args.withdrawn_share_of_break_prob,
        terminated_loss_frac=args.terminated_loss_frac,
        withdrawn_loss_frac=args.withdrawn_loss_frac,
        holder_model=args.holder_model,
        holder_rolling_window_events=args.holder_rolling_window_events,
        holder_min_fit_events=args.holder_min_fit_events,
        holder_p_grid_size=args.holder_p_grid_size,
        holder_q_grid_size=args.holder_q_grid_size,
        default_irrational_cash_prob=args.default_irrational_cash_prob,
        default_rational_share=args.default_rational_share,
        capacity_build_days=args.capacity_build_days,
        capacity_max_adv_participation=args.capacity_max_adv_participation,
        capacity_max_position_pct_shares=args.capacity_max_position_pct_shares,
        positive_holder_share_of_active=args.positive_holder_share_of_active,
        positive_no_sell_edge=args.positive_no_sell_edge,
        capacity_grid_points=args.capacity_grid_points,
    )


if __name__ == "__main__":
    cfg = config_from_args(parse_args())
    out = build_signals(cfg)
    summary = summarize_strategy(out, cfg.summary_output_path)
    has_signal = "signal" in out.columns
    ent = out[out["signal"] == "ENTER"] if has_signal else pd.DataFrame()
    rev = out[out["signal"] == "REVERSE"] if has_signal else pd.DataFrame()
    trades = out[out["signal"].isin(["ENTER", "REVERSE"])] if has_signal else pd.DataFrame()
    print(
        f"=== TRADE BLOTTER ({len(out)} deals, {len(ent)} ENTER, {len(rev)} REVERSE, "
        f"{len(trades)} total trades) ===\n"
    )
    cols = ["target", "signal", "selected_strategy", "signal_reason", "elect", "M",
            "fair_value", "risk_adjusted_fair_value", "E_return_%",
            "long_E_return_%", "reverse_E_return_%", "downside_5%_%",
            "loss_probability_%", "p_completed", "p_terminated",
            "p_withdrawn", "hedge_side", "hedge_ratio", "size_x",
            "capacity_status", "capacity_raw_max_notional", "capacity_max_notional",
            "capacity_optimal_notional", "capacity_max_E_return_%",
            "capacity_optimal_E_return_%", "capacity_optimal_expected_pnl",
            "capacity_max_pct_shares_outstanding", "capacity_optimal_pct_shares_outstanding",
            "capacity_max_binding_constraint", "capacity_optimal_binding_constraint",
            "capacity_holder_model_source", "capacity_holder_fit_n",
            "capacity_holder_p_hat", "capacity_holder_q_hat",
            "capacity_positive_holder_share_of_active_%",
            "capacity_noisy_cash_prob", "capacity_optimal_cash_demand_shift_pctpts",
            "realized_return_%",
            "capacity_optimal_baseline_realized_return_%",
            "capacity_optimal_self_impact_realized_return_%",
            "capacity_optimal_baseline_realized_pnl",
            "capacity_optimal_self_impact_realized_pnl",
            "self_impact_can_eliminate_arbitrage",
            "self_impact_break_even_pct_shares_outstanding"]
    if len(out):
        cols = [c for c in cols if c in out.columns]
        print(out[cols].to_string(index=False))
    if len(trades):
        realized = trades["realized_return_%"].dropna()
        print(f"\nTrade book: mean E[return]={trades['E_return_%'].mean():.2f}%  "
              f"mean realized={realized.mean():.2f}%  "
              f"hit rate (realized>0)={(realized>0).mean()*100:.0f}%")
        if "capacity_notional" in trades.columns:
            known_capacity = trades[trades["capacity_status"].eq("ok")]
            cap_realized = trades["capacity_adjusted_realized_return_%"].dropna()
            print(f"capacity: {len(known_capacity)}/{len(trades)} trades sized; "
                  f"median notional=${known_capacity['capacity_notional'].median():,.0f}  "
                  f"median ownership={known_capacity['capacity_pct_shares_outstanding'].median():.2f}%")
            print(f"capacity-adjusted: mean E={trades['capacity_adjusted_E_return_%'].mean():.2f}%  "
                  f"mean realized={cap_realized.mean():.2f}%  "
                  f"hit rate={(cap_realized>0).mean()*100:.0f}%")
        if len(ent):
            ent_realized = ent["realized_return_%"].dropna()
            print(f"  ENTER:   n={len(ent):2d} mean E={ent['E_return_%'].mean():.2f}% "
                  f"mean realized={ent_realized.mean():.2f}%")
            if "capacity_adjusted_realized_return_%" in ent:
                ent_cap_realized = ent["capacity_adjusted_realized_return_%"].dropna()
                print(f"           capacity median=${ent['capacity_notional'].median():,.0f}  "
                      f"cap-adj mean E={ent['capacity_adjusted_E_return_%'].mean():.2f}%  "
                      f"cap-adj realized={ent_cap_realized.mean():.2f}%")
        if len(rev):
            rev_realized = rev["realized_return_%"].dropna()
            print(f"  REVERSE: n={len(rev):2d} mean E={rev['E_return_%'].mean():.2f}% "
                  f"mean realized={rev_realized.mean():.2f}%")
            if "capacity_adjusted_realized_return_%" in rev:
                rev_cap_realized = rev["capacity_adjusted_realized_return_%"].dropna()
                print(f"           capacity median=${rev['capacity_notional'].median():,.0f}  "
                      f"cap-adj mean E={rev['capacity_adjusted_E_return_%'].mean():.2f}%  "
                      f"cap-adj realized={rev_cap_realized.mean():.2f}%")
        # does the signal have skill? corr of predicted vs realized
        paired = trades[["E_return_%", "realized_return_%"]].dropna()
        c = np.corrcoef(paired["E_return_%"], paired["realized_return_%"])[0, 1] if len(paired) > 2 else np.nan
        print(f"signal skill: corr(E[return], realized) = {c:+.2f}")
        if "capacity_adjusted_E_return_%" in trades:
            cap_paired = trades[["capacity_adjusted_E_return_%", "capacity_adjusted_realized_return_%"]].dropna()
            cap_c = (
                np.corrcoef(cap_paired["capacity_adjusted_E_return_%"],
                            cap_paired["capacity_adjusted_realized_return_%"])[0, 1]
                if len(cap_paired) > 2 else np.nan
            )
            print(f"capacity-adjusted signal skill: corr(E[return], realized) = {cap_c:+.2f}")

    def _money(value):
        return "n/a" if value is None or not np.isfinite(float(value)) else f"${float(value):,.0f}"

    def _pct(value):
        return "n/a" if value is None or not np.isfinite(float(value)) else f"{float(value):.2f}%"

    opt = summary.get("profit_optimal", {})
    mx = summary.get("risk_gated_maximum", {})
    rb = summary.get("realized_baseline_accept_historical_election", {}).get("optimal", {})
    rs = summary.get("realized_self_impact_election", {}).get("optimal", {})
    tq = summary.get("trade_quality", {})
    proof = summary.get("proof", {})
    print(f"\n=== COMBINED STRATEGY SUMMARY ({cfg.summary_output_path}) ===")
    print(
        "optimal sizing: "
        f"notional={_money(opt.get('total_notional'))}  "
        f"E[pnl]={_money(opt.get('total_expected_pnl'))}  "
        f"weighted E[return]={_pct(opt.get('notional_weighted_expected_return_%'))}"
    )
    print(
        "risk-gated maximum: "
        f"notional={_money(mx.get('total_notional'))}  "
        f"E[pnl]={_money(mx.get('total_expected_pnl'))}  "
        f"weighted E[return]={_pct(mx.get('notional_weighted_expected_return_%'))}"
    )
    print(
        "realized baseline (accept historical election): "
        f"pnl={_money(rb.get('total_realized_pnl'))}  "
        f"return on deployed={_pct(rb.get('return_on_deployed_notional_%'))}  "
        f"wrong={rb.get('wrong_trade_count', 0)}"
    )
    print(
        "realized self-impact election: "
        f"pnl={_money(rs.get('total_realized_pnl'))}  "
        f"return on deployed={_pct(rs.get('return_on_deployed_notional_%'))}  "
        f"wrong={rs.get('wrong_trade_count', 0)}"
    )
    print(
        "trade quality: "
        f"missed profitable PASS rows={tq.get('missed_profitable_pass_count_marginal_proxy', 0)}  "
        f"profitable REVIEW rows={tq.get('review_profitable_count_marginal_proxy', 0)}"
    )
    print(
        "self-impact proof: "
        f"long elimination possible in pure-noisy stress={proof.get('long_self_impact_can_eliminate_count', 0)}  "
        f"median break-even={_pct(proof.get('long_self_impact_break_even_median_pct_shares'))}"
    )
