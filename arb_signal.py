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
import numpy as np
import pandas as pd
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
    output_path: str = "arb_signals.csv"
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
        realized_val, normalized_probs = apply_outcome_overlay(
            held,
            entry_value=M,
            p_completed=outcome["p_completed"],
            p_terminated=outcome["p_terminated"],
            p_withdrawn=outcome["p_withdrawn"],
            terminated_value=terminated_value,
            withdrawn_value=withdrawn_value,
            rng=rng,
        )
        ret_dist = (realized_val - M) / M
        risk = summarize_returns(ret_dist)
        decision = decision_value(risk, cfg.decision_metric)
        dn5 = risk["p05"]
        cvar5 = risk["cvar05"]
        loss_probability = risk["loss_probability"]
        e_ret = risk["mean"]
        risk_adjusted_fair_value = M * (1.0 + e_ret)
        terminated_return = (terminated_value - M) / M
        withdrawn_return = (withdrawn_value - M) / M

        # realized (historical) outcome given the actual election demand
        actual_outcome = normalize_outcome_label(d)
        actual_f_cash = pd.to_numeric(pd.Series([d.get("f_cash", np.nan)]), errors="coerce").iloc[0]
        if actual_outcome == "terminated":
            realized_value = terminated_value
            realized_ret = terminated_return
            realized_source = "deal_outcome_label"
        elif actual_outcome == "withdrawn":
            realized_value = withdrawn_value
            realized_ret = withdrawn_return
            realized_source = "deal_outcome_label"
        elif np.isfinite(actual_f_cash):
            cr, sr, _, _ = prorate(np.array([actual_f_cash]), d.pi_cash, d.C, stock_val)
            realized_value = (cr if elect == "CASH" else sr)[0]
            realized_ret = (realized_value - M) / M
            realized_source = "realized_f_cash"
            if not actual_outcome:
                actual_outcome = "completed"
        else:
            realized_value = np.nan
            realized_ret = np.nan
            realized_source = "missing_realized_outcome"

        # data-quality guard: a real post-announcement merger-arb spread lives in ~[-30%, +30%].
        # anything outside that is almost certainly a bad entry price or misparsed term -> REVIEW,
        # never a tradeable signal. keeps the blotter honest by construction.
        if abs(arb_ret) > 0.30:
            signal, size = "REVIEW", 0.0
        elif decision > cfg.hurdle and dn5 >= cfg.min_p05_return and loss_probability <= cfg.max_loss_probability:
            risk_scale = max(-cvar5, -dn5, 1e-3)
            signal, size = "ENTER", float(np.clip(e_ret / risk_scale, 0, 3))
        elif decision <= cfg.hurdle:
            signal, size = "PASS", 0.0
        elif dn5 < cfg.min_p05_return:
            signal, size = "PASS_P05", 0.0
        else:
            signal, size = "PASS_LOSS_PROB", 0.0

        rows.append({"event_id": d.event_id, "target": str(d.target_name)[:26], "signal": signal,
                     "elect": elect, "M": round(M, 2), "fair_value": round(V, 2),
                     "risk_adjusted_fair_value": round(risk_adjusted_fair_value, 2),
                     "arb_return_%": round(arb_ret * 100, 2), "E_return_%": round(e_ret * 100, 2),
                     "decision_metric": cfg.decision_metric, "decision_value_%": round(decision * 100, 2),
                     "downside_5%_%": round(dn5 * 100, 2), "cvar_5%_%": round(cvar5 * 100, 2),
                     "loss_probability_%": round(loss_probability * 100, 2),
                     "p_completed": round(normalized_probs["completed"], 4),
                     "p_terminated": round(normalized_probs["terminated"], 4),
                     "p_withdrawn": round(normalized_probs["withdrawn"], 4),
                     "outcome_probability_source": outcome["outcome_probability_source"],
                     "terminated_return_%": round(terminated_return * 100, 2),
                     "withdrawn_return_%": round(withdrawn_return * 100, 2),
                     "terminated_value": round(terminated_value, 2),
                     "withdrawn_value": round(withdrawn_value, 2),
                     "hedge_ratio": round(hedge_ratio, 3),
                     "size_x": round(size, 2),
                     "actual_outcome": actual_outcome,
                     "realized_return_source": realized_source,
                     "realized_return_%": round(realized_ret * 100, 2) if np.isfinite(realized_ret) else np.nan})
    out = pd.DataFrame(rows)
    if not out.empty:
        out = out.sort_values("E_return_%", ascending=False)
    out.to_csv(cfg.output_path, index=False)
    return out


def parse_args():
    p = argparse.ArgumentParser(description="Build risk-aware cash-or-stock election-arb signals.")
    p.add_argument("--deals", default="arb_deals.csv", help="Input deal terms table from arb_terms.py")
    p.add_argument("--market-daily", default="ma_market_wrds/wrds_market_daily.csv",
                   help="WRDS daily market file used for entry/recovery prices")
    p.add_argument("--out", default="arb_signals.csv", help="Output signal blotter")
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
    return p.parse_args()


def config_from_args(args):
    return SignalConfig(
        deals_path=args.deals,
        market_daily_path=args.market_daily,
        output_path=args.out,
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
    )


if __name__ == "__main__":
    out = build_signals(config_from_args(parse_args()))
    ent = out[out["signal"] == "ENTER"] if "signal" in out.columns else pd.DataFrame()
    print(f"=== TRADE BLOTTER ({len(out)} deals, {len(ent)} ENTER) ===\n")
    cols = ["target", "signal", "elect", "M", "fair_value", "risk_adjusted_fair_value",
            "arb_return_%", "E_return_%", "downside_5%_%", "cvar_5%_%",
            "loss_probability_%", "p_completed", "p_terminated", "p_withdrawn",
            "hedge_ratio", "size_x", "realized_return_%"]
    if len(out):
        print(out[cols].to_string(index=False))
    if len(ent):
        realized = ent["realized_return_%"].dropna()
        print(f"\nENTER book: mean E[return]={ent['E_return_%'].mean():.2f}%  "
              f"mean realized={realized.mean():.2f}%  "
              f"hit rate (realized>0)={(realized>0).mean()*100:.0f}%")
        # does the signal have skill? corr of predicted vs realized
        paired = ent[["E_return_%", "realized_return_%"]].dropna()
        c = np.corrcoef(paired["E_return_%"], paired["realized_return_%"])[0, 1] if len(paired) > 2 else np.nan
        print(f"signal skill: corr(E[return], realized) = {c:+.2f}")
