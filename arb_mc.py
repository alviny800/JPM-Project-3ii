#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
arb_mc.py  —  MONTE CARLO ENGINE for cash-or-stock election arbitrage.

The ONLY stochastic node in the payoff chain is aggregate election demand (f_cash = the
fraction of shares that elect cash at the deadline). Everything downstream is deterministic
given the deal terms. So the model is:

    draw f_cash  ->  proration mechanics  ->  optimal-election consideration  ->  edge / P&L
    (repeat N times -> a distribution)

The trade layer can then overlay completed/terminated/withdrawn state probabilities.  The
MC engine keeps the older aggregate p_break interface for compatibility, but the newer
three-state inputs are preferred when available.

Two economic facts drive the whole thing:
  1. In a fully-prorated deal the *blended* (average) consideration is FIXED by the cash pool
     pi_cash:   blended = pi_cash*C + (1-pi_cash)*stock_val   — independent of demand.
  2. The arb edge comes from OPTIMAL ELECTION: elect the richer side; if that side is
     *under-subscribed* you capture more of it than the blended average. How much you capture
     depends on the demand realization -> that is what we simulate.

Demand model:
  - Unconditional: Beta fit (method of moments) to the realized f_cash across the 72 deals,
    plus the raw empirical sample for a nonparametric draw.
  - Spread-conditioned (optional): E[f_cash | spread] = logistic(a + b*spread). Rational
    holders tilt toward the richer side, so b>0 is the prior; we CALIBRATE (a,b) on the
    fixed-ratio deals. NOTE: on our data b is ~0 (flat) — the framework supports conditioning,
    the data just says the tilt is weak. We MC over parameter uncertainty rather than assert it.
"""
from __future__ import annotations
import numpy as np
import pandas as pd


# ----------------------------- demand model -----------------------------
def fit_beta(x):
    """Method-of-moments Beta(a,b) for a sample of fractions in (0,1)."""
    x = np.asarray(x, float)
    x = x[(x > 0) & (x < 1)]
    if len(x) < 5:
        return (1.0, 1.0)          # uninformative fallback
    m, v = x.mean(), x.var(ddof=1)
    if v <= 0 or v >= m * (1 - m):
        return (max(m * 20, .5), max((1 - m) * 20, .5))
    c = m * (1 - m) / v - 1
    return (m * c, (1 - m) * c)


def fit_conditional(spread, f_cash):
    """Logistic slope of demand on spread: logit(f) = a + b*spread. Returns (a,b, se_b)."""
    s = np.asarray(spread, float); f = np.asarray(f_cash, float)
    ok = np.isfinite(s) & np.isfinite(f) & (f > 0) & (f < 1)
    s, f = s[ok], f[ok]
    if len(s) < 8:
        return (0.0, 0.0, np.nan)
    y = np.log(f / (1 - f))
    b, a = np.polyfit(s, y, 1)
    resid = y - (a + b * s)
    se_b = np.sqrt((resid.var(ddof=2)) / ((s - s.mean()) ** 2).sum()) if len(s) > 2 else np.nan
    return (a, b, se_b)


class DemandModel:
    """Unconditional Beta/empirical + optional spread conditioning with parameter uncertainty."""
    def __init__(self, f_cash_sample, spread=None):
        self.sample = np.asarray([v for v in f_cash_sample if np.isfinite(v)], float)
        self.a, self.b = fit_beta(self.sample)
        self.la, self.lb, self.lb_se = (0.0, 0.0, np.nan)
        if spread is not None:
            self.la, self.lb, self.lb_se = fit_conditional(spread, f_cash_sample)

    def draw(self, n, spread=None, rng=None, condition=False, param_uncertainty=True):
        rng = rng or np.random.default_rng(12345)
        if condition and np.isfinite(spread) and np.isfinite(self.lb):
            lb = self.lb
            if param_uncertainty and np.isfinite(self.lb_se):
                lb = rng.normal(self.lb, self.lb_se)          # MC over the (thin) slope estimate
            mu = 1 / (1 + np.exp(-(self.la + lb * spread)))
            # keep the fitted dispersion, recenter to the conditional mean
            k = self.a + self.b
            a2, b2 = max(mu * k, .5), max((1 - mu) * k, .5)
            return rng.beta(a2, b2, n)
        return rng.beta(self.a, self.b, n)


# --------------------------- proration mechanics ---------------------------
def prorate(f_cash, pi_cash, C, stock_val):
    """
    Given aggregate cash demand f_cash and the fixed cash pool pi_cash, return per-share
    consideration for (cash-electing holder, stock-electing holder, blended average,
    optimal-election holder). Vectorized over f_cash.
    """
    f_cash = np.clip(np.asarray(f_cash, float), 1e-6, 1 - 1e-6)
    pi_stock = 1 - pi_cash
    f_stock = 1 - f_cash

    # cash over-subscribed -> cash-electors prorated toward stock; else full cash
    cash_fill = np.minimum(1.0, pi_cash / f_cash)            # frac of a cash-elector's shares paid cash
    cash_holder = cash_fill * C + (1 - cash_fill) * stock_val
    # stock over-subscribed -> stock-electors prorated toward cash; else full stock
    stock_fill = np.minimum(1.0, pi_stock / f_stock)
    stock_holder = stock_fill * stock_val + (1 - stock_fill) * C

    blended = pi_cash * C + pi_stock * stock_val            # fixed, demand-independent
    optimal = np.maximum(cash_holder, stock_holder)         # you elect the richer realized side
    return cash_holder, stock_holder, blended, optimal


def normalize_state_probabilities(p_completed=None, p_terminated=None, p_withdrawn=None, p_break=0.0):
    """Normalize completed/terminated/withdrawn probabilities.

    Backward compatibility: callers that only pass p_break get the old two-state
    behavior, represented as all break probability in the terminated bucket.
    """
    if p_terminated is None and p_withdrawn is None:
        pb = float(np.clip(p_break, 0.0, 1.0))
        p_completed = 1.0 - pb if p_completed is None else p_completed
        p_terminated = pb
        p_withdrawn = 0.0
    else:
        p_terminated = 0.0 if p_terminated is None else p_terminated
        p_withdrawn = 0.0 if p_withdrawn is None else p_withdrawn
        p_completed = 1.0 - p_terminated - p_withdrawn if p_completed is None else p_completed

    probs = np.array([p_completed, p_terminated, p_withdrawn], dtype=float)
    probs = np.clip(np.nan_to_num(probs, nan=0.0), 0.0, 1.0)
    total = probs.sum()
    if total <= 0:
        probs = np.array([1.0, 0.0, 0.0])
    else:
        probs = probs / total
    return {"completed": float(probs[0]), "terminated": float(probs[1]), "withdrawn": float(probs[2])}


def apply_outcome_overlay(complete_values, entry_value, p_completed=None, p_terminated=None, p_withdrawn=None,
                          p_break=0.0, terminated_value=None, withdrawn_value=None,
                          break_loss_frac=0.25, terminated_loss_frac=0.25,
                          withdrawn_loss_frac=0.35, rng=None):
    """Map completion payoff draws into a three-state realized payoff distribution."""
    rng = rng or np.random.default_rng(7)
    complete_values = np.asarray(complete_values, float)
    entry_value = float(entry_value)
    probs = normalize_state_probabilities(
        p_completed=p_completed,
        p_terminated=p_terminated,
        p_withdrawn=p_withdrawn,
        p_break=p_break,
    )
    if terminated_value is None:
        loss = break_loss_frac if p_terminated is None and p_withdrawn is None else terminated_loss_frac
        terminated_value = entry_value * (1.0 - loss)
    if withdrawn_value is None:
        loss = break_loss_frac if p_terminated is None and p_withdrawn is None else withdrawn_loss_frac
        withdrawn_value = entry_value * (1.0 - loss)

    u = rng.random(len(complete_values))
    realized = np.where(
        u < probs["completed"],
        complete_values,
        np.where(u < probs["completed"] + probs["terminated"], terminated_value, withdrawn_value),
    )
    return realized, probs


# ------------------------------ simulation ------------------------------
def simulate_deal(deal, model, n=20000, condition=False, p_break=0.0, break_loss_frac=0.25,
                  p_completed=None, p_terminated=None, p_withdrawn=None,
                  terminated_loss_frac=0.25, withdrawn_loss_frac=0.35,
                  entry_value=None, terminated_value=None, withdrawn_value=None,
                  rng=None):
    """
    MC one deal. Returns a dict of the consideration/edge/return distributions.
    edge = optimal-election consideration - blended average  (the proration-capture alpha, per share)
    If p_break>0, overlay the legacy two-state break scenario.  If
    p_terminated/p_withdrawn are supplied, overlay a completed/terminated/withdrawn
    state tree instead.
    """
    rng = rng or np.random.default_rng(7)
    C, R, Pacq, pi = deal["C"], deal["R"], deal["P_acq"], deal["pi_cash"]
    stock_val = R * Pacq
    f = model.draw(n, spread=deal.get("spread", np.nan), rng=rng, condition=condition)
    cash_h, stock_h, blended, optimal = prorate(f, pi, C, stock_val)
    edge = optimal - blended
    entry_value = blended if entry_value is None else entry_value
    realized, probs = apply_outcome_overlay(
        optimal,
        entry_value=entry_value,
        p_completed=p_completed,
        p_terminated=p_terminated,
        p_withdrawn=p_withdrawn,
        p_break=p_break,
        terminated_value=terminated_value,
        withdrawn_value=withdrawn_value,
        break_loss_frac=break_loss_frac,
        terminated_loss_frac=terminated_loss_frac,
        withdrawn_loss_frac=withdrawn_loss_frac,
        rng=rng,
    )
    return {"f_cash": f, "cash_holder": cash_h, "stock_holder": stock_h,
            "blended": float(blended), "optimal": optimal, "edge": edge, "realized": realized,
            "p_completed": probs["completed"], "p_terminated": probs["terminated"],
            "p_withdrawn": probs["withdrawn"], "spread": deal.get("spread", np.nan),
            "stock_val": float(stock_val)}


def summarize(sim):
    e = sim["edge"]; r = sim["realized"]; b = sim["blended"]
    return {"blended": b, "edge_mean": float(e.mean()), "edge_p50": float(np.median(e)),
            "edge_p05": float(np.percentile(e, 5)), "edge_p95": float(np.percentile(e, 95)),
            "edge_pct_of_blended": float(e.mean() / b * 100) if b else np.nan,
            "realized_mean": float(r.mean()), "realized_p05": float(np.percentile(r, 5))}


if __name__ == "__main__":
    # smoke test on the MC-ready deals
    d = pd.read_csv("arb_deals.csv")
    cal = d["f_cash"].dropna().values
    spr = d.dropna(subset=["f_cash", "spread"])
    model = DemandModel(cal, spread=None)
    model_c = DemandModel(spr["f_cash"].values, spread=spr["spread"].values)
    print(f"[mc] demand Beta(a={model.a:.2f}, b={model.b:.2f}) fit on {len(cal)} realized demands; "
          f"mean={model.a/(model.a+model.b):.2f}")
    print(f"[mc] spread-conditioning logit slope b={model_c.lb:+.4f} (se={model_c.lb_se:.4f}) "
          f"-> {'~flat (weak tilt)' if abs(model_c.lb) < 2*(model_c.lb_se or 1) else 'material'}")
    ready = d.dropna(subset=["C", "R", "P_acq", "f_cash"])
    ready = ready[ready.ratio_type == "fixed"]
    print(f"[mc] MC-ready fixed deals: {len(ready)}")
    ex = ready.iloc[0]
    s = summarize(simulate_deal(ex, model))
    print(f"[mc] example {ex['target_name'][:30]}: blended={s['blended']:.2f} "
          f"edge_mean={s['edge_mean']:.3f} ({s['edge_pct_of_blended']:.1f}% of blended)")
