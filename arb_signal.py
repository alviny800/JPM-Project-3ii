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
  SIGNAL    arb return = (V - M)/M ; overlay deal-break risk -> expected return + downside
            ENTER if expected return clears the hurdle, else PASS ; rank + size by edge/risk

This is a forward-looking generator: point it at a live deal's terms+prices and it emits the
trade. Run on our historical deals it also prints the REALIZED outcome next to each signal.

Caveats (flagged, not hidden): the deals priced here all closed (survivorship) — the deal-break
term is a scenario overlay, not calibrated on this subset; a break-inclusive P&L backtest needs
the terminated deals. Hedge assumes the expected stock fraction (residual hedging error ignored).
"""
from __future__ import annotations
import numpy as np
import pandas as pd
from arb_mc import DemandModel, prorate

P_BREAK, RECOVERY_LAG, ENTRY_LAG = 0.12, 12, 5     # break prob; pre-announce days; post-announce entry days
HURDLE = 0.005                                       # 0.5% expected return to enter
NDRAW = 20000


def price_on(df, date, mode="onbefore"):
    df = df.dropna(subset=["price_date", "price"]).sort_values("price_date")
    if mode == "onbefore":
        s = df[df.price_date <= date]
        return s["price"].iloc[-1] if len(s) else np.nan
    s = df[df.price_date >= date]
    return s["price"].iloc[0] if len(s) else np.nan


def build_signals():
    arb = pd.read_csv("arb_deals.csv")
    ready = arb.dropna(subset=["C", "R", "P_acq", "f_cash", "pi_cash"])
    ready = ready[ready.ratio_type == "fixed"].copy()

    daily = pd.read_csv("ma_market_wrds/wrds_market_daily.csv")
    daily["price_date"] = pd.to_datetime(daily["price_date"], errors="coerce")
    daily["announce_date"] = pd.to_datetime(daily["announce_date"], errors="coerce")

    model = DemandModel(arb["f_cash"].dropna().values)          # the single default demand model
    rng = np.random.default_rng(11)
    f = model.draw(NDRAW, rng=rng)                               # one shared demand sample

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
        break_loss = max(M - recovery, 0) if np.isfinite(recovery) else 0.30 * M
        e_pnl = (1 - P_BREAK) * arb_spread - P_BREAK * break_loss
        e_ret = e_pnl / M
        # downside: 5th pct of realized return with the break overlay
        broke = rng.random(NDRAW) < P_BREAK
        realized_val = np.where(broke, M - break_loss, held)
        ret_dist = (realized_val - M) / M
        dn5 = np.percentile(ret_dist, 5)

        # realized (historical) outcome given the actual election demand
        cr, sr, _, _ = prorate(np.array([d.f_cash]), d.pi_cash, d.C, stock_val)
        realized_value = (cr if elect == "CASH" else sr)[0]
        realized_ret = (realized_value - M) / M

        # data-quality guard: a real post-announcement merger-arb spread lives in ~[-30%, +30%].
        # anything outside that is almost certainly a bad entry price or misparsed term -> REVIEW,
        # never a tradeable signal. keeps the blotter honest by construction.
        if abs(arb_ret) > 0.30:
            signal, size = "REVIEW", 0.0
        elif e_ret > HURDLE:
            signal, size = "ENTER", float(np.clip(e_ret / max(-dn5, 1e-3), 0, 3))
        else:
            signal, size = "PASS", 0.0

        rows.append({"event_id": d.event_id, "target": str(d.target_name)[:26], "signal": signal,
                     "elect": elect, "M": round(M, 2), "fair_value": round(V, 2),
                     "arb_return_%": round(arb_ret * 100, 2), "E_return_%": round(e_ret * 100, 2),
                     "downside_5%_%": round(dn5 * 100, 2), "hedge_ratio": round(hedge_ratio, 3),
                     "size_x": round(size, 2), "realized_return_%": round(realized_ret * 100, 2)})
    out = pd.DataFrame(rows).sort_values("E_return_%", ascending=False)
    out.to_csv("arb_signals.csv", index=False)
    return out


if __name__ == "__main__":
    out = build_signals()
    ent = out[out.signal == "ENTER"]
    print(f"=== TRADE BLOTTER ({len(out)} deals, {len(ent)} ENTER) ===\n")
    cols = ["target", "signal", "elect", "M", "fair_value", "arb_return_%", "E_return_%",
            "downside_5%_%", "hedge_ratio", "size_x", "realized_return_%"]
    print(out[cols].to_string(index=False))
    if len(ent):
        print(f"\nENTER book: mean E[return]={ent['E_return_%'].mean():.2f}%  "
              f"mean realized={ent['realized_return_%'].mean():.2f}%  "
              f"hit rate (realized>0)={(ent['realized_return_%']>0).mean()*100:.0f}%")
        # does the signal have skill? corr of predicted vs realized
        c = np.corrcoef(ent["E_return_%"], ent["realized_return_%"])[0, 1] if len(ent) > 2 else np.nan
        print(f"signal skill: corr(E[return], realized) = {c:+.2f}")
