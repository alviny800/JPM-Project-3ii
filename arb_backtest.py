#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
arb_backtest.py  —  validation of the election-arb framework against realized history.

Two backtests that run on data we ALREADY have (no new WRDS pull):

  A. CALIBRATION backtest (is the demand model honest?)
     Leave-one-out: fit the demand Beta on all deals but one, compute the PIT = CDF of the
     realized demand under that fitted model. If the model is well-calibrated, PITs are
     Uniform(0,1): mean ~0.5, ~80% inside [0.1,0.9], KS-to-uniform not rejected. This directly
     answers "can we trust the distribution the Monte Carlo samples from?"

  B. REALIZED-EDGE event study (did the strategy's edge actually exist?)
     For each deal with terms + realized demand, push the REALIZED f_cash through the proration
     engine and measure optimal-election consideration minus the blended average — the actual
     historical proration-capture. Distribution across deals shows whether the alpha is real and
     how big. (Full cash P&L vs entry price needs the extended-window WRDS pull — scoped separately.)

NOTE ON SCOPE: this validates the demand model and the proration-capture alpha. A full
survivorship-aware trade P&L additionally requires (i) prices extended to each deal's close and
(ii) the terminated deals for deal-break risk — both flagged in the writeup, not silently omitted.
"""
from __future__ import annotations
import numpy as np
import pandas as pd
from scipy import stats
from arb_mc import fit_beta, prorate


def calibration_backtest(f_cash):
    f = np.asarray([v for v in f_cash if np.isfinite(v)], float)
    f = f[(f > 0) & (f < 1)]
    pit = np.empty(len(f))
    for i in range(len(f)):
        a, b = fit_beta(np.delete(f, i))
        pit[i] = stats.beta.cdf(f[i], a, b)
    inside80 = np.mean((pit >= 0.1) & (pit <= 0.9))
    ks = stats.kstest(pit, "uniform")
    return {"n": len(f), "pit_mean": float(pit.mean()), "pit_in_10_90": float(inside80),
            "ks_stat": float(ks.statistic), "ks_p": float(ks.pvalue), "pit": pit}


def realized_edge(deals):
    d = deals.dropna(subset=["C", "R", "P_acq", "f_cash", "pi_cash"]).copy()
    d = d[d.ratio_type == "fixed"]
    rows = []
    for _, r in d.iterrows():
        _, _, blended, optimal = prorate(r["f_cash"], r["pi_cash"], r["C"], r["R"] * r["P_acq"])
        rows.append({"event_id": r["event_id"], "target_name": r["target_name"],
                     "blended": float(blended), "optimal": float(optimal),
                     "edge": float(optimal - blended),
                     "edge_pct": float((optimal - blended) / blended * 100) if blended else np.nan})
    out = pd.DataFrame(rows)
    return out


if __name__ == "__main__":
    d = pd.read_csv("arb_deals.csv")
    cal = calibration_backtest(d["f_cash"].values)
    print("=== A. calibration backtest (demand model) ===")
    print(f"  n={cal['n']}  PIT mean={cal['pit_mean']:.3f} (ideal 0.50)  "
          f"in[0.1,0.9]={cal['pit_in_10_90']:.2f} (ideal 0.80)  KS p={cal['ks_p']:.3f} "
          f"({'calibrated' if cal['ks_p'] > 0.05 else 'miscalibrated'})")
    re = realized_edge(d)
    print("\n=== B. realized-edge event study (proration capture) ===")
    print(f"  deals: {len(re)}   mean edge={re.edge.mean():.3f} ({re.edge_pct.mean():.2f}% of blended)  "
          f"median={re.edge.median():.3f}  positive: {(re.edge > 1e-6).mean()*100:.0f}%")
    re.to_csv("arb_realized_edge.csv", index=False)
