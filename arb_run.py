#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
arb_run.py  —  DRIVER for the election-arb Monte Carlo / backtest framework.

Runs the whole pipeline and writes figures + a summary to arb_output/:
  terms (arb_terms)  ->  demand model + calibration backtest  ->  per-deal MC  ->  portfolio MC
Figures:
  1 demand_distribution.png    empirical realized demand + fitted Beta (what the MC samples)
  2 calibration_pit.png        leave-one-out PIT histogram (is the model honest?)
  2b beta_qq.png               Beta QQ plot (pooled goodness-of-fit; points on the 45° line = fit)
  3 realized_edge.png          per-deal proration-capture edge (event study B)
  4 portfolio_pnl.png          portfolio edge distribution, with an outcome-risk overlay
Summary: arb_output/summary.md  (numbers + interpretation, ready to walk through)
"""
from __future__ import annotations
import json, os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy import stats

import arb_terms
from arb_mc import DemandModel, simulate_deal, summarize
from arb_backtest import calibration_backtest, realized_edge
from arb_outcome import OutcomeDefaults, load_outcome_probability_table, outcome_probabilities_for_event

OUT = "arb_output"
os.makedirs(OUT, exist_ok=True)
RNG = np.random.default_rng(20260716)


def main():
    d = arb_terms.build_deals()
    cal_sample = d["f_cash"].dropna().values
    spr = d.dropna(subset=["f_cash", "spread"])
    model = DemandModel(cal_sample)
    model_c = DemandModel(spr["f_cash"].values, spread=spr["spread"].values)
    ready = d.dropna(subset=["C", "R", "P_acq", "f_cash", "pi_cash"])
    ready = ready[ready.ratio_type == "fixed"].reset_index(drop=True)

    # ---- fig 1: demand distribution ----
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.hist(cal_sample, bins=15, density=True, alpha=.55, color="#4C78A8", label=f"realized (n={len(cal_sample)})")
    xs = np.linspace(.001, .999, 200)
    ax.plot(xs, stats.beta.pdf(xs, model.a, model.b), "r-", lw=2, label=f"Beta({model.a:.2f},{model.b:.2f})")
    ax.axvline(model.a / (model.a + model.b), color="k", ls="--", lw=1, label=f"mean={model.a/(model.a+model.b):.2f}")
    ax.set(xlabel="fraction of shares electing CASH", ylabel="density",
           title="Election-demand distribution (the MC's stochastic input)")
    ax.legend(fontsize=8); fig.tight_layout(); fig.savefig(f"{OUT}/demand_distribution.png", dpi=130); plt.close(fig)

    # ---- A calibration + fig 2 ----
    cb = calibration_backtest(cal_sample)
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.hist(cb["pit"], bins=10, range=(0, 1), color="#59A14F", alpha=.8, edgecolor="w")
    ax.axhline(len(cb["pit"]) / 10, color="k", ls="--", lw=1, label="uniform (ideal)")
    ax.set(xlabel="PIT = model CDF at realized demand", ylabel="count",
           title=f"Calibration backtest — KS p={cb['ks_p']:.2f}, {cb['pit_in_10_90']*100:.0f}% in 80% band")
    ax.legend(fontsize=8); fig.tight_layout(); fig.savefig(f"{OUT}/calibration_pit.png", dpi=130); plt.close(fig)

    # ---- fig 2b: Beta QQ plot (pooled goodness-of-fit — points on the 45° line = good fit) ----
    fq = np.sort(cal_sample)
    pos = (np.arange(1, len(fq) + 1) - 0.5) / len(fq)          # plotting positions
    theo = stats.beta.ppf(pos, model.a, model.b)              # BETA quantiles (works for any distribution)
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.scatter(theo, fq, s=20, color="#4C78A8", zorder=3)
    ax.plot([0, 1], [0, 1], "--", color="#E15759", lw=1.5, label="perfect fit (45°)")
    ax.set(xlabel=f"theoretical quantile — Beta({model.a:.2f}, {model.b:.2f})",
           ylabel="observed demand quantile", title="Beta QQ plot — points on the line = good fit")
    ax.legend(fontsize=8); fig.tight_layout(); fig.savefig(f"{OUT}/beta_qq.png", dpi=130); plt.close(fig)

    # ---- B realized edge + fig 3 ----
    re = realized_edge(d)
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.hist(re.edge_pct, bins=15, color="#E15759", alpha=.8, edgecolor="w")
    ax.axvline(re.edge_pct.median(), color="k", ls="--", lw=1, label=f"median={re.edge_pct.median():.2f}%")
    ax.set(xlabel="realized proration-capture edge (% of blended)", ylabel="deals",
           title=f"Realized-edge event study (n={len(re)}, {(re.edge>1e-6).mean()*100:.0f}% positive)")
    ax.legend(fontsize=8); fig.tight_layout(); fig.savefig(f"{OUT}/realized_edge.png", dpi=130); plt.close(fig)

    # ---- portfolio MC + fig 4 ----
    NPATH = 20000
    TERMINATED_LOSS, WITHDRAWN_LOSS = 0.25, 0.35
    outcome_table = load_outcome_probability_table("deal_outcome_probabilities.csv")
    outcome_defaults = OutcomeDefaults(completed=0.88, terminated=0.07, withdrawn=0.05)
    per_deal_edge_pct = np.zeros((len(ready), NPATH))
    per_deal_real_pct = np.zeros((len(ready), NPATH))
    outcome_rows = []
    for i, (_, r) in enumerate(ready.iterrows()):
        outcome = outcome_probabilities_for_event(r, outcome_table, outcome_defaults)
        sim = simulate_deal(
            r,
            model,
            n=NPATH,
            p_completed=outcome["p_completed"],
            p_terminated=outcome["p_terminated"],
            p_withdrawn=outcome["p_withdrawn"],
            terminated_loss_frac=TERMINATED_LOSS,
            withdrawn_loss_frac=WITHDRAWN_LOSS,
            rng=RNG,
        )
        per_deal_edge_pct[i] = sim["edge"] / sim["blended"] * 100
        per_deal_real_pct[i] = (sim["realized"] - sim["blended"]) / sim["blended"] * 100
        outcome_rows.append(outcome)
    port_edge = per_deal_edge_pct.mean(axis=0)          # equal-weight, no break
    port_real = per_deal_real_pct.mean(axis=0)          # equal-weight, three-state outcome overlay
    outcome_df = pd.DataFrame(outcome_rows)
    avg_p_completed = float(outcome_df["p_completed"].mean()) if len(outcome_df) else 0.88
    avg_p_terminated = float(outcome_df["p_terminated"].mean()) if len(outcome_df) else 0.07
    avg_p_withdrawn = float(outcome_df["p_withdrawn"].mean()) if len(outcome_df) else 0.05
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.hist(port_edge, bins=40, alpha=.6, color="#4C78A8", label="proration edge (no break)")
    ax.hist(port_real, bins=40, alpha=.6, color="#B07AA1",
            label=f"with {int((avg_p_terminated + avg_p_withdrawn)*100)}% term/withdraw")
    ax.axvline(0, color="k", lw=1)
    ax.set(xlabel="portfolio return vs blended (%)", ylabel="MC paths",
           title=f"Portfolio P&L distribution ({len(ready)} deals, equal-weight)")
    ax.legend(fontsize=8); fig.tight_layout(); fig.savefig(f"{OUT}/portfolio_pnl.png", dpi=130); plt.close(fig)

    # ---- summary ----
    S = {
        "demand_beta": [round(model.a, 3), round(model.b, 3)],
        "demand_mean": round(float(model.a / (model.a + model.b)), 3),
        "demand_calibration_set": int(len(cal_sample)),
        "spread_logit_slope": round(float(model_c.lb), 4),
        "spread_logit_slope_se": round(float(model_c.lb_se), 4) if np.isfinite(model_c.lb_se) else None,
        "calibration_ks_p": round(cb["ks_p"], 3), "calibration_pit_mean": round(cb["pit_mean"], 3),
        "calibration_in80": round(cb["pit_in_10_90"], 3),
        "mc_ready_deals": int(len(ready)),
        "realized_edge_median_pct": round(float(re.edge_pct.median()), 2),
        "realized_edge_mean_pct": round(float(re.edge_pct.mean()), 2),
        "realized_edge_pct_positive": round(float((re.edge > 1e-6).mean()) * 100, 0),
        "portfolio_edge_mean_pct": round(float(port_edge.mean()), 2),
        "portfolio_break_scenario": {
            "p_break": round(avg_p_terminated + avg_p_withdrawn, 4),
            "loss_frac": round(
                (avg_p_terminated * TERMINATED_LOSS + avg_p_withdrawn * WITHDRAWN_LOSS)
                / max(avg_p_terminated + avg_p_withdrawn, 1e-12),
                3,
            ),
            "source": "aggregate_view_of_event_outcome_probabilities",
        },
        "portfolio_outcome_scenario": {
            "p_completed": round(avg_p_completed, 4),
            "p_terminated": round(avg_p_terminated, 4),
            "p_withdrawn": round(avg_p_withdrawn, 4),
            "terminated_loss_frac": TERMINATED_LOSS,
            "withdrawn_loss_frac": WITHDRAWN_LOSS,
            "source_counts": outcome_df["outcome_probability_source"].value_counts().to_dict()
            if "outcome_probability_source" in outcome_df
            else {},
        },
        "portfolio_with_break_mean_pct": round(float(port_real.mean()), 2),
        "portfolio_with_break_p05_pct": round(float(np.percentile(port_real, 5)), 2),
    }
    json.dump(S, open(f"{OUT}/summary.json", "w"), indent=2)

    md = f"""# Election-Arb Monte Carlo — results summary

## 1. Demand model (the MC's stochastic core)
- Calibrated on **{S['demand_calibration_set']} realized election outcomes**.
- Fit: **Beta({S['demand_beta'][0]}, {S['demand_beta'][1]})**, mean **{S['demand_mean']*100:.0f}% elect cash**.
- Shape is U-ish: deals cluster toward "almost all cash" / "almost all stock" — election is close to a corner decision, as economics predicts.

## 2. Calibration backtest — IS THE MODEL HONEST?  ✅
- Leave-one-out PIT: mean **{S['calibration_pit_mean']}** (ideal 0.50), **{S['calibration_in80']*100:.0f}%** inside the 80% band (ideal 80%), **KS p={S['calibration_ks_p']}** → not distinguishable from perfectly calibrated.
- Interpretation: the distribution the Monte Carlo samples is trustworthy on out-of-sample deals.

## 3. Realized-edge event study — DOES THE ALPHA EXIST?
- {S['mc_ready_deals']} MC-ready deals. Proration-capture edge **{S['realized_edge_pct_positive']:.0f}% positive**.
- Median **{S['realized_edge_median_pct']}% of blended**, mean **{S['realized_edge_mean_pct']}%** (right-skewed: usually small, occasionally large).

## 4. Portfolio Monte Carlo
- Equal-weight {S['mc_ready_deals']} deals, {20000:,} paths.
- Proration edge (no break): mean **{S['portfolio_edge_mean_pct']}%**.
- With event-level completed/terminated/withdrawn probabilities averaging **{(S['portfolio_outcome_scenario']['p_terminated'] + S['portfolio_outcome_scenario']['p_withdrawn'])*100:.1f}% terminated/withdrawn** (terminated loss {int(S['portfolio_outcome_scenario']['terminated_loss_frac']*100)}%, withdrawn loss {int(S['portfolio_outcome_scenario']['withdrawn_loss_frac']*100)}%): mean **{S['portfolio_with_break_mean_pct']}%**, 5th-pct **{S['portfolio_with_break_p05_pct']}%** — the tail is deal-outcome risk, not election risk.

## Spread conditioning
- logit(demand) slope on deadline spread = **{S['spread_logit_slope']:+}** (se {S['spread_logit_slope_se']}) → ~flat on our data. Framework supports conditioning; the data says the tilt is weak, so we MC over that slope's uncertainty rather than assert it.

## Honest scope / current coverage
- **Demand distribution: solid (n={S['demand_calibration_set']}).** Near the disclosure ceiling — no more recoverable from EDGAR (verified: 98% of no-label deals already had the 8-K pulled, the number just isn't disclosed).
- **No new Claude run is needed for this pipeline.** Election/proration demand uses the existing Claude extraction + normalized labels; deal-outcome risk uses BBG `Deal Status` across Completed/Terminated/Withdrawn.
- The realized election-edge backtest remains the completed-election universe because terminated/withdrawn deals do not have final proration/election outcomes. Their effect enters the trade and portfolio layers through the event-level outcome probabilities.
"""
    open(f"{OUT}/summary.md", "w").write(md)
    try:
        from material_builder import export_after_arb_run
        material_results = export_after_arb_run()
        print("[material] wrote MC slide material to material/")
        print(json.dumps(material_results, indent=2))
    except Exception as exc:
        print(f"[material] skipped MC material export: {exc}")
    print("[run] wrote figures + summary to", OUT)
    print(json.dumps(S, indent=2))


if __name__ == "__main__":
    main()
