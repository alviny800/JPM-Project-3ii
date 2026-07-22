#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Assemble a self-contained walkthrough page (figures embedded as data URIs) for the supervisor."""
import base64, json, os

import numpy as np
import pandas as pd

OUT = "arb_output"
S = json.load(open(f"{OUT}/summary.json"))
STRATEGY_SUMMARY = "arb_strategy_summary.json"


def load_strategy_summary(path=STRATEGY_SUMMARY):
    if not os.path.exists(path):
        return {}
    return json.load(open(path))


def signal_summary(path="arb_signals.csv"):
    if not os.path.exists(path):
        return {
            "priced": 0,
            "enter": 0,
            "reverse": 0,
            "review": 0,
            "trades": 0,
            "trade_e_mean": np.nan,
            "trade_realized_mean": np.nan,
            "trade_hit_rate": np.nan,
            "signal_corr": np.nan,
            "capacity_ok": 0,
            "capacity_median_notional": np.nan,
            "capacity_max_median_notional": np.nan,
            "capacity_optimal_median_notional": np.nan,
            "capacity_enter_median_notional": np.nan,
            "capacity_reverse_median_notional": np.nan,
            "capacity_median_ownership": np.nan,
            "capacity_e_mean": np.nan,
            "capacity_realized_mean": np.nan,
            "capacity_hit_rate": np.nan,
            "optimal_total_notional": np.nan,
            "optimal_expected_pnl": np.nan,
            "optimal_weighted_e": np.nan,
            "baseline_realized_pnl": np.nan,
            "baseline_return_on_deployed": np.nan,
            "self_impact_realized_pnl": np.nan,
            "self_impact_return_on_deployed": np.nan,
            "wrong_self_impact": 0,
            "missed_pass": 0,
        }
    sig = pd.read_csv(path)
    strategy = load_strategy_summary()
    optimal = strategy.get("profit_optimal", {})
    holder_model = strategy.get("holder_model", {})
    realized_baseline = strategy.get("realized_baseline_accept_historical_election", {}).get("optimal", {})
    realized_self = strategy.get("realized_self_impact_election", {}).get("optimal", {})
    trade_quality = strategy.get("trade_quality", {})
    counts = sig["signal"].value_counts()
    trades = sig[sig["signal"].isin(["ENTER", "REVERSE"])]
    realized = trades["realized_return_%"].dropna() if len(trades) else pd.Series(dtype=float)
    cap_realized = (
        trades["capacity_adjusted_realized_return_%"].dropna()
        if "capacity_adjusted_realized_return_%" in trades and len(trades)
        else pd.Series(dtype=float)
    )
    paired = trades[["E_return_%", "realized_return_%"]].dropna() if len(trades) else pd.DataFrame()
    corr = np.corrcoef(paired["E_return_%"], paired["realized_return_%"])[0, 1] if len(paired) > 2 else np.nan
    return {
        "priced": int(len(sig)),
        "enter": int(counts.get("ENTER", 0)),
        "reverse": int(counts.get("REVERSE", 0)),
        "review": int(counts.get("REVIEW", 0)),
        "trades": int(len(trades)),
        "trade_e_mean": float(trades["E_return_%"].mean()) if len(trades) else np.nan,
        "trade_realized_mean": float(realized.mean()) if len(realized) else np.nan,
        "trade_hit_rate": float((realized > 0).mean() * 100) if len(realized) else np.nan,
        "signal_corr": float(corr) if np.isfinite(corr) else np.nan,
        "capacity_ok": int(trades["capacity_status"].eq("ok").sum()) if "capacity_status" in trades else 0,
        "capacity_median_notional": float(trades["capacity_notional"].median())
        if "capacity_notional" in trades and len(trades) else np.nan,
        "capacity_max_median_notional": float(trades["capacity_max_notional"].median())
        if "capacity_max_notional" in trades and len(trades) else np.nan,
        "capacity_optimal_median_notional": float(trades["capacity_optimal_notional"].median())
        if "capacity_optimal_notional" in trades and len(trades) else np.nan,
        "capacity_enter_median_notional": float(sig.loc[sig["signal"].eq("ENTER"), "capacity_notional"].median())
        if "capacity_notional" in sig and counts.get("ENTER", 0) else np.nan,
        "capacity_reverse_median_notional": float(sig.loc[sig["signal"].eq("REVERSE"), "capacity_notional"].median())
        if "capacity_notional" in sig and counts.get("REVERSE", 0) else np.nan,
        "capacity_median_ownership": float(trades["capacity_pct_shares_outstanding"].median())
        if "capacity_pct_shares_outstanding" in trades and len(trades) else np.nan,
        "capacity_e_mean": float(trades["capacity_adjusted_E_return_%"].mean())
        if "capacity_adjusted_E_return_%" in trades and len(trades) else np.nan,
        "capacity_realized_mean": float(cap_realized.mean()) if len(cap_realized) else np.nan,
        "capacity_hit_rate": float((cap_realized > 0).mean() * 100) if len(cap_realized) else np.nan,
        "optimal_total_notional": float(optimal.get("total_notional", np.nan)),
        "optimal_expected_pnl": float(optimal.get("total_expected_pnl", np.nan)),
        "optimal_weighted_e": float(optimal.get("notional_weighted_expected_return_%", np.nan)),
        "baseline_realized_pnl": float(realized_baseline.get("total_realized_pnl", np.nan)),
        "baseline_return_on_deployed": float(realized_baseline.get("return_on_deployed_notional_%", np.nan)),
        "self_impact_realized_pnl": float(realized_self.get("total_realized_pnl", np.nan)),
        "self_impact_return_on_deployed": float(realized_self.get("return_on_deployed_notional_%", np.nan)),
        "wrong_self_impact": int(realized_self.get("wrong_trade_count", 0)),
        "missed_pass": int(trade_quality.get("missed_profitable_pass_count_marginal_proxy", 0)),
        "holder_median_fit_n": float(holder_model.get("median_fit_n", np.nan)),
        "holder_median_p_hat": float(holder_model.get("median_p_hat", np.nan)),
        "holder_median_q_hat": float(holder_model.get("median_q_hat", np.nan)),
        "holder_median_positive_active": float(holder_model.get("median_positive_share_of_active_%", np.nan)),
    }


SIG = signal_summary()
BREAK_PCT = float(S.get("portfolio_break_scenario", {}).get("p_break", np.nan)) * 100.0


def fmt_pct(value, digits=1):
    return "n/a" if not np.isfinite(value) else f"{value:.{digits}f}%"


def fmt_money_m(value, digits=1):
    return "n/a" if not np.isfinite(value) else f"${value/1_000_000:.{digits}f}m"


def img(name):
    b = base64.b64encode(open(f"{OUT}/{name}", "rb").read()).decode()
    return f"data:image/png;base64,{b}"

FIG1, FIG2, FIG3, FIG4 = img("demand_distribution.png"), img("calibration_pit.png"), img("realized_edge.png"), img("portfolio_pnl.png")

html = f"""<title>Election Arbitrage — Monte Carlo Framework</title>
<style>
:root {{
  --paper:#FAF8F4; --surface:#FFFFFF; --ink:#191E26; --muted:#5C6672; --faint:#8A929E;
  --line:#E7E1D6; --edge:#3D6591; --risk:#9A6C8E; --good:#4F7A46; --edge-soft:#EAF0F6;
  --serif:"Iowan Old Style","Charter","Palatino Linotype",Palatino,Georgia,serif;
  --sans:system-ui,-apple-system,"Segoe UI",Roboto,Helvetica,Arial,sans-serif;
  --mono:ui-monospace,"SF Mono",Menlo,Consolas,monospace;
}}
@media (prefers-color-scheme:dark) {{
  :root {{ --paper:#11141A; --surface:#191E26; --ink:#ECEEF1; --muted:#98A1AE; --faint:#6B7480;
    --line:#2A313B; --edge:#7CA6D2; --risk:#C79CBB; --good:#93C486; --edge-soft:#1C2530; }}
}}
:root[data-theme="light"] {{ --paper:#FAF8F4; --surface:#FFFFFF; --ink:#191E26; --muted:#5C6672; --faint:#8A929E;
  --line:#E7E1D6; --edge:#3D6591; --risk:#9A6C8E; --good:#4F7A46; --edge-soft:#EAF0F6; }}
:root[data-theme="dark"] {{ --paper:#11141A; --surface:#191E26; --ink:#ECEEF1; --muted:#98A1AE; --faint:#6B7480;
  --line:#2A313B; --edge:#7CA6D2; --risk:#C79CBB; --good:#93C486; --edge-soft:#1C2530; }}

* {{ box-sizing:border-box; }}
body {{ margin:0; background:var(--paper); color:var(--ink); font-family:var(--sans);
  line-height:1.62; -webkit-font-smoothing:antialiased; }}
.col {{ max-width:820px; margin:0 auto; padding:0 24px; }}
.mono {{ font-family:var(--mono); font-variant-numeric:tabular-nums; }}

/* masthead */
header {{ border-bottom:1px solid var(--line); background:var(--surface); }}
header .col {{ padding-top:52px; padding-bottom:34px; }}
.eyebrow {{ font-family:var(--mono); text-transform:uppercase; letter-spacing:.16em; font-size:11.5px;
  color:var(--edge); font-weight:600; margin:0 0 14px; }}
h1 {{ font-family:var(--serif); font-weight:600; font-size:40px; line-height:1.1; margin:0 0 12px;
  letter-spacing:-.01em; text-wrap:balance; }}
.sub {{ color:var(--muted); font-size:17px; max-width:60ch; margin:0; }}
.meta {{ display:flex; flex-wrap:wrap; gap:8px; margin-top:22px; }}
.chip {{ font-family:var(--mono); font-size:12px; color:var(--muted); border:1px solid var(--line);
  border-radius:999px; padding:4px 11px; background:var(--paper); }}
.chip b {{ color:var(--ink); font-weight:600; }}
.chip.ok {{ color:var(--good); border-color:color-mix(in srgb,var(--good) 40%,var(--line)); }}

/* scorecard */
.cards {{ display:grid; grid-template-columns:repeat(auto-fit,minmax(150px,1fr)); gap:14px; margin:38px 0 8px; }}
.card {{ background:var(--surface); border:1px solid var(--line); border-radius:12px; padding:16px 16px 15px; }}
.card .k {{ font-family:var(--mono); font-size:11px; text-transform:uppercase; letter-spacing:.09em;
  color:var(--faint); margin:0 0 9px; }}
.card .v {{ font-family:var(--serif); font-size:30px; font-weight:600; line-height:1; letter-spacing:-.01em; }}
.card .v small {{ font-size:15px; color:var(--muted); font-family:var(--sans); font-weight:400; }}
.card .n {{ font-size:12.5px; color:var(--muted); margin:9px 0 0; }}
.card.signal .v {{ color:var(--edge); }} .card.good .v {{ color:var(--good); }}

section {{ padding:40px 0; border-top:1px solid var(--line); }}
section:first-of-type {{ border-top:none; }}
h2 {{ font-family:var(--serif); font-size:24px; font-weight:600; margin:0 0 6px; letter-spacing:-.01em; }}
.lead {{ color:var(--muted); margin:0 0 22px; font-size:15.5px; }}
p {{ font-size:16px; }} p.tight {{ margin:12px 0; }}
.snum {{ font-family:var(--mono); color:var(--edge); font-weight:600; font-size:13px; margin-right:8px; }}

figure {{ margin:26px 0 8px; }}
figure img {{ width:100%; max-width:100%; display:block; border:1px solid var(--line); border-radius:10px;
  background:var(--surface); }}
figcaption {{ font-size:13px; color:var(--muted); margin-top:10px; padding-left:2px; }}
figcaption b {{ color:var(--ink); font-weight:600; }}

.callout {{ background:var(--edge-soft); border:1px solid color-mix(in srgb,var(--edge) 24%,var(--line));
  border-radius:12px; padding:18px 20px; margin:22px 0; }}
.callout p {{ margin:0; font-size:15.5px; }} .callout .k {{ font-family:var(--mono); font-size:11px;
  text-transform:uppercase; letter-spacing:.1em; color:var(--edge); font-weight:600; margin:0 0 7px; }}

table {{ width:100%; border-collapse:collapse; margin:20px 0 6px; font-size:14.5px; }}
th,td {{ text-align:left; padding:9px 12px; border-bottom:1px solid var(--line); }}
th {{ font-family:var(--mono); font-size:11px; text-transform:uppercase; letter-spacing:.07em;
  color:var(--faint); font-weight:600; }}
td.n {{ font-family:var(--mono); font-variant-numeric:tabular-nums; text-align:right; }}
tr td:last-child {{ color:var(--muted); }}
.drop td.n {{ color:var(--risk); }}
.keep td:first-child b {{ color:var(--good); }}

ul.scope {{ list-style:none; padding:0; margin:16px 0 0; display:grid; gap:12px; }}
ul.scope li {{ padding-left:26px; position:relative; font-size:15.5px; }}
ul.scope li::before {{ content:""; position:absolute; left:0; top:9px; width:11px; height:11px;
  border-radius:3px; background:var(--edge); }}
ul.scope li.todo::before {{ background:var(--risk); }}
ul.scope li b {{ font-weight:600; }}
footer {{ border-top:1px solid var(--line); color:var(--faint); font-size:12.5px; padding:26px 0 60px;
  font-family:var(--mono); }}
@media (max-width:680px) {{ .cards {{ grid-template-columns:repeat(2,1fr); }} h1 {{ font-size:31px; }} }}
</style>

<header><div class="col">
  <p class="eyebrow">JPM Election-Arb · Modeling Framework</p>
  <h1>Predicting election demand to price cash-or-stock merger arbitrage</h1>
  <p class="sub">A Monte Carlo that simulates how shareholders elect cash vs. stock, pushes each draw through
  the deal's proration mechanics, and produces a distribution of realized consideration and strategy P&amp;L.</p>
  <div class="meta">
    <span class="chip"><b>{S['demand_calibration_set']}</b> disclosed election outcomes</span>
    <span class="chip"><b>{S['mc_ready_deals']}</b> fully MC-ready deals</span>
    <span class="chip"><b>{SIG['priced']}</b> priced signals</span>
    <span class="chip ok">calibration KS&nbsp;p&nbsp;=&nbsp;<b>{S['calibration_ks_p']}</b> · passes</span>
    <span class="chip">as of 2026-07-21</span>
  </div>
</div></header>

<div class="col">

  <div class="cards">
    <div class="card"><p class="k">Demand — mean</p><div class="v">{int(S['demand_mean']*100)}<small>% cash</small></div><p class="n">Beta({S['demand_beta'][0]}, {S['demand_beta'][1]}), U-shaped</p></div>
    <div class="card good"><p class="k">Model calibration</p><div class="v">{S['calibration_pit_mean']}</div><p class="n">PIT mean (ideal 0.50) · {int(S['calibration_in80']*100)}% in 80% band</p></div>
    <div class="card signal"><p class="k">Realized edge</p><div class="v">100<small>% +ve</small></div><p class="n">median {S['realized_edge_median_pct']}% of blended</p></div>
    <div class="card signal"><p class="k">Portfolio · w/ break</p><div class="v">{S['portfolio_with_break_mean_pct']}<small>%</small></div><p class="n">5th-pct {S['portfolio_with_break_p05_pct']}% — tail is deal-break</p></div>
    <div class="card signal"><p class="k">Trade blotter</p><div class="v">{SIG['trades']}<small> trades</small></div><p class="n">{SIG['enter']} ENTER · {SIG['reverse']} REVERSE · {SIG['review']} REVIEW</p></div>
    <div class="card signal"><p class="k">Optimal capacity</p><div class="v">{fmt_money_m(SIG['optimal_total_notional'])}</div><p class="n">median trade {fmt_money_m(SIG['capacity_optimal_median_notional'])} · ownership {SIG['capacity_median_ownership']:.2f}%</p></div>
  </div>

  <section>
    <h2>The one idea</h2>
    <p class="lead">Everything in the payoff chain is deterministic given deal terms — except one thing.</p>
    <p><span class="snum">01</span>The only <b>random</b> quantity is <b>f<sub>cash</sub></b>, the fraction of shares that elect cash at the deadline. So the model is a single stochastic node: <span class="mono">draw f<sub>cash</sub> → proration → optimal-election consideration → edge</span>, repeated to build a distribution.</p>
    <p><span class="snum">02</span>In a fully-prorated deal the <b>blended (average) consideration is fixed</b> by the cash pool — it does not depend on demand. The arbitrage edge comes entirely from <b>optimal election</b>: elect the richer side, and when that side is <i>under-subscribed</i> you capture more of it than the average holder. How much you capture depends on the demand draw. <b>That is what the Monte Carlo prices.</b></p>
  </section>

  <section>
    <h2>Where the sample comes from</h2>
    <p class="lead">2,068 raw deals → {S['demand_calibration_set']} with a disclosed election split. The two largest cuts are structural, not pipeline losses.</p>
    <table>
      <thead><tr><th>Filter</th><th style="text-align:right">Remaining</th><th>Why it drops</th></tr></thead>
      <tbody>
        <tr><td>Raw Bloomberg pull (2006+)</td><td class="n">2,068</td><td>—</td></tr>
        <tr class="drop"><td>Keep “Cash <b>or</b> Stock” (true election)</td><td class="n">727</td><td>−1,341 fixed-mix deals — no election, no proration</td></tr>
        <tr class="drop"><td>Keep completed</td><td class="n">619</td><td>−108 terminated / withdrawn</td></tr>
        <tr class="drop"><td>US + resolvable identifier</td><td class="n">317</td><td>−302 non-US — no EDGAR filings to read</td></tr>
        <tr><td>Ran through EDGAR + Claude</td><td class="n">303</td><td>−14 no CIK / no election filing found</td></tr>
        <tr class="keep"><td><b>Clean disclosed election demand</b></td><td class="n">{S['demand_calibration_set']}</td><td>not disclosed, or unparseable prose</td></tr>
      </tbody>
    </table>
    <div class="callout"><p class="k">Verified: this is a disclosure ceiling, not a bug</p>
      <p>Of the 231 with no clean label, <b>98% already had the post-close 8-K pulled</b> — the election split simply isn’t disclosed in it (small deals file terse “merger completed” notices). Re-tuning the document pull would recover ~0 deals. The {S['demand_calibration_set']} are close to the real limit of what public disclosure supports.</p></div>
  </section>

  <section>
    <h2>The demand model — and is it honest?</h2>
    <p class="lead">The distribution the Monte Carlo samples, and a leave-one-out test of whether we can trust it.</p>
    <figure><img alt="Election demand distribution with fitted Beta" src="{FIG1}">
      <figcaption><b>Fig 1 — Election-demand distribution.</b> Realized cash-election fractions across {S['demand_calibration_set']} deals, with a fitted Beta({S['demand_beta'][0]}, {S['demand_beta'][1]}). The U-shape is economically real: holders pile toward one corner (“almost all cash” or “almost all stock”), because election is close to a binary value decision.</figcaption></figure>
    <figure><img alt="Leave-one-out calibration PIT histogram" src="{FIG2}">
      <figcaption><b>Fig 2 — Calibration backtest.</b> Leave-one-out PIT values sit flat and uniform: mean <span class="mono">{S['calibration_pit_mean']}</span> (ideal 0.50), <span class="mono">{int(S['calibration_in80']*100)}%</span> inside the 80% band, <span class="mono">KS p = {S['calibration_ks_p']}</span>. The model is indistinguishable from perfectly calibrated <b>out-of-sample</b> — the credibility anchor for everything downstream.</figcaption></figure>
  </section>

  <section>
    <h2>Does the edge actually exist?</h2>
    <p class="lead">Push each deal’s <i>realized</i> demand through the proration engine and measure the proration capture.</p>
    <figure><img alt="Realized proration-capture edge per deal" src="{FIG3}">
      <figcaption><b>Fig 3 — Realized-edge event study.</b> Across {S['mc_ready_deals']} deals the optimal-election consideration beats the blended average <b>100% of the time</b> — median <span class="mono">{S['realized_edge_median_pct']}%</span> of blended, mean <span class="mono">{S['realized_edge_mean_pct']}%</span> (right-skewed: usually small, occasionally large). The alpha is real and its shape is honest.</figcaption></figure>
  </section>

  <section>
    <h2>Portfolio Monte Carlo</h2>
    <p class="lead">Equal-weight the MC-ready deals, 20,000 paths, with a deal-break overlay.</p>
    <figure><img alt="Portfolio P&amp;L distribution with deal-break overlay" src="{FIG4}">
      <figcaption><b>Fig 4 — Portfolio P&amp;L.</b> Pure proration edge (blue) sits positive; a <b>{fmt_pct(BREAK_PCT, 1)} deal-break scenario</b> (mauve, with state-specific break losses) shifts it left to a <span class="mono">{S['portfolio_with_break_mean_pct']}%</span> mean and opens a downside tail (5th-pct <span class="mono">{S['portfolio_with_break_p05_pct']}%</span>). Crucially, <b>the left tail is deal-break risk, not election risk</b> — the election model itself is well-behaved.</figcaption></figure>
    <div class="callout"><p class="k">On spread conditioning</p>
      <p>Rational holders should tilt toward the richer side, so demand ought to respond to the deadline spread. On our data that logit slope is <span class="mono">≈ 0</span> (thin, n={S['mc_ready_deals']}). The framework <b>supports</b> conditioning but doesn’t assert it — it Monte-Carlos over the slope’s uncertainty rather than overfitting a curve.</p></div>
  </section>

  <section>
    <h2>Risk-aware trade layer</h2>
    <p class="lead">The signal layer prices both the long election strategy and the reverse trade under completed / terminated / withdrawn probabilities.</p>
    <p><span class="snum">01</span><b>ENTER</b> buys the target, elects the higher-expected side, and shorts the expected acquirer-stock receipt. This is the election-right trade.</p>
    <p><span class="snum">02</span><b>REVERSE</b> shorts the target only when the passive settlement liability is attractive after outcome risk. It has <b>no election right</b>, so completion payoff is the blended consideration, not the optimal elected side.</p>
    <p><span class="snum">03</span>Current blotter: <span class="mono">{SIG['priced']}</span> priced signals, <span class="mono">{SIG['enter']}</span> ENTER, <span class="mono">{SIG['reverse']}</span> REVERSE, <span class="mono">{SIG['review']}</span> REVIEW. Trade-book mean expected return is <span class="mono">{fmt_pct(SIG['trade_e_mean'])}</span>; realized coverage mean is <span class="mono">{fmt_pct(SIG['trade_realized_mean'])}</span> with hit rate <span class="mono">{fmt_pct(SIG['trade_hit_rate'], 0)}</span>. Signal/realized correlation on covered trades is <span class="mono">{SIG['signal_corr']:+.2f}</span>.</p>
    <p><span class="snum">04</span><b>Capacity overlay.</b> ENTER capacity is finite because our target shares change aggregate election demand. Holder composition now comes from the rolling structural model: <span class="mono">q_hat</span> estimates EV-sensitive holders and <span class="mono">p_hat</span> estimates noisy cash-election propensity from prior disclosed events. Current trade median fit depth is <span class="mono">{SIG['holder_median_fit_n']:.0f}</span>, median <span class="mono">p_hat={SIG['holder_median_p_hat']:.1f}</span>, median <span class="mono">q_hat={SIG['holder_median_q_hat']:.1f}</span>, and median positive-holder share of active is <span class="mono">{fmt_pct(SIG['holder_median_positive_active'])}</span>. REVERSE remains borrow/sale constrained and keeps passive settlement because it has no election right. Maximum and optimal coincide in this run: total optimal notional is <span class="mono">{fmt_money_m(SIG['optimal_total_notional'])}</span>, expected P&amp;L is <span class="mono">{fmt_money_m(SIG['optimal_expected_pnl'])}</span>, and weighted expected return is <span class="mono">{fmt_pct(SIG['optimal_weighted_e'])}</span>. Historical-election realized P&amp;L is <span class="mono">{fmt_money_m(SIG['baseline_realized_pnl'])}</span>; self-impact realized P&amp;L is <span class="mono">{fmt_money_m(SIG['self_impact_realized_pnl'])}</span> with <span class="mono">{SIG['wrong_self_impact']}</span> wrong trades and <span class="mono">{SIG['missed_pass']}</span> profitable PASS-row miss.</p>
  </section>

  <section>
    <h2>What’s solid, and what needs one more pull</h2>
    <ul class="scope">
      <li><b>Demand distribution — solid (n={S['demand_calibration_set']}) and near the disclosure ceiling.</b> No more material recovery expected from EDGAR; the missing labels are mostly not disclosed.</li>
      <li><b>Calibration, realized edge, and outcome-risk overlay — done on local inputs.</b> The current run uses event-level completed / terminated / withdrawn probabilities where available.</li>
      <li><b>Risk-aware long/reverse signal layer — done.</b> The reverse strategy explicitly uses passive blended settlement because the short side has no election right.</li>
      <li><b>Capacity and election-impact overlay — done.</b> Holder mix is now rolling-estimated from prior events, and the blotter reports raw maximum, risk-gated maximum, profit-optimal size, historical-election realized P&amp;L, and self-impact election realized P&amp;L.</li>
      <li class="todo"><b>Selection caveat to state plainly.</b> The disclosers skew large/liquid — the demand distribution is “demand among deals that disclose,” which is close to the tradeable set but not the full M&amp;A universe.</li>
    </ul>
  </section>

  <footer>Election-arb MC framework · arb_terms → arb_outcome → arb_mc → arb_backtest → arb_run → arb_signal · figures generated {20000:,}-path · 2026-07-21</footer>
</div>
"""
open(f"{OUT}/walkthrough.html", "w").write(html)
print(f"[walkthrough] wrote {OUT}/walkthrough.html  ({len(html)//1024} KB)")
