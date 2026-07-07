#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
election_arb_eda.py

Week-3 EDA + statistical tests on election/proration deals.

Goal: understand the empirical drivers of realized cash-vs-stock election
demand, so we can build a structural model of `p_active(spread)` — the
probability an active investor elects cash as a function of the deadline-date
spread between cash-election value and stock-election value.

V1 modeling assumptions
-----------------------
- Active investors treated as ONE population (no sub-categorization by hedge
  fund vs. mutual fund vs. retail). One distribution to learn.
- Lent shares are treated as PASSIVE. We use `passive_control_percent` from
  the WRDS ownership pipeline, which (per the colleague's spec) folds in
  N-PORT on-loan balances on the passive side.
- Default-rule lazy money goes the way of the default. So:
      passive_cash_demand_share = passive% if default_is_cash else 0
- Active cash-election rate is then backed out:
      active_cash_election_rate =
          (realized_cash_share - passive_cash_demand_share) / (1 - passive%)

Where the script sits in the pipeline
-------------------------------------
BBG M&A CSV
  → download_ma_edgar_files.py + Claude  → llm_field_extractions.csv
  → download_ownership_etf_data.py       → ownership_mix_by_event.csv
  → download_wrds_market_data.py         → event_market_features.csv
  → election_arb_eda.py  (this script)   → eda_output/

Inputs
------
- --extractions  llm_field_extractions.csv  (long form: event_id × field_name)
- --ownership    ownership_mix_by_event.csv (one row per event_id)
- --market       event_market_features.csv  (one row per event_id, or per
                                              event_id + date if intraday)

Outputs
-------
- <output-dir>/merged_panel.csv         — joined deal-level panel
- <output-dir>/plots/*.png              — histograms, scatters, heatmaps
- <output-dir>/tables/*.csv             — OLS coefficients, K-S tests,
                                          fitted p_active(spread)
- <output-dir>/eda_summary.md           — narrative for the writeup

Requires: pandas, numpy, matplotlib, scipy
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Dict, Optional

import matplotlib
matplotlib.use("Agg")  # headless — write PNGs without a display
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats


# ---------------------------------------------------------------------------
# Load + normalize the three pipeline outputs
# ---------------------------------------------------------------------------

CLAUDE_FIELDS_WIDE = [
    "consideration_menu",
    "cash_consideration_per_share",
    "stock_consideration_per_share",
    "exchange_ratio",
    "cash_cap",
    "stock_cap",
    "proration_formula",
    "non_election_default_rule",
    "election_deadline",
    "record_date",
    "preliminary_proration_results",
    "final_proration_results",
    "realized_cash_election_demand",
    "realized_stock_election_demand",
    "deal_completion_or_break",
]


def pivot_claude_long_to_wide(df_long: pd.DataFrame) -> pd.DataFrame:
    """Claude output is one row per (event_id, field_name). Pivot to wide form."""
    if "field_name" not in df_long.columns or "value" not in df_long.columns:
        raise ValueError(
            "Expected long-form Claude output with `field_name` + `value` columns. "
            f"Got: {list(df_long.columns)}"
        )
    keep = df_long[df_long["field_name"].isin(CLAUDE_FIELDS_WIDE)].copy()
    index_cols = [c for c in ["event_id", "target_name", "acquirer_name"] if c in keep.columns]
    wide = keep.pivot_table(
        index=index_cols,
        columns="field_name",
        values="value",
        aggfunc="first",
    ).reset_index()
    wide.columns.name = None
    return wide


def coerce_numeric(s: pd.Series) -> pd.Series:
    """Pull the first number out of mostly-text fields. Claude often returns
    strings like '$50.00' or '~65% elected cash'; we want the number."""
    if s.dtype.kind in "iuf":
        return s
    extracted = (
        s.astype(str)
        .str.replace(",", "", regex=False)
        .str.extract(r"(-?\d+\.?\d*)", expand=False)
    )
    return pd.to_numeric(extracted, errors="coerce")


def normalize_default_rule(s: pd.Series) -> pd.Series:
    """Returns 'cash', 'stock', 'mixed', or NaN."""
    t = s.astype(str).str.lower()
    out = pd.Series(np.nan, index=s.index, dtype=object)
    out[t.str.contains("cash", na=False) & ~t.str.contains("stock", na=False)] = "cash"
    out[t.str.contains("stock", na=False) & ~t.str.contains("cash", na=False)] = "stock"
    out[t.str.contains("mix", na=False) | t.str.contains("blend", na=False)] = "mixed"
    return out


def load_and_merge(
    extractions_path: Path,
    ownership_path: Path,
    market_path: Path,
) -> pd.DataFrame:
    """Load the three CSVs and merge into one deal-level DataFrame keyed by event_id."""
    print(f"[load] extractions: {extractions_path}", file=sys.stderr)
    ext_long = pd.read_csv(extractions_path)
    ext = pivot_claude_long_to_wide(ext_long)

    # Coerce numeric/categorical extracted fields
    for col in [
        "cash_consideration_per_share",
        "stock_consideration_per_share",
        "exchange_ratio",
        "cash_cap",
        "stock_cap",
        "realized_cash_election_demand",
        "realized_stock_election_demand",
        "final_proration_results",
        "preliminary_proration_results",
    ]:
        if col in ext.columns:
            ext[col + "_num"] = coerce_numeric(ext[col])

    if "non_election_default_rule" in ext.columns:
        ext["default_rule"] = normalize_default_rule(ext["non_election_default_rule"])

    print(f"[load] ownership: {ownership_path}", file=sys.stderr)
    own = pd.read_csv(ownership_path)

    print(f"[load] market: {market_path}", file=sys.stderr)
    mkt = pd.read_csv(market_path)
    # If market has per-event-date rows, collapse to one row per event_id
    # (announce-date row by default). For V1 take the first match.
    if "event_id" in mkt.columns and mkt["event_id"].duplicated().any():
        mkt = mkt.sort_values("event_id").groupby("event_id", as_index=False).first()

    df = ext.merge(own, on="event_id", how="left").merge(mkt, on="event_id", how="left")
    print(f"[load] merged: {len(df):,} events", file=sys.stderr)
    return df


# ---------------------------------------------------------------------------
# Derived columns: realized rates, spread, active-investor election back-out
# ---------------------------------------------------------------------------

def derive_modeling_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Add the columns the EDA + tests need."""
    out = df.copy()

    # Realized cash share — from realized_cash_election_demand if available.
    # If reported as a percentage (>1), divide by 100. If as fraction, leave.
    if "realized_cash_election_demand_num" in out.columns:
        rced = out["realized_cash_election_demand_num"]
        out["realized_cash_share"] = np.where(rced > 1.0, rced / 100.0, rced)
    else:
        out["realized_cash_share"] = np.nan

    # Spread at deadline = cash election value - stock election value per share.
    # Ideally uses prices ON the election deadline; for V1 we use whatever
    # acquirer_price the market CSV provides (likely announcement-date).
    cash_pps = out.get("cash_consideration_per_share_num")
    er = out.get("exchange_ratio_num")
    acq_p = out.get("acquirer_price")
    out["cash_election_value"] = cash_pps
    if er is not None and acq_p is not None:
        out["stock_election_value"] = er * acq_p
    else:
        out["stock_election_value"] = np.nan
    out["spread"] = out["cash_election_value"] - out["stock_election_value"]

    # Oversubscribed flag (only meaningful when cap is a fraction of total).
    if "realized_cash_share" in out.columns and "cash_cap_num" in out.columns:
        cap_frac = np.where(out["cash_cap_num"] <= 1.0, out["cash_cap_num"], np.nan)
        out["cash_cap_frac"] = cap_frac
        out["oversubscribed"] = out["realized_cash_share"] > cap_frac
    else:
        out["cash_cap_frac"] = np.nan
        out["oversubscribed"] = np.nan

    # Passive cash demand share = passive% × (1 if default=cash else 0).
    # `passive_control_percent` per the colleague's spec already includes
    # lent shares on the passive side.
    if "passive_control_percent" in out.columns and "default_rule" in out.columns:
        pcp = pd.to_numeric(out["passive_control_percent"], errors="coerce")
        pcp = np.where(pcp > 1.0, pcp / 100.0, pcp)
        out["passive_pct"] = pcp
        out["passive_cash_demand_share"] = np.where(
            out["default_rule"] == "cash", pcp,
            np.where(out["default_rule"] == "stock", 0.0, 0.5 * pcp)
        )
    else:
        out["passive_pct"] = np.nan
        out["passive_cash_demand_share"] = np.nan

    # Back out active cash-election rate.
    eps = 1e-3
    out["active_share"] = (1.0 - out["passive_pct"]).clip(lower=eps)
    out["active_cash_election_rate"] = (
        (out["realized_cash_share"] - out["passive_cash_demand_share"]) / out["active_share"]
    ).clip(lower=0.0, upper=1.0)

    return out


# ---------------------------------------------------------------------------
# EDA — each chart is its own function
# ---------------------------------------------------------------------------

def _save(fig, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)
    print(f"[plot] wrote {path}", file=sys.stderr)


def plot_realized_cash_election_hist(df: pd.DataFrame, out_dir: Path) -> None:
    s = df["realized_cash_share"].dropna()
    if s.empty:
        print("[plot] skip realized-cash-hist: no data", file=sys.stderr); return
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.hist(s, bins=30, edgecolor="black", alpha=0.8)
    ax.set_xlabel("Realized cash share of consideration")
    ax.set_ylabel("# deals")
    ax.set_title(f"Distribution of realized cash election share (n={len(s)})")
    _save(fig, out_dir / "plots" / "01_realized_cash_share_hist.png")


def plot_spread_vs_cash_election(df: pd.DataFrame, out_dir: Path) -> None:
    d = df.dropna(subset=["spread", "active_cash_election_rate"])
    if d.empty:
        print("[plot] skip spread-vs-election: no data", file=sys.stderr); return
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.scatter(d["spread"], d["active_cash_election_rate"], alpha=0.6)
    ax.axvline(0, color="k", linestyle="--", linewidth=0.5)
    ax.set_xlabel("Spread at deadline ($ cash value − stock value per share)")
    ax.set_ylabel("Active investor cash-election rate (backed out)")
    ax.set_title(f"Active election demand vs. spread (n={len(d)})")
    if len(d) >= 3:
        coef = np.polyfit(d["spread"], d["active_cash_election_rate"], 1)
        xs = np.linspace(d["spread"].min(), d["spread"].max(), 50)
        ax.plot(xs, np.polyval(coef, xs), "r-",
                label=f"y = {coef[0]:.4f}x + {coef[1]:.3f}")
        ax.legend()
    _save(fig, out_dir / "plots" / "02_spread_vs_active_election.png")


def plot_demand_vs_cap(df: pd.DataFrame, out_dir: Path) -> None:
    d = df.dropna(subset=["realized_cash_share", "cash_cap_frac"])
    if d.empty:
        print("[plot] skip demand-vs-cap: no data", file=sys.stderr); return
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.scatter(d["cash_cap_frac"], d["realized_cash_share"], alpha=0.6)
    lim = float(max(d["cash_cap_frac"].max(), d["realized_cash_share"].max(), 1.0))
    ax.plot([0, lim], [0, lim], "k--", linewidth=0.7, label="y = x (no proration)")
    ax.set_xlabel("Cash cap (fraction of total)")
    ax.set_ylabel("Realized cash share")
    ax.set_title(f"Realized demand vs. cap (n={len(d)})")
    ax.legend()
    _save(fig, out_dir / "plots" / "03_demand_vs_cap.png")


def plot_election_by_default_rule(df: pd.DataFrame, out_dir: Path) -> None:
    d = df.dropna(subset=["default_rule", "realized_cash_share"])
    if d.empty:
        print("[plot] skip election-by-default: no data", file=sys.stderr); return
    groups, labels = [], []
    for k in ["cash", "stock", "mixed"]:
        sub = d.loc[d["default_rule"] == k, "realized_cash_share"].values
        if len(sub) > 0:
            groups.append(sub)
            labels.append(f"{k} (n={len(sub)})")
    if not groups:
        return
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.boxplot(groups, labels=labels)
    ax.set_ylabel("Realized cash share")
    ax.set_title("Realized cash election by default rule")
    _save(fig, out_dir / "plots" / "04_election_by_default_rule.png")


def plot_election_by_passive_pct(df: pd.DataFrame, out_dir: Path) -> None:
    d = df.dropna(subset=["passive_pct", "realized_cash_share", "default_rule"])
    if d.empty:
        print("[plot] skip election-by-passive: no data", file=sys.stderr); return
    fig, ax = plt.subplots(figsize=(7, 5))
    colors = {"cash": "tab:blue", "stock": "tab:orange", "mixed": "tab:green"}
    for rule, sub in d.groupby("default_rule"):
        ax.scatter(sub["passive_pct"], sub["realized_cash_share"],
                   alpha=0.6, label=f"default={rule} (n={len(sub)})",
                   color=colors.get(rule, "gray"))
    ax.set_xlabel("Passive control % (incl. lent shares)")
    ax.set_ylabel("Realized cash share")
    ax.set_title("Election outcome vs. passive ownership, by default rule")
    ax.legend()
    _save(fig, out_dir / "plots" / "05_election_by_passive_pct.png")


def plot_active_passive_heatmap(df: pd.DataFrame, out_dir: Path) -> None:
    d = df.dropna(subset=["passive_pct", "spread", "realized_cash_share"])
    if len(d) < 20:
        print(f"[plot] skip heatmap: too few rows ({len(d)})", file=sys.stderr); return
    d = d.assign(
        passive_bucket=pd.qcut(d["passive_pct"], 4, duplicates="drop"),
        spread_bucket=pd.qcut(d["spread"], 4, duplicates="drop"),
    )
    grid = d.groupby(["passive_bucket", "spread_bucket"])["realized_cash_share"].mean().unstack()
    fig, ax = plt.subplots(figsize=(7, 5))
    im = ax.imshow(grid.values, aspect="auto", cmap="RdYlBu_r")
    ax.set_xticks(range(len(grid.columns)))
    ax.set_xticklabels([str(c) for c in grid.columns], rotation=30, ha="right")
    ax.set_yticks(range(len(grid.index)))
    ax.set_yticklabels([str(c) for c in grid.index])
    ax.set_xlabel("Spread bucket"); ax.set_ylabel("Passive % bucket")
    ax.set_title("Mean realized cash share by passive × spread bucket")
    plt.colorbar(im, ax=ax)
    _save(fig, out_dir / "plots" / "06_passive_spread_heatmap.png")


def plot_proration_time_series(df: pd.DataFrame, out_dir: Path) -> None:
    date_col = next(
        (c for c in ["announce_date", "Announce Date", "dateann", "announceddate"] if c in df.columns),
        None,
    )
    if date_col is None:
        print("[plot] skip time-series: no announce date column", file=sys.stderr); return
    d = df.dropna(subset=["realized_cash_share"]).copy()
    d[date_col] = pd.to_datetime(d[date_col], errors="coerce")
    d = d.dropna(subset=[date_col])
    if d.empty:
        return
    d["year"] = d[date_col].dt.year
    yearly = d.groupby("year")["realized_cash_share"].agg(["mean", "count"])
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(yearly.index, yearly["mean"], "o-")
    for x, (mean, count) in zip(yearly.index, yearly.values):
        ax.annotate(f"n={int(count)}", (x, mean), fontsize=7, alpha=0.6,
                    xytext=(0, 5), textcoords="offset points")
    ax.set_xlabel("Year announced"); ax.set_ylabel("Mean realized cash share")
    ax.set_title("Election outcomes over time")
    _save(fig, out_dir / "plots" / "07_proration_time_series.png")


# ---------------------------------------------------------------------------
# Statistical tests
# ---------------------------------------------------------------------------

def run_ols(df: pd.DataFrame, out_dir: Path) -> pd.DataFrame:
    """OLS of realized_cash_share on [spread, passive_pct, default_is_cash, log_deal_size]."""
    feats = ["spread", "passive_pct"]
    df = df.copy()
    if "default_rule" in df.columns:
        df["default_is_cash"] = (df["default_rule"] == "cash").astype(int)
        feats.append("default_is_cash")
    if "deal_value" in df.columns:
        df["log_deal_size"] = np.log1p(coerce_numeric(df["deal_value"]).fillna(0))
        feats.append("log_deal_size")

    d = df.dropna(subset=feats + ["realized_cash_share"])
    if len(d) < len(feats) + 2:
        print(f"[stats] skip OLS: not enough rows ({len(d)})", file=sys.stderr)
        return pd.DataFrame()

    X = np.column_stack([d[f].astype(float).values for f in feats])
    X = np.column_stack([np.ones(len(X)), X])  # intercept
    y = d["realized_cash_share"].values
    coef, *_ = np.linalg.lstsq(X, y, rcond=None)
    resid = y - X @ coef
    ss_res = float((resid ** 2).sum())
    ss_tot = float(((y - y.mean()) ** 2).sum())
    r2 = 1 - ss_res / ss_tot if ss_tot > 0 else np.nan

    rows = [{"feature": "intercept", "coef": coef[0]}]
    for f, c in zip(feats, coef[1:]):
        rows.append({"feature": f, "coef": c})
    rows.append({"feature": "__R2__", "coef": r2})
    rows.append({"feature": "__n__", "coef": len(d)})

    out_path = out_dir / "tables" / "ols_realized_cash_share.csv"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(out_path, index=False)
    print(f"[stats] OLS R²={r2:.3f} on n={len(d)}, wrote {out_path}", file=sys.stderr)
    return pd.DataFrame(rows)


def run_ks_test_by_default(df: pd.DataFrame, out_dir: Path) -> pd.DataFrame:
    """K-S test: are realized_cash_share distributions the same for cash-default vs stock-default?"""
    d = df.dropna(subset=["realized_cash_share", "default_rule"])
    cash = d.loc[d["default_rule"] == "cash", "realized_cash_share"].values
    stock = d.loc[d["default_rule"] == "stock", "realized_cash_share"].values
    if len(cash) < 5 or len(stock) < 5:
        print(f"[stats] skip K-S: need ≥5 per group (cash={len(cash)}, stock={len(stock)})", file=sys.stderr)
        return pd.DataFrame()
    stat, p = stats.ks_2samp(cash, stock)
    out = pd.DataFrame([{
        "test": "ks_2samp",
        "group_a": f"default=cash (n={len(cash)})",
        "group_b": f"default=stock (n={len(stock)})",
        "statistic": stat,
        "p_value": p,
        "reject_same_dist_at_5pct": p < 0.05,
    }])
    out_path = out_dir / "tables" / "ks_test_default_rule.csv"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(out_path, index=False)
    print(f"[stats] K-S stat={stat:.3f}, p={p:.4f}", file=sys.stderr)
    return out


def fit_active_election_function(df: pd.DataFrame, out_dir: Path) -> Dict:
    """Empirical p_active(spread): linear fit for V1. Logistic/piecewise can come later."""
    d = df.dropna(subset=["spread", "active_cash_election_rate"])
    if len(d) < 10:
        print(f"[stats] skip active fit: too few rows ({len(d)})", file=sys.stderr)
        return {}
    x = d["spread"].values
    y = d["active_cash_election_rate"].values
    slope, intercept, r, p, se = stats.linregress(x, y)
    out = {
        "form": "p_active(spread) = intercept + slope * spread",
        "intercept": intercept,
        "slope": slope,
        "r_value": r,
        "p_value": p,
        "std_error": se,
        "n": len(d),
    }
    out_path = out_dir / "tables" / "active_election_function.csv"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame([out]).to_csv(out_path, index=False)
    print(f"[stats] p_active(spread): intercept={intercept:.3f}, slope={slope:.4f}, R={r:.3f}", file=sys.stderr)
    return out


# ---------------------------------------------------------------------------
# Summary writer
# ---------------------------------------------------------------------------

def write_summary(df: pd.DataFrame, out_dir: Path) -> None:
    n = len(df)
    n_with_realized = int(df["realized_cash_share"].notna().sum())
    n_oversub = int(df["oversubscribed"].sum()) if df["oversubscribed"].notna().any() else 0
    md = f"""# Election Arb EDA — Summary

**Total events analyzed:** {n:,}
**Events with realized cash share label:** {n_with_realized:,}
**Events flagged oversubscribed (realized cash > cash cap):** {n_oversub:,}

## V1 modeling assumptions

- Active investors treated as ONE population (no sub-categorization).
- Lent shares are passive (uses `passive_control_percent` from the WRDS
  ownership pipeline, which folds in N-PORT on-loan balances).
- Passive investors take the default rule deterministically.
- `p_active(spread)` is fit as a simple linear function for V1; logistic
  or piecewise refinement can come later.

## Outputs

### Plots
- `plots/01_realized_cash_share_hist.png` — distribution shape (bimodal? clustered at cap?)
- `plots/02_spread_vs_active_election.png` — **the money chart.** Active election demand response to spread.
- `plots/03_demand_vs_cap.png` — how often demand exceeds cap (proration regime)
- `plots/04_election_by_default_rule.png` — boxplot by default rule
- `plots/05_election_by_passive_pct.png` — scatter colored by default rule
- `plots/06_passive_spread_heatmap.png` — interaction effect heatmap
- `plots/07_proration_time_series.png` — time trends (hedge fund concentration effect?)

### Stats tables
- `tables/ols_realized_cash_share.csv` — OLS coefficients for V1 prediction
- `tables/ks_test_default_rule.csv` — formal test that default rule matters
- `tables/active_election_function.csv` — fitted `p_active(spread)` for the Monte Carlo

## Known caveats

- **Spread proxy.** This version uses whatever `acquirer_price` the market CSV
  provides (likely announcement-date). The active election function should
  ideally be fit on the *deadline-date* spread. Worth re-running once the
  market script is extended to pull deadline-date prices.
- **Numeric coercion of Claude outputs.** `realized_cash_election_demand` and
  related fields come back as free-text; we extract the first number. Manual
  review of edge cases (e.g. fields containing dollar amounts vs. percentages)
  is recommended.
- **Sample size.** Election deals are rare. With under a few hundred deals,
  statistical power is limited — interpret p-values conservatively and prefer
  effect-size thinking over significance thresholds.
- **Lent-shares-are-passive assumption.** Per supervisor instruction. If
  in-scope later, the more nuanced view is that the borrower controls the
  vote/election on lent shares — would push some lent stock back into the
  active bucket.
"""
    out_path = out_dir / "eda_summary.md"
    out_path.write_text(md, encoding="utf-8")
    print(f"[write] wrote {out_path}", file=sys.stderr)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--extractions", required=True, type=Path,
                   help="Path to llm_field_extractions.csv (Claude SEC outputs)")
    p.add_argument("--ownership", required=True, type=Path,
                   help="Path to ownership_mix_by_event.csv (WRDS ownership pipeline)")
    p.add_argument("--market", required=True, type=Path,
                   help="Path to event_market_features.csv (WRDS CRSP)")
    p.add_argument("--output-dir", default=Path("eda_output"), type=Path,
                   help="Directory to write plots, tables, and summary")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    df = load_and_merge(args.extractions, args.ownership, args.market)
    df = derive_modeling_columns(df)

    panel_path = args.output_dir / "merged_panel.csv"
    df.to_csv(panel_path, index=False)
    print(f"[write] wrote merged panel: {panel_path}", file=sys.stderr)

    # Plots
    plot_realized_cash_election_hist(df, args.output_dir)
    plot_spread_vs_cash_election(df, args.output_dir)
    plot_demand_vs_cap(df, args.output_dir)
    plot_election_by_default_rule(df, args.output_dir)
    plot_election_by_passive_pct(df, args.output_dir)
    plot_active_passive_heatmap(df, args.output_dir)
    plot_proration_time_series(df, args.output_dir)

    # Stats
    run_ols(df, args.output_dir)
    run_ks_test_by_default(df, args.output_dir)
    fit_active_election_function(df, args.output_dir)

    write_summary(df, args.output_dir)
    print(f"\n[done] outputs in {args.output_dir.resolve()}", file=sys.stderr)


if __name__ == "__main__":
    main()
