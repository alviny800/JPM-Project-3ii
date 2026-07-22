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
import re
import sys
from pathlib import Path
from typing import Dict, Optional, Tuple

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


def first_number(value) -> Optional[float]:
    if value is None:
        return None
    try:
        if pd.isna(value):
            return None
    except Exception:
        pass
    if isinstance(value, (int, float)):
        return float(value)
    text = str(value).replace(",", "")
    match = re.search(r"-?\d+(?:\.\d+)?", text)
    if not match:
        return None
    return float(match.group(0))


def parse_fraction_value(value) -> Optional[float]:
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
    text = str(value).strip().lower().replace(",", "")
    if not text or text in {"nan", "none", "null", "not_found", "not applicable", "n/a"}:
        return None
    pct = re.search(r"([0-9]+(?:\.[0-9]+)?)\s*%", text)
    if pct:
        return float(pct.group(1)) / 100.0
    frac = re.search(r"\b([0-9]+(?:\.[0-9]+)?)\s*/\s*([0-9]+(?:\.[0-9]+)?)\b", text)
    if frac and float(frac.group(2)) != 0:
        return float(frac.group(1)) / float(frac.group(2))
    dec = re.search(r"\b0\.[0-9]+\b", text)
    if dec:
        return float(dec.group(0))
    whole = first_number(text)
    if whole is not None and 1.0 < whole <= 100.0 and any(w in text for w in ["percent", "pct"]):
        return whole / 100.0
    if any(w in text for w in ["one-half", "one half", "half of", "fifty percent"]):
        return 0.5
    return None


def share_from_value(value, denominator: Optional[float] = None) -> Optional[float]:
    text = "" if value is None else str(value).strip().lower()
    if not text or text in {"nan", "none", "null", "not_found", "not applicable", "n/a"}:
        return None
    if "%" in text:
        x = first_number(text)
        return None if x is None else float(np.clip(x / 100.0, 0.0, 1.0))
    x = first_number(value)
    if x is None:
        return None
    if 0.0 <= x <= 1.0:
        return x
    if 1.0 < x <= 100.0 and any(w in text for w in ["percent", "pct"]):
        return x / 100.0
    if denominator and denominator > 0:
        return float(np.clip(x / denominator, 0.0, 1.0))
    if 1.0 < x <= 100.0:
        return x / 100.0
    return None


def proration_factor_from_text(text: str, option: str) -> Optional[float]:
    text_l = "" if text is None else str(text).lower()
    option_l = option.lower()
    windows = []
    for match in re.finditer(option_l, text_l):
        start = max(0, match.start() - 120)
        end = min(len(text_l), match.end() + 180)
        window = text_l[start:end]
        if "prorat" in window or "election" in window:
            windows.append(window)
    if not windows and option_l in text_l:
        windows = [text_l]
    for window in windows:
        frac = parse_fraction_value(window)
        if frac is not None:
            return frac
    return None


def realized_cash_share_and_source(row: pd.Series) -> Tuple[Optional[float], str, float]:
    shares_out = first_number(row.get("target_shares_outstanding"))
    cash = share_from_value(row.get("realized_cash_election_demand"), shares_out)
    if cash is not None:
        return cash, "direct_realized_cash_election_demand", 1.0

    stock = share_from_value(row.get("realized_stock_election_demand"), shares_out)
    if stock is not None:
        return float(np.clip(1.0 - stock, 0.0, 1.0)), "direct_realized_stock_election_demand", 1.0

    cash_cap = parse_fraction_value(row.get("cash_cap"))
    text = row.get("final_proration_results")
    source = "final_proration_results_backed_out_from_cap"
    if text is None or str(text).strip().lower() in {"", "nan", "not_found", "not applicable"}:
        text = row.get("preliminary_proration_results")
        source = "preliminary_proration_results_backed_out_from_cap"
    fill = proration_factor_from_text(str(text), "cash") if text is not None else None
    if cash_cap is not None and fill and fill > 0:
        weight = 0.65 if source.startswith("final") else 0.45
        return float(np.clip(cash_cap / fill, 0.0, 1.0)), source, weight
    return None, "", 0.0


def cap_fraction_from_row(row: pd.Series, field: str) -> Optional[float]:
    raw = row.get(field)
    fraction = parse_fraction_value(raw)
    if fraction is not None:
        return fraction
    cap_shares = first_number(raw)
    shares_out = first_number(row.get("target_shares_outstanding"))
    if cap_shares is not None and shares_out and shares_out > 0 and cap_shares > 100.0:
        return float(np.clip(cap_shares / shares_out, 0.0, 1.0))
    return None
def coerce_percent(s: pd.Series) -> pd.Series:
    """Extract the election PERCENTAGE from prose, on a 0-100 scale.

    Claude returns strings like '108,054,170 shares elected cash, representing 40.85%
    of the 264,507,424 outstanding' — the FIRST number is a share count, not the
    percentage. coerce_numeric() would grab 108054170 and corrupt the label. Here we
    prefer the value immediately before a '%'; fall back to a bare fraction in [0,1]
    (expressed as percent); and NEVER return a raw share count."""
    if s.dtype.kind in "iuf":
        return s
    txt = s.astype(str).str.replace(",", "", regex=False)
    pct = pd.to_numeric(txt.str.extract(r"(\d+\.?\d*)\s*%", expand=False), errors="coerce")
    frac = pd.to_numeric(txt.str.extract(r"(?<![\d.])(0?\.\d+)(?![\d%])", expand=False), errors="coerce")
    frac = frac.where((frac >= 0) & (frac <= 1)) * 100.0
    out = pct.fillna(frac)
    # guard: a valid election/proration share is 0-100%; drop anything outside.
    return out.where((out >= 0) & (out <= 100))


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
    normalized_path: Path = None,
) -> pd.DataFrame:
    """Load the three CSVs and merge into one deal-level DataFrame keyed by event_id.
    If normalized_path is given, its clean `pct_elected_cash` (election DEMAND, separated
    from post-proration allocation by the normalization pass) is merged in and used as the
    dependent variable in place of the crude regex-parsed field."""
    print(f"[load] extractions: {extractions_path}", file=sys.stderr)
    ext_long = pd.read_csv(extractions_path)
    ext = pivot_claude_long_to_wide(ext_long)

    # Dollar/ratio terms: grab the first number. Percentage labels: grab the % (not the
    # leading share count), so realized-demand/proration land on a sane 0-100 scale.
    PCT_FIELDS = {
        "realized_cash_election_demand",
        "realized_stock_election_demand",
        "final_proration_results",
        "preliminary_proration_results",
    }
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
            ext[col + "_num"] = coerce_percent(ext[col]) if col in PCT_FIELDS else coerce_numeric(ext[col])

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

    # Drop columns from the right side that already exist in the accumulating frame
    # (except the join key) so pandas doesn't suffix duplicates as _x/_y. This keeps the
    # extraction's parsed term columns (cash_consideration_per_share_num, exchange_ratio_num)
    # intact — the market CSV carries its own copies, which would otherwise shadow them.
    def _nonoverlap(right: pd.DataFrame, have_cols) -> pd.DataFrame:
        keep = [c for c in right.columns if c == "event_id" or c not in set(have_cols)]
        return right[keep]

    df = ext.copy()
    df = df.merge(_nonoverlap(own, df.columns), on="event_id", how="left")
    df = df.merge(_nonoverlap(mkt, df.columns), on="event_id", how="left")

    if normalized_path is not None and Path(normalized_path).exists():
        print(f"[load] normalized labels: {normalized_path}", file=sys.stderr)
        norm = pd.read_csv(normalized_path)
        keep = [c for c in ["event_id", "pct_elected_cash", "pct_elected_stock",
                            "pct_received_cash", "cash_proration_factor", "disclosure_type"]
                if c in norm.columns]
        df = df.merge(_nonoverlap(norm[keep], df.columns), on="event_id", how="left")

    print(f"[load] merged: {len(df):,} events", file=sys.stderr)
    return df


# ---------------------------------------------------------------------------
# Derived columns: realized rates, spread, active-investor election back-out
# ---------------------------------------------------------------------------

def derive_modeling_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Add the columns the EDA + tests need."""
    out = df.copy()

    realized = out.apply(realized_cash_share_and_source, axis=1)
    out["realized_cash_share"] = [x[0] for x in realized]
    out["realized_label_source"] = [x[1] for x in realized]
    out["realized_label_quality_weight"] = [x[2] for x in realized]
    # Realized cash share = election DEMAND (% who ELECTED cash, pre-proration).
    # Prefer the NORMALIZED pct_elected_cash (the clean, elected-vs-received-separated
    # labels) when present — that's the defensible dependent variable. Only fall back to
    # the crude regex-parsed field when no normalized labels were provided.
    if "pct_elected_cash" in out.columns:
        out["realized_cash_share"] = pd.to_numeric(out["pct_elected_cash"], errors="coerce") / 100.0
    elif "realized_cash_election_demand_num" in out.columns:
        rced = out["realized_cash_election_demand_num"]
        out["realized_cash_share"] = np.where(rced > 1.0, rced / 100.0, rced)
    else:
        out["realized_cash_share"] = np.nan

    # Spread at trade-entry/election-mechanics date when available, falling back
    # to the market snapshot price.
    cash_pps = out.get("cash_consideration_per_share_num")
    er = out.get("exchange_ratio_num")
    if "entry_acquirer_price" in out.columns:
        acq_p = out["entry_acquirer_price"].where(out["entry_acquirer_price"].notna(), out.get("acquirer_price"))
        out["spread_price_source"] = np.where(out["entry_acquirer_price"].notna(), "entry_acquirer_price", "acquirer_price")
    else:
        acq_p = out.get("acquirer_price")
        out["spread_price_source"] = "acquirer_price"
    out["cash_election_value"] = cash_pps
    if er is not None and acq_p is not None:
        out["stock_election_value"] = er * acq_p
    else:
        out["stock_election_value"] = np.nan
    out["spread"] = out["cash_election_value"] - out["stock_election_value"]

    # Oversubscribed flag (only meaningful when cap is a fraction of total).
    if "cash_cap" in out.columns:
        out["cash_cap_frac"] = out.apply(lambda row: cap_fraction_from_row(row, "cash_cap"), axis=1)
        out["oversubscribed"] = out["realized_cash_share"] > out["cash_cap_frac"]
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
    try:
        ax.boxplot(groups, tick_labels=labels)   # matplotlib >= 3.9
    except TypeError:
        ax.boxplot(groups, labels=labels)        # older matplotlib
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
    """Empirical bounded p_active(spread) fit.

    The original V1 used a linear probability model.  For trading, a bounded
    logistic curve is safer because election probabilities cannot be below zero
    or above one.  We keep the linear fit columns for continuity.
    """
    d = df.dropna(subset=["spread", "active_cash_election_rate"])
    if len(d) < 10:
        print(f"[stats] skip active fit: too few rows ({len(d)})", file=sys.stderr)
        return {}
    x = d["spread"].values
    y = d["active_cash_election_rate"].values
    linear_slope, linear_intercept, linear_r, linear_p, linear_se = stats.linregress(x, y)

    y_clip = np.clip(y, 1e-4, 1.0 - 1e-4)
    logit_y = np.log(y_clip / (1.0 - y_clip))
    if "realized_label_quality_weight" in d.columns:
        w = pd.to_numeric(d["realized_label_quality_weight"], errors="coerce").fillna(0.0).values
        w = np.where(w > 0, w, 0.25)
    else:
        w = np.ones(len(d))
    X = np.column_stack([np.ones(len(x)), x])
    Xw = X * np.sqrt(w)[:, None]
    yw = logit_y * np.sqrt(w)
    coef, *_ = np.linalg.lstsq(Xw, yw, rcond=None)
    logistic_intercept = float(coef[0])
    logistic_slope = float(coef[1])
    fitted = 1.0 / (1.0 + np.exp(-(logistic_intercept + logistic_slope * x)))
    resid = y - fitted
    ss_res = float((resid ** 2).sum())
    ss_tot = float(((y - y.mean()) ** 2).sum())
    logistic_r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else np.nan
    out = {
        "form": "p_active(spread) = sigmoid(logistic_intercept + logistic_slope * spread)",
        "logistic_intercept": logistic_intercept,
        "logistic_slope": logistic_slope,
        "logistic_r2_probability_space": logistic_r2,
        "linear_probability_intercept": linear_intercept,
        "linear_probability_slope": linear_slope,
        "linear_probability_r_value": linear_r,
        "linear_probability_p_value": linear_p,
        "linear_probability_std_error": linear_se,
        "n": len(d),
        "effective_n": float(w.sum()),
        "label_weighting": "realized_label_quality_weight",
    }
    out_path = out_dir / "tables" / "active_election_function.csv"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame([out]).to_csv(out_path, index=False)
    print(
        f"[stats] bounded p_active(spread): intercept={logistic_intercept:.3f}, "
        f"slope={logistic_slope:.4f}, R2={logistic_r2:.3f}",
        file=sys.stderr,
    )
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
- Realized labels carry a source/quality flag; direct realized-demand labels
  receive full weight, while proration-backed-out labels receive lower weight.
- `p_active(spread)` is fit as a bounded logistic function, with the older
  linear probability coefficients preserved for diagnostics.

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
  provides when `entry_acquirer_price` is missing. The active election function
  should ideally be fit on the *deadline-date* spread.
- **Numeric coercion of Claude outputs.** `realized_cash_election_demand` and
  related fields come back as free-text; this script now separates percentages,
  share counts, and proration-backed-out labels, but manual review of edge cases
  remains recommended.
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
    p.add_argument("--normalized", default=None, type=Path,
                   help="Optional normalized_labels.csv (clean pct_elected_cash demand); used as the dependent variable when present")
    p.add_argument("--output-dir", default=Path("eda_output"), type=Path,
                   help="Directory to write plots, tables, and summary")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    df = load_and_merge(args.extractions, args.ownership, args.market, args.normalized)
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
