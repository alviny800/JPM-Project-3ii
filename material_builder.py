#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Build slide-ready material for the post-Monte-Carlo arb layers.

The core pipeline writes model artifacts in their native folders.  This module
collects the important outputs and performance diagnostics into ./material so
the results can be dropped into explanatory slides without hunting through the
repo.
"""
from __future__ import annotations

import json
import math
import shutil
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Optional

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


MATERIAL_DIR = Path("material")

LAYER_MAP = [
    {
        "order": 0,
        "layer": "monte_carlo_baseline",
        "source": "arb_run.py, arb_mc.py, arb_backtest.py",
        "role": "Demand model, calibration, realized proration edge, and portfolio MC baseline.",
        "key_outputs": "summary.json, demand/calibration/edge/portfolio charts, MC-ready deal table.",
        "performance": "KS PIT calibration, realized-edge mean/median, portfolio mean/p05.",
    },
    {
        "order": 1,
        "layer": "outcome_risk",
        "source": "arb_outcome.py, deal_outcome_model.py",
        "role": "Convert BBG Deal Status into event-level completed/terminated/withdrawn probabilities.",
        "key_outputs": "deal_outcome_probabilities.csv and class/probability summaries.",
        "performance": "Temporal OOS Brier score/skill, balanced accuracy, macro-F1, class counts, average break probability.",
    },
    {
        "order": 2,
        "layer": "holder_structure",
        "source": "structural_election_model.py via arb_capacity.py",
        "role": "Rolling p/q holder model for noisy versus EV-sensitive election behavior.",
        "key_outputs": "Rolling holder estimates, source counts, p/q distributions, predicted cash demand.",
        "performance": "Coverage, fit history size, prediction MAE/correlation where actual demand exists.",
    },
    {
        "order": 3,
        "layer": "capacity_self_impact",
        "source": "arb_capacity.py",
        "role": "Size trades under source supply, ADV, position limits, and election self-impact.",
        "key_outputs": "Optimal/max notional by trade, binding constraints, expected PnL, cash-demand shift.",
        "performance": "Capacity coverage, total notional, expected PnL, binding constraints, break-even impact.",
    },
    {
        "order": 4,
        "layer": "risk_gated_signal",
        "source": "arb_signal.py",
        "role": "Choose ENTER, REVERSE, REVIEW, or PASS with hedges and risk gates.",
        "key_outputs": "Signal blotter, selected strategy, election, hedge ratio, expected/realized returns.",
        "performance": "Hit rate, expected-vs-realized correlation, downside/loss-probability diagnostics.",
    },
    {
        "order": 5,
        "layer": "combined_strategy",
        "source": "arb_signal.py summarize_strategy",
        "role": "Roll capacity-sized ENTER and REVERSE trades into a portfolio-level strategy view.",
        "key_outputs": "Notional, expected PnL, realized baseline/self-impact PnL, trade-quality summary.",
        "performance": "Weighted expected return, realized return on deployed capital, wrong trades, missed passes.",
    },
    {
        "order": 6,
        "layer": "strategy_results",
        "source": "arb_signal.py risk-adjusted diagnostics",
        "role": "Separate forward-looking strategy economics from completion-only realized-return diagnostics.",
        "key_outputs": "Result metrics, corrected cross-sectional table, and validation-boundary notes.",
        "performance": "Expected PnL, covered realized PnL, expected-vs-realized correlation, mean/std sensitivity.",
    },
]


def _ensure_dir(path: Any = MATERIAL_DIR) -> Path:
    out = Path(path)
    out.mkdir(parents=True, exist_ok=True)
    return out


def _read_csv(path: Any) -> pd.DataFrame:
    p = Path(path)
    if not p.exists():
        return pd.DataFrame()
    return pd.read_csv(p)


def _read_json(path: Any) -> Dict[str, Any]:
    p = Path(path)
    if not p.exists():
        return {}
    with p.open("r", encoding="utf-8") as f:
        return json.load(f)


def _numeric(df: pd.DataFrame, col: str) -> pd.Series:
    if df.empty or col not in df.columns:
        return pd.Series(dtype=float)
    return pd.to_numeric(df[col], errors="coerce")


def _finite(series: Iterable[Any]) -> pd.Series:
    s = pd.to_numeric(pd.Series(series), errors="coerce")
    return s[np.isfinite(s)]


def _finite_mean(series: Iterable[Any]) -> Optional[float]:
    s = _finite(series)
    return float(s.mean()) if len(s) else None


def _finite_median(series: Iterable[Any]) -> Optional[float]:
    s = _finite(series)
    return float(s.median()) if len(s) else None


def _finite_sum(series: Iterable[Any]) -> Optional[float]:
    s = _finite(series)
    return float(s.sum()) if len(s) else None


def _finite_corr(left: Iterable[Any], right: Iterable[Any]) -> Optional[float]:
    pair = pd.DataFrame({
        "left": pd.to_numeric(pd.Series(left), errors="coerce"),
        "right": pd.to_numeric(pd.Series(right), errors="coerce"),
    }).dropna()
    if len(pair) <= 2:
        return None
    return float(np.corrcoef(pair["left"], pair["right"])[0, 1])


def _json_ready(value: Any) -> Any:
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
        return x if math.isfinite(x) else None
    try:
        if pd.isna(value):
            return None
    except Exception:
        pass
    return value


def _write_json(path: Path, data: Dict[str, Any]) -> None:
    path.write_text(json.dumps(_json_ready(data), indent=2, sort_keys=True), encoding="utf-8")


def _write_table(path: Path, rows: List[Dict[str, Any]]) -> pd.DataFrame:
    df = pd.DataFrame(rows)
    df.to_csv(path, index=False)
    return df


def _copy_if_exists(src: Any, dst: Path) -> bool:
    p = Path(src)
    if not p.exists():
        return False
    shutil.copy2(p, dst)
    return True


def _save_bar(path: Path, labels: Iterable[Any], values: Iterable[Any], title: str,
              ylabel: str = "", color: str = "#4C78A8", horizontal: bool = False) -> None:
    labels = [str(x) for x in labels]
    values = [0.0 if pd.isna(x) else float(x) for x in values]
    fig_h = max(3.5, min(10.0, 0.35 * len(labels) + 2.0))
    fig, ax = plt.subplots(figsize=(7.0, fig_h if horizontal else 4.2))
    if horizontal:
        order = np.arange(len(labels))
        ax.barh(order, values, color=color, alpha=0.88)
        ax.set_yticks(order)
        ax.set_yticklabels(labels)
        ax.invert_yaxis()
        ax.set_xlabel(ylabel)
    else:
        ax.bar(labels, values, color=color, alpha=0.88)
        ax.set_ylabel(ylabel)
        ax.tick_params(axis="x", labelrotation=25)
    ax.set_title(title)
    ax.grid(axis="x" if horizontal else "y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def _save_hist(path: Path, values: Iterable[Any], title: str, xlabel: str,
               color: str = "#59A14F", bins: int = 20) -> None:
    clean = _finite(values)
    if not len(clean):
        return
    fig, ax = plt.subplots(figsize=(7, 4.2))
    ax.hist(clean, bins=bins, color=color, alpha=0.82, edgecolor="white")
    ax.axvline(clean.median(), color="black", ls="--", lw=1, label=f"median={clean.median():.2f}")
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel("count")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def _save_scatter(path: Path, x: Iterable[Any], y: Iterable[Any], title: str,
                  xlabel: str, ylabel: str, color: str = "#4C78A8") -> None:
    df = pd.DataFrame({
        "x": pd.to_numeric(pd.Series(x), errors="coerce"),
        "y": pd.to_numeric(pd.Series(y), errors="coerce"),
    }).dropna()
    if len(df) < 2:
        return
    fig, ax = plt.subplots(figsize=(6.2, 5.0))
    ax.scatter(df["x"], df["y"], s=38, color=color, alpha=0.82)
    corr = _finite_corr(df["x"], df["y"])
    subtitle = "" if corr is None else f" (corr={corr:+.2f})"
    ax.set_title(title + subtitle)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def _normalize_fraction_series(series: pd.Series) -> pd.Series:
    x = pd.to_numeric(series, errors="coerce")
    return x.where(~((x > 1.0) & (x <= 100.0)), x / 100.0)


def write_layer_map(output_dir: Any = MATERIAL_DIR) -> pd.DataFrame:
    out = _ensure_dir(output_dir)
    df = pd.DataFrame(LAYER_MAP)
    df.to_csv(out / "00_layer_map.csv", index=False)
    return df


def export_mc_material(output_dir: Any = MATERIAL_DIR,
                       arb_output_dir: Any = "arb_output",
                       deals_path: Any = "arb_deals.csv") -> Dict[str, Any]:
    out = _ensure_dir(output_dir)
    summary = _read_json(Path(arb_output_dir) / "summary.json")
    deals = _read_csv(deals_path)

    if not deals.empty:
        cols = [c for c in [
            "event_id", "target_name", "ratio_type", "C", "R", "P_acq", "stock_val",
            "spread", "pi_cash", "pi_cash_source", "f_cash", "deal_outcome_label",
        ] if c in deals.columns]
        deals[cols].to_csv(out / "00_mc_ready_deal_terms.csv", index=False)
        coverage = []
        for col in ["C", "R", "P_acq", "spread", "pi_cash", "f_cash"]:
            if col in deals.columns:
                coverage.append({
                    "field": col,
                    "non_null": int(deals[col].notna().sum()),
                    "coverage_pct": float(deals[col].notna().mean() * 100.0),
                })
        coverage_df = _write_table(out / "00_mc_terms_coverage.csv", coverage)
        if not coverage_df.empty:
            _save_bar(
                out / "00_mc_terms_coverage.png",
                coverage_df["field"],
                coverage_df["coverage_pct"],
                "MC input coverage",
                "coverage %",
                color="#4C78A8",
            )
        if "spread" in deals.columns:
            _save_hist(out / "00_mc_spread_distribution.png", deals["spread"], "Deadline spread distribution", "spread ($)")

    for name in [
        "demand_distribution.png",
        "calibration_pit.png",
        "beta_qq.png",
        "realized_edge.png",
        "portfolio_pnl.png",
    ]:
        _copy_if_exists(Path(arb_output_dir) / name, out / f"00_mc_{name}")
    _copy_if_exists(Path(arb_output_dir) / "summary.md", out / "00_mc_summary.md")
    if summary:
        _write_json(out / "00_mc_performance_summary.json", summary)
        metrics = {
            "demand_calibration_set": summary.get("demand_calibration_set"),
            "calibration_ks_p": summary.get("calibration_ks_p"),
            "mc_ready_deals": summary.get("mc_ready_deals"),
            "realized_edge_mean_pct": summary.get("realized_edge_mean_pct"),
            "portfolio_edge_mean_pct": summary.get("portfolio_edge_mean_pct"),
            "portfolio_with_break_mean_pct": summary.get("portfolio_with_break_mean_pct"),
            "portfolio_with_break_p05_pct": summary.get("portfolio_with_break_p05_pct"),
        }
        metric_rows = [{"metric": k, "value": v} for k, v in metrics.items()]
        _write_table(out / "00_mc_key_metrics.csv", metric_rows)
    return {"layer": "monte_carlo_baseline", "status": "ok", "summary_keys": sorted(summary.keys())}


def _export_outcome_temporal_oos(
    out: Path,
    bbg_path: Any,
    min_train_per_class: int = 10,
) -> Dict[str, Any]:
    """Evaluate tuned NB probabilities with nested temporal validation."""
    from arb_outcome import (
        BbgOutcomeNaiveBayes,
        OUTCOME_STATES,
        OutcomeDefaults,
        OutcomeNBParams,
        load_bbg_with_keys,
        precompute_bbg_model_features,
        tune_bbg_outcome_naive_bayes,
        tune_outcome_decision_prior_power,
    )

    labels = list(OUTCOME_STATES)
    bbg = load_bbg_with_keys(str(bbg_path))
    bbg = bbg[bbg["_outcome_label"].isin(labels)].copy()
    bbg["_announce_date"] = pd.to_datetime(bbg["Announce Date"], errors="coerce")
    bbg = bbg.dropna(subset=["_announce_date"]).sort_values("_announce_date")
    bbg = precompute_bbg_model_features(bbg)
    if bbg.empty:
        return {"oos_evaluation_status": "missing_labeled_bbg_rows"}

    records: List[Dict[str, Any]] = []
    tuning_choices: List[Dict[str, Any]] = []
    inner_min_train_per_class = max(3, min_train_per_class // 2)
    first_year = int(bbg["_announce_date"].dt.year.min())
    last_year = int(bbg["_announce_date"].dt.year.max())
    for year in range(first_year + 1, last_year + 1):
        cutoff = pd.Timestamp(year=year, month=1, day=1)
        next_cutoff = pd.Timestamp(year=year + 1, month=1, day=1)
        train = bbg[bbg["_announce_date"] < cutoff]
        test = bbg[(bbg["_announce_date"] >= cutoff) & (bbg["_announce_date"] < next_cutoff)]
        train_counts = train["_outcome_label"].value_counts()
        if test.empty or any(int(train_counts.get(label, 0)) < min_train_per_class for label in labels):
            continue

        tuned_params, tuning = tune_bbg_outcome_naive_bayes(
            train,
            OutcomeDefaults(),
            min_train_per_class=inner_min_train_per_class,
        )
        tuned_model = BbgOutcomeNaiveBayes(
            train,
            OutcomeDefaults(),
            params=tuned_params,
        )
        untuned_model = BbgOutcomeNaiveBayes(
            train,
            OutcomeDefaults(),
            params=OutcomeNBParams(),
        )
        decision_prior_power, decision_tuning = tune_outcome_decision_prior_power(
            train,
            tuned_params,
            OutcomeDefaults(),
            min_train_per_class=inner_min_train_per_class,
        )
        untuned_decision_prior_power, _ = tune_outcome_decision_prior_power(
            train,
            OutcomeNBParams(),
            OutcomeDefaults(),
            min_train_per_class=inner_min_train_per_class,
        )
        best_tuning = tuning.iloc[0]
        best_decision_tuning = decision_tuning.iloc[0]
        tuning_choices.append({
            "outer_test_year": year,
            "outer_train_rows": tuned_model.fit_n,
            **tuned_params.as_dict(),
            "inner_validation_years": best_tuning.get("validation_years", ""),
            "inner_validation_rows": best_tuning.get("validation_rows", 0),
            "inner_validation_brier_score": best_tuning.get("multiclass_brier_score"),
            "decision_prior_power": decision_prior_power,
            "decision_inner_macro_f1": best_decision_tuning.get("macro_f1"),
            "decision_inner_balanced_accuracy": best_decision_tuning.get("balanced_accuracy"),
        })
        for _, row in test.iterrows():
            probs = tuned_model.predict_proba(row)
            untuned_probs = untuned_model.predict_proba(row)
            records.append({
                "announce_year": year,
                "announce_date": row.get("Announce Date"),
                "target_name": row.get("Target Name"),
                "acquirer_name": row.get("Acquirer Name"),
                "actual_outcome_label": row["_outcome_label"],
                **{f"p_{label}": probs[label] for label in labels},
                **{f"untuned_p_{label}": untuned_probs[label] for label in labels},
                **{f"prior_{label}": untuned_model.priors[label] for label in labels},
                "train_n": tuned_model.fit_n,
                **{f"train_{label}": tuned_model.label_counts[label] for label in labels},
                **{f"tuned_{key}": value for key, value in tuned_params.as_dict().items()},
                "decision_prior_power": decision_prior_power,
                "untuned_decision_prior_power": untuned_decision_prior_power,
            })

    predictions = pd.DataFrame(records)
    if predictions.empty:
        return {"oos_evaluation_status": "insufficient_temporal_training_history"}
    predictions.to_csv(out / "01_outcome_temporal_oos_predictions.csv", index=False)
    _write_table(out / "01_outcome_temporal_tuning_choices.csv", tuning_choices)

    actual = predictions["actual_outcome_label"].astype(str)
    probs = predictions[[f"p_{label}" for label in labels]].to_numpy(dtype=float)
    untuned_probs = predictions[[f"untuned_p_{label}" for label in labels]].to_numpy(dtype=float)
    priors = predictions[[f"prior_{label}" for label in labels]].to_numpy(dtype=float)
    onehot = np.column_stack([(actual == label).astype(float) for label in labels])
    actual_idx = np.array([labels.index(label) for label in actual], dtype=int)
    eps = 1e-12

    tuned_brier = float(np.mean(np.sum((probs - onehot) ** 2, axis=1)))
    untuned_brier = float(np.mean(np.sum((untuned_probs - onehot) ** 2, axis=1)))
    prior_brier = float(np.mean(np.sum((priors - onehot) ** 2, axis=1)))
    tuned_nll = float(-np.mean(np.log(np.clip(probs[np.arange(len(probs)), actual_idx], eps, 1.0))))
    untuned_nll = float(-np.mean(np.log(np.clip(untuned_probs[np.arange(len(probs)), actual_idx], eps, 1.0))))
    prior_nll = float(-np.mean(np.log(np.clip(priors[np.arange(len(priors)), actual_idx], eps, 1.0))))
    tuned_brier_skill = float(1.0 - tuned_brier / prior_brier) if prior_brier > 0 else None
    untuned_brier_skill = float(1.0 - untuned_brier / prior_brier) if prior_brier > 0 else None

    def classification_metrics(prob_matrix: np.ndarray) -> Tuple[pd.Series, Dict[str, Dict[str, Any]], float, float]:
        pred_idx = prob_matrix.argmax(axis=1)
        pred_label = pd.Series([labels[i] for i in pred_idx], index=predictions.index)
        class_metrics: Dict[str, Dict[str, Any]] = {}
        for label in labels:
            actual_label = actual == label
            predicted_label = pred_label == label
            tp = int((actual_label & predicted_label).sum())
            fp = int((~actual_label & predicted_label).sum())
            fn = int((actual_label & ~predicted_label).sum())
            precision = float(tp / (tp + fp)) if tp + fp else 0.0
            recall = float(tp / (tp + fn)) if tp + fn else 0.0
            f1 = float(2.0 * precision * recall / (precision + recall)) if precision + recall else 0.0
            class_metrics[label] = {
                "rows": int(actual_label.sum()),
                "precision": precision,
                "recall": recall,
                "f1": f1,
            }
        balanced_accuracy = float(np.mean([class_metrics[label]["recall"] for label in labels]))
        macro_f1 = float(np.mean([class_metrics[label]["f1"] for label in labels]))
        return pred_label, class_metrics, balanced_accuracy, macro_f1

    _, plain_class_metrics, plain_balanced_accuracy, plain_macro_f1 = classification_metrics(probs)
    _, untuned_plain_class_metrics, untuned_plain_balanced_accuracy, untuned_plain_macro_f1 = classification_metrics(untuned_probs)
    prior_floor = np.clip(priors, 1e-12, None)
    decision_power = predictions["decision_prior_power"].to_numpy(dtype=float)[:, None]
    untuned_decision_power = predictions["untuned_decision_prior_power"].to_numpy(dtype=float)[:, None]
    pred_label, class_metrics, balanced_accuracy, macro_f1 = classification_metrics(
        probs / np.power(prior_floor, decision_power)
    )
    _, untuned_class_metrics, untuned_balanced_accuracy, untuned_macro_f1 = classification_metrics(
        untuned_probs / np.power(prior_floor, untuned_decision_power)
    )

    cm = pd.crosstab(actual, pred_label, rownames=["actual"], colnames=["predicted"], dropna=False)
    cm = cm.reindex(index=labels, columns=labels, fill_value=0)
    cm.to_csv(out / "01_outcome_confusion_matrix.csv")
    by_actual = predictions.groupby("actual_outcome_label").agg(
        rows=("actual_outcome_label", "size"),
        avg_p_completed=("p_completed", "mean"),
        avg_p_terminated=("p_terminated", "mean"),
        avg_p_withdrawn=("p_withdrawn", "mean"),
    ).reset_index()
    by_actual.to_csv(out / "01_outcome_probability_by_actual.csv", index=False)

    year_rows: List[Dict[str, Any]] = []
    for year, group in predictions.groupby("announce_year"):
        year_actual = group["actual_outcome_label"].astype(str)
        year_probs = group[[f"p_{label}" for label in labels]].to_numpy(dtype=float)
        year_untuned_probs = group[[f"untuned_p_{label}" for label in labels]].to_numpy(dtype=float)
        year_priors = group[[f"prior_{label}" for label in labels]].to_numpy(dtype=float)
        year_onehot = np.column_stack([(year_actual == label).astype(float) for label in labels])
        year_tuned_brier = float(np.mean(np.sum((year_probs - year_onehot) ** 2, axis=1)))
        year_untuned_brier = float(np.mean(np.sum((year_untuned_probs - year_onehot) ** 2, axis=1)))
        year_prior_brier = float(np.mean(np.sum((year_priors - year_onehot) ** 2, axis=1)))
        year_rows.append({
            "announce_year": int(year),
            "rows": int(len(group)),
            "tuned_nb_brier_score": year_tuned_brier,
            "untuned_nb_brier_score": year_untuned_brier,
            "prior_brier_score": year_prior_brier,
            "tuned_brier_skill_score": float(1.0 - year_tuned_brier / year_prior_brier) if year_prior_brier > 0 else None,
            "untuned_brier_skill_score": float(1.0 - year_untuned_brier / year_prior_brier) if year_prior_brier > 0 else None,
        })
    _write_table(out / "01_outcome_temporal_oos_by_year.csv", year_rows)
    metric_rows = [
        {"metric": "tuned_nb_multiclass_brier_score", "value": tuned_brier},
        {"metric": "untuned_nb_multiclass_brier_score", "value": untuned_brier},
        {"metric": "prior_brier_score", "value": prior_brier},
        {"metric": "tuned_nb_brier_skill_score", "value": tuned_brier_skill},
        {"metric": "untuned_nb_brier_skill_score", "value": untuned_brier_skill},
        {"metric": "tuned_nb_negative_log_likelihood", "value": tuned_nll},
        {"metric": "untuned_nb_negative_log_likelihood", "value": untuned_nll},
        {"metric": "prior_negative_log_likelihood", "value": prior_nll},
        {"metric": "tuned_nb_balanced_accuracy", "value": balanced_accuracy},
        {"metric": "untuned_nb_balanced_accuracy", "value": untuned_balanced_accuracy},
        {"metric": "tuned_nb_macro_f1", "value": macro_f1},
        {"metric": "untuned_nb_macro_f1", "value": untuned_macro_f1},
        {"metric": "tuned_nb_plain_argmax_balanced_accuracy", "value": plain_balanced_accuracy},
        {"metric": "untuned_nb_plain_argmax_balanced_accuracy", "value": untuned_plain_balanced_accuracy},
        {"metric": "tuned_nb_plain_argmax_macro_f1", "value": plain_macro_f1},
        {"metric": "untuned_nb_plain_argmax_macro_f1", "value": untuned_plain_macro_f1},
    ]
    _write_table(out / "01_outcome_temporal_oos_metrics.csv", metric_rows)
    _save_bar(
        out / "01_outcome_brier_comparison.png",
        ["Tuned NB", "Untuned NB", "Prior-only"],
        [tuned_brier, untuned_brier, prior_brier],
        "Temporal out-of-sample probability error",
        "multiclass Brier score (lower is better)",
        color="#B07AA1",
    )

    fig, ax = plt.subplots(figsize=(5.4, 4.8))
    im = ax.imshow(cm.values, cmap="Blues")
    ax.set_xticks(range(len(labels))); ax.set_xticklabels(labels, rotation=25)
    ax.set_yticks(range(len(labels))); ax.set_yticklabels(labels)
    ax.set_title("Tuned NB temporal OOS: prior-adjusted decisions")
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, int(cm.values[i, j]), ha="center", va="center", color="black")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(out / "01_outcome_confusion_matrix.png", dpi=150)
    plt.close(fig)

    full_params, full_tuning = tune_bbg_outcome_naive_bayes(
        bbg,
        OutcomeDefaults(),
        min_train_per_class=min_train_per_class,
    )
    full_tuning.to_csv(out / "01_outcome_full_training_tuning_grid.csv", index=False)
    full_decision_power, full_decision_tuning = tune_outcome_decision_prior_power(
        bbg,
        full_params,
        OutcomeDefaults(),
        min_train_per_class=inner_min_train_per_class,
    )
    full_decision_tuning.to_csv(out / "01_outcome_full_training_decision_tuning.csv", index=False)

    return {
        "oos_evaluation_status": "ok",
        "oos_evaluation_method": (
            "Nested expanding-window calendar-year evaluation. Each outer test year "
            "uses only earlier announcement years; hyperparameters are selected by "
            "an inner expanding-window Brier-score search within the outer training "
            "sample. Outer folds require at least "
            f"{min_train_per_class} observations per class; inner folds require "
            f"{inner_min_train_per_class}."
        ),
        "oos_start_year": int(predictions["announce_year"].min()),
        "oos_end_year": int(predictions["announce_year"].max()),
        "oos_row_count": int(len(predictions)),
        "oos_actual_label_counts": actual.value_counts().to_dict(),
        "oos_multiclass_brier_score": tuned_brier,
        "oos_tuned_nb_multiclass_brier_score": tuned_brier,
        "oos_untuned_nb_multiclass_brier_score": untuned_brier,
        "oos_prior_brier_score": prior_brier,
        "oos_brier_skill_score": tuned_brier_skill,
        "oos_tuned_nb_brier_skill_score": tuned_brier_skill,
        "oos_untuned_nb_brier_skill_score": untuned_brier_skill,
        "oos_negative_log_likelihood": tuned_nll,
        "oos_tuned_nb_negative_log_likelihood": tuned_nll,
        "oos_untuned_nb_negative_log_likelihood": untuned_nll,
        "oos_prior_negative_log_likelihood": prior_nll,
        "oos_balanced_accuracy": balanced_accuracy,
        "oos_tuned_nb_balanced_accuracy": balanced_accuracy,
        "oos_untuned_nb_balanced_accuracy": untuned_balanced_accuracy,
        "oos_macro_f1": macro_f1,
        "oos_tuned_nb_macro_f1": macro_f1,
        "oos_untuned_nb_macro_f1": untuned_macro_f1,
        "oos_classification_decision_rule": (
            "argmax of predicted probability divided by training prior raised to "
            "a fold-specific power selected on earlier years by macro-F1"
        ),
        "oos_tuned_nb_plain_argmax_balanced_accuracy": plain_balanced_accuracy,
        "oos_untuned_nb_plain_argmax_balanced_accuracy": untuned_plain_balanced_accuracy,
        "oos_tuned_nb_plain_argmax_macro_f1": plain_macro_f1,
        "oos_untuned_nb_plain_argmax_macro_f1": untuned_plain_macro_f1,
        "oos_class_metrics": class_metrics,
        "oos_untuned_class_metrics": untuned_class_metrics,
        "oos_plain_argmax_class_metrics": plain_class_metrics,
        "oos_untuned_plain_argmax_class_metrics": untuned_plain_class_metrics,
        "full_training_selected_params": full_params.as_dict(),
        "full_training_selected_decision_prior_power": full_decision_power,
    }


def export_outcome_material(
    output_dir: Any = MATERIAL_DIR,
    outcome_path: Any = "deal_outcome_probabilities.csv",
    bbg_path: Any = "BBG Data Pull 2006+ Final.csv",
) -> Dict[str, Any]:
    out = _ensure_dir(output_dir)
    df = _read_csv(outcome_path)
    if df.empty:
        return {"layer": "outcome_risk", "status": "missing_input", "path": str(outcome_path)}

    df.to_csv(out / "01_outcome_probabilities.csv", index=False)
    prob_cols = ["p_completed", "p_terminated", "p_withdrawn"]
    present_probs = [c for c in prob_cols if c in df.columns]
    df["_p_break"] = _numeric(df, "p_terminated").fillna(0) + _numeric(df, "p_withdrawn").fillna(0)
    avg_probs = [
        {"state": c.replace("p_", ""), "avg_probability": float(_numeric(df, c).mean())}
        for c in present_probs
    ]
    avg_probs.append({"state": "break", "avg_probability": float(df["_p_break"].mean())})
    avg_df = _write_table(out / "01_outcome_average_probabilities.csv", avg_probs)
    _save_bar(
        out / "01_outcome_average_probabilities.png",
        avg_df["state"],
        avg_df["avg_probability"] * 100,
        "Average event outcome probabilities",
        "probability %",
        color="#B07AA1",
    )
    _save_hist(out / "01_outcome_break_probability_distribution.png", df["_p_break"] * 100,
               "Break probability distribution", "p terminated + p withdrawn (%)", "#E15759")

    label_col = "actual_outcome_label" if "actual_outcome_label" in df.columns else ""
    labeled = df[df[label_col].astype(str).isin(["completed", "terminated", "withdrawn"])].copy() if label_col else pd.DataFrame()
    summary: Dict[str, Any] = {
        "row_count": int(len(df)),
        "source_counts": df.get("outcome_probability_source", pd.Series(dtype=object)).value_counts(dropna=False).to_dict(),
        "match_source_counts": df.get("outcome_match_source", pd.Series(dtype=object)).value_counts(dropna=False).to_dict(),
        "avg_p_completed": _finite_mean(_numeric(df, "p_completed")),
        "avg_p_terminated": _finite_mean(_numeric(df, "p_terminated")),
        "avg_p_withdrawn": _finite_mean(_numeric(df, "p_withdrawn")),
        "avg_p_break": _finite_mean(df["_p_break"]),
    }
    if not labeled.empty:
        summary.update({
            "event_frame_labeled_row_count": int(len(labeled)),
            "event_frame_actual_label_counts": labeled[label_col].astype(str).value_counts().to_dict(),
            "event_frame_performance_note": (
                "The matched 293-event frame is completion-only and is not used "
                "to report three-class classifier performance."
            ),
        })
    try:
        summary.update(_export_outcome_temporal_oos(out, bbg_path))
    except Exception as exc:
        summary.update({
            "oos_evaluation_status": "error",
            "oos_evaluation_error": str(exc),
        })

    _write_json(out / "01_outcome_performance_summary.json", summary)
    return {"layer": "outcome_risk", "status": "ok", "row_count": int(len(df))}


def export_holder_material(output_dir: Any = MATERIAL_DIR,
                           panel_path: Any = "eda_output/merged_panel.csv") -> Dict[str, Any]:
    out = _ensure_dir(output_dir)
    try:
        from arb_capacity import CapacityConfig, load_capacity_table
        holder = load_capacity_table(str(panel_path), CapacityConfig()).reset_index()
    except Exception as exc:
        _write_json(out / "02_holder_material_error.json", {"error": str(exc)})
        return {"layer": "holder_structure", "status": "error", "error": str(exc)}
    if holder.empty:
        return {"layer": "holder_structure", "status": "missing_input", "path": str(panel_path)}

    holder.to_csv(out / "02_holder_model_estimates.csv", index=False)
    source_counts = holder.get("holder_model_source", pd.Series(dtype=object)).value_counts(dropna=False).reset_index()
    if not source_counts.empty:
        source_counts.columns = ["holder_model_source", "rows"]
        source_counts.to_csv(out / "02_holder_source_counts.csv", index=False)
        _save_bar(out / "02_holder_source_counts.png", source_counts["holder_model_source"],
                  source_counts["rows"], "Holder model source coverage", "rows", "#76B7B2", horizontal=True)

    actual = pd.Series(dtype=float)
    if "pct_elected_cash" in holder.columns:
        actual = _normalize_fraction_series(holder["pct_elected_cash"])
    elif "realized_cash_share" in holder.columns:
        actual = _normalize_fraction_series(holder["realized_cash_share"])
    predicted = _numeric(holder, "holder_predicted_cash_demand_share")
    pair = pd.DataFrame({"actual": actual, "predicted": predicted}).dropna()
    if len(pair):
        pair.to_csv(out / "02_holder_predicted_vs_actual.csv", index=False)
        _save_scatter(out / "02_holder_predicted_vs_actual_cash_demand.png",
                      pair["predicted"] * 100, pair["actual"] * 100,
                      "Holder model predicted vs actual demand",
                      "predicted cash demand %", "actual cash demand %", "#59A14F")
    _save_scatter(out / "02_holder_p_q_scatter.png",
                  _numeric(holder, "holder_p_hat"), _numeric(holder, "holder_q_hat"),
                  "Rolling holder p/q estimates", "p: noisy cash probability", "q: EV-sensitive ownership", "#F28E2B")
    _save_hist(out / "02_holder_fit_history_distribution.png", _numeric(holder, "holder_fit_n"),
               "Rolling holder fit history", "prior labeled events", "#4C78A8")

    summary = {
        "row_count": int(len(holder)),
        "source_counts": holder.get("holder_model_source", pd.Series(dtype=object)).value_counts(dropna=False).to_dict(),
        "median_fit_n": _finite_median(_numeric(holder, "holder_fit_n")),
        "median_p_hat": _finite_median(_numeric(holder, "holder_p_hat")),
        "median_q_hat": _finite_median(_numeric(holder, "holder_q_hat")),
        "median_positive_share": _finite_median(_numeric(holder, "holder_positive_share")),
        "median_noisy_share": _finite_median(_numeric(holder, "holder_noisy_share")),
        "prediction_coverage": int(len(pair)),
        "prediction_mae_pctpts": float((pair["actual"] - pair["predicted"]).abs().mean() * 100.0) if len(pair) else None,
        "prediction_corr": _finite_corr(pair["predicted"], pair["actual"]) if len(pair) else None,
    }
    _write_json(out / "02_holder_performance_summary.json", summary)
    return {"layer": "holder_structure", "status": "ok", "row_count": int(len(holder))}


def export_capacity_material(output_dir: Any = MATERIAL_DIR,
                             signals_path: Any = "arb_signals.csv") -> Dict[str, Any]:
    out = _ensure_dir(output_dir)
    sig = _read_csv(signals_path)
    if sig.empty:
        return {"layer": "capacity_self_impact", "status": "missing_input", "path": str(signals_path)}
    trades = sig[sig.get("signal", pd.Series(dtype=object)).astype(str).isin(["ENTER", "REVERSE"])].copy()
    if trades.empty:
        return {"layer": "capacity_self_impact", "status": "no_trades"}

    cols = [c for c in [
        "event_id", "target", "signal", "selected_strategy", "capacity_status",
        "capacity_optimal_notional", "capacity_optimal_expected_pnl",
        "capacity_optimal_E_return_%", "capacity_optimal_pct_shares_outstanding",
        "capacity_optimal_binding_constraint", "capacity_raw_max_notional",
        "capacity_alpha_decay_%", "capacity_optimal_cash_demand_shift_pctpts",
        "self_impact_can_eliminate_arbitrage", "self_impact_break_even_pct_shares_outstanding",
    ] if c in trades.columns]
    trades[cols].sort_values("capacity_optimal_expected_pnl", ascending=False).to_csv(
        out / "03_capacity_trade_sizing.csv", index=False
    )
    bind = trades.get("capacity_optimal_binding_constraint", pd.Series(dtype=object)).replace("", "unknown").value_counts().reset_index()
    if not bind.empty:
        bind.columns = ["binding_constraint", "trades"]
        bind.to_csv(out / "03_capacity_binding_constraints.csv", index=False)
        _save_bar(out / "03_capacity_binding_constraints.png", bind["binding_constraint"],
                  bind["trades"], "Capacity binding constraints", "trades", "#E15759", horizontal=True)

    ranked = trades.dropna(subset=["capacity_optimal_notional"]).copy()
    ranked = ranked.sort_values("capacity_optimal_notional", ascending=True).tail(20)
    if len(ranked):
        _save_bar(out / "03_capacity_notional_by_trade.png", ranked["target"],
                  _numeric(ranked, "capacity_optimal_notional") / 1e6,
                  "Top capacity-sized trades", "optimal notional ($mm)", "#4C78A8", horizontal=True)
        _save_bar(out / "03_capacity_expected_pnl_by_trade.png", ranked["target"],
                  _numeric(ranked, "capacity_optimal_expected_pnl") / 1e6,
                  "Expected PnL by trade", "expected PnL ($mm)", "#59A14F", horizontal=True)
    _save_scatter(out / "03_capacity_notional_vs_expected_return.png",
                  _numeric(trades, "capacity_optimal_notional") / 1e6,
                  _numeric(trades, "capacity_optimal_E_return_%"),
                  "Capacity notional vs expected return", "optimal notional ($mm)", "E return %", "#B07AA1")
    _save_hist(out / "03_capacity_self_impact_break_even.png",
               _numeric(trades, "self_impact_break_even_pct_shares_outstanding"),
               "Self-impact break-even size", "% shares outstanding", "#F28E2B")

    ok = trades.get("capacity_status", pd.Series("", index=trades.index)).astype(str).eq("ok")
    notional = _numeric(trades, "capacity_optimal_notional")
    expected_pnl = _numeric(trades, "capacity_optimal_expected_pnl")
    summary = {
        "trade_count": int(len(trades)),
        "capacity_ok_count": int(ok.sum()),
        "capacity_ok_pct": float(ok.mean() * 100.0) if len(ok) else None,
        "total_optimal_notional": _finite_sum(notional[ok]),
        "total_expected_pnl": _finite_sum(expected_pnl[ok]),
        "notional_weighted_expected_return_%": (
            float(expected_pnl[ok].sum() / notional[ok].sum() * 100.0)
            if ok.any() and notional[ok].sum() > 0 else None
        ),
        "median_optimal_notional": _finite_median(notional[ok]),
        "median_pct_shares_outstanding": _finite_median(_numeric(trades, "capacity_optimal_pct_shares_outstanding")[ok]),
        "median_alpha_decay_pctpts": _finite_median(_numeric(trades, "capacity_alpha_decay_%")[ok]),
        "binding_constraint_counts": trades.get("capacity_optimal_binding_constraint", pd.Series(dtype=object)).value_counts().to_dict(),
        "self_impact_can_eliminate_count": int(
            trades.get("self_impact_can_eliminate_arbitrage", pd.Series(False, index=trades.index)).fillna(False).astype(bool).sum()
        ),
        "median_self_impact_break_even_pct_shares": _finite_median(_numeric(trades, "self_impact_break_even_pct_shares_outstanding")),
    }
    _write_json(out / "03_capacity_performance_summary.json", summary)
    return {"layer": "capacity_self_impact", "status": "ok", "trade_count": int(len(trades))}


def export_signal_material(output_dir: Any = MATERIAL_DIR,
                           signals_path: Any = "arb_signals.csv") -> Dict[str, Any]:
    out = _ensure_dir(output_dir)
    sig = _read_csv(signals_path)
    if sig.empty:
        return {"layer": "risk_gated_signal", "status": "missing_input", "path": str(signals_path)}
    sig.to_csv(out / "04_signal_full_blotter.csv", index=False)
    focus_cols = [c for c in [
        "event_id", "target", "signal", "selected_strategy", "signal_reason", "elect",
        "E_return_%", "downside_5%_%", "loss_probability_%", "realized_return_%",
        "p_completed", "p_terminated", "p_withdrawn", "hedge_side", "hedge_ratio",
        "capacity_optimal_notional", "capacity_optimal_expected_pnl",
        "capacity_adjusted_E_return_%", "capacity_adjusted_realized_return_%",
    ] if c in sig.columns]
    sig[focus_cols].sort_values("E_return_%", ascending=False).to_csv(out / "04_signal_slide_blotter.csv", index=False)
    counts = sig["signal"].astype(str).value_counts().reset_index()
    counts.columns = ["signal", "rows"]
    counts.to_csv(out / "04_signal_counts.csv", index=False)
    _save_bar(out / "04_signal_counts.png", counts["signal"], counts["rows"], "Risk-gated signal counts", "deals", "#4C78A8")

    trades = sig[sig["signal"].astype(str).isin(["ENTER", "REVERSE"])].copy()
    if len(trades):
        _save_scatter(out / "04_signal_expected_vs_realized.png",
                      _numeric(trades, "E_return_%"), _numeric(trades, "realized_return_%"),
                      "Signal expected vs realized return", "expected return %", "realized return %", "#59A14F")
        side_perf = trades.groupby("signal").agg(
            trades=("signal", "count"),
            avg_E_return_pct=("E_return_%", "mean"),
            avg_realized_return_pct=("realized_return_%", "mean"),
            hit_rate_pct=("realized_return_%", lambda s: float((pd.to_numeric(s, errors="coerce").dropna() > 0).mean() * 100.0)),
        ).reset_index()
        side_perf.to_csv(out / "04_signal_performance_by_trade_type.csv", index=False)
        _save_bar(out / "04_signal_trade_type_realized_return.png", side_perf["signal"],
                  side_perf["avg_realized_return_pct"], "Realized return by trade type", "realized return %", "#F28E2B")
        _save_hist(out / "04_signal_trade_return_distribution.png", _numeric(trades, "realized_return_%"),
                   "Trade realized return distribution", "realized return %", "#76B7B2")

    realized = _numeric(trades, "realized_return_%")
    e_ret = _numeric(trades, "E_return_%")
    cap_realized = _numeric(trades, "capacity_adjusted_realized_return_%")
    cap_e = _numeric(trades, "capacity_adjusted_E_return_%")
    summary = {
        "priced_signal_count": int(len(sig)),
        "trade_count": int(len(trades)),
        "signal_counts": sig["signal"].astype(str).value_counts().to_dict(),
        "mean_trade_expected_return_%": _finite_mean(e_ret),
        "mean_trade_realized_return_%": _finite_mean(realized),
        "trade_hit_rate_%": float((realized.dropna() > 0).mean() * 100.0) if len(realized.dropna()) else None,
        "expected_vs_realized_corr": _finite_corr(e_ret, realized),
        "mean_capacity_adjusted_expected_return_%": _finite_mean(cap_e),
        "mean_capacity_adjusted_realized_return_%": _finite_mean(cap_realized),
        "capacity_adjusted_hit_rate_%": float((cap_realized.dropna() > 0).mean() * 100.0) if len(cap_realized.dropna()) else None,
        "capacity_adjusted_expected_vs_realized_corr": _finite_corr(cap_e, cap_realized),
        "avg_downside_5pct_%": _finite_mean(_numeric(trades, "downside_5%_%")),
        "avg_loss_probability_%": _finite_mean(_numeric(trades, "loss_probability_%")),
    }
    _write_json(out / "04_signal_performance_summary.json", summary)
    return {"layer": "risk_gated_signal", "status": "ok", "row_count": int(len(sig))}


def export_strategy_material(output_dir: Any = MATERIAL_DIR,
                             summary_path: Any = "arb_strategy_summary.json") -> Dict[str, Any]:
    out = _ensure_dir(output_dir)
    summary = _read_json(summary_path)
    if not summary:
        return {"layer": "combined_strategy", "status": "missing_input", "path": str(summary_path)}
    _write_json(out / "05_strategy_summary.json", summary)
    pd.json_normalize(summary, sep=".").to_csv(out / "05_strategy_summary_flat.csv", index=False)

    opt = summary.get("profit_optimal", {})
    max_book = summary.get("risk_gated_maximum", {})
    realized_base = summary.get("realized_baseline_accept_historical_election", {}).get("optimal", {})
    realized_self = summary.get("realized_self_impact_election", {}).get("optimal", {})
    rows = [
        {"metric": "optimal_notional", "value": opt.get("total_notional")},
        {"metric": "optimal_expected_pnl", "value": opt.get("total_expected_pnl")},
        {"metric": "optimal_weighted_expected_return_%", "value": opt.get("notional_weighted_expected_return_%")},
        {"metric": "risk_gated_max_notional", "value": max_book.get("total_notional")},
        {"metric": "realized_baseline_pnl", "value": realized_base.get("total_realized_pnl")},
        {"metric": "realized_baseline_return_on_deployed_%", "value": realized_base.get("return_on_deployed_notional_%")},
        {"metric": "realized_self_impact_pnl", "value": realized_self.get("total_realized_pnl")},
        {"metric": "realized_self_impact_return_on_deployed_%", "value": realized_self.get("return_on_deployed_notional_%")},
        {"metric": "wrong_trades_self_impact", "value": realized_self.get("wrong_trade_count")},
    ]
    key = _write_table(out / "05_strategy_key_metrics.csv", rows)
    pnl_rows = key[key["metric"].isin([
        "optimal_expected_pnl", "realized_baseline_pnl", "realized_self_impact_pnl",
    ])].copy()
    if not pnl_rows.empty:
        _save_bar(out / "05_strategy_pnl_bridge.png", pnl_rows["metric"],
                  pd.to_numeric(pnl_rows["value"], errors="coerce") / 1e6,
                  "Strategy PnL bridge", "$mm", "#59A14F")
    count_rows = [
        {"bucket": "ENTER", "count": summary.get("enter_count", 0)},
        {"bucket": "REVERSE", "count": summary.get("reverse_count", 0)},
        {"bucket": "REVIEW", "count": summary.get("review_count", 0)},
        {"bucket": "PASS", "count": summary.get("pass_count", 0)},
    ]
    count_df = _write_table(out / "05_strategy_trade_counts.csv", count_rows)
    _save_bar(out / "05_strategy_trade_counts.png", count_df["bucket"], count_df["count"],
              "Combined strategy action counts", "deals", "#4C78A8")
    return {"layer": "combined_strategy", "status": "ok", "keys": sorted(summary.keys())}


def export_strategy_result_material(output_dir: Any = MATERIAL_DIR,
                                    summary_path: Any = "arb_strategy_summary.json",
                                    signals_path: Any = "arb_signals.csv",
                                    strategy_daily_path: Any = "arb_strategy_daily_returns.csv",
                                    strategy_event_daily_path: Any = "arb_strategy_event_daily_returns.csv") -> Dict[str, Any]:
    out = _ensure_dir(output_dir)
    summary = _read_json(summary_path)
    sig = _read_csv(signals_path)
    if not summary or sig.empty:
        return {
            "layer": "strategy_results",
            "status": "missing_input",
            "summary_path": str(summary_path),
            "signals_path": str(signals_path),
        }

    opt = summary.get("profit_optimal", {})
    realized = summary.get("realized_self_impact_election", {}).get("optimal", {})
    diagnostics = summary.get("risk_adjusted_performance", {})
    from arb_signal import _render_risk_png, _risk_adjusted_block
    current_trades = sig[
        sig.get("signal", pd.Series(dtype=object)).astype(str).isin(["ENTER", "REVERSE"])
    ].copy()
    if not current_trades.empty:
        diagnostics = _risk_adjusted_block(current_trades)
    history = summary.get("historical_performance", {})
    all_diag = diagnostics.get("all_trades", {})
    ex_top = diagnostics.get("all_ex_largest_winner", {})
    completion_history = history.get("completion_only", {})
    all_history = history.get("all_tradable", {})
    completion_active = completion_history.get("active_day_capacity_weighted", {})
    completion_calendar = completion_history.get("calendar_day_capacity_weighted", {})
    all_active = all_history.get("active_day_capacity_weighted", {})
    all_calendar = all_history.get("calendar_day_capacity_weighted", {})

    metrics = [
        {"metric": "trade_count", "value": summary.get("trade_count")},
        {"metric": "enter_count", "value": summary.get("enter_count")},
        {"metric": "reverse_count", "value": summary.get("reverse_count")},
        {"metric": "optimal_notional", "value": opt.get("total_notional")},
        {"metric": "optimal_expected_pnl", "value": opt.get("total_expected_pnl")},
        {"metric": "optimal_weighted_expected_return_%", "value": opt.get("notional_weighted_expected_return_%")},
        {"metric": "realized_covered_trade_count", "value": realized.get("covered_trade_count")},
        {"metric": "realized_self_impact_pnl", "value": realized.get("total_realized_pnl")},
        {"metric": "realized_return_on_deployed_%", "value": realized.get("return_on_deployed_notional_%")},
        {"metric": "expected_vs_realized_corr", "value": realized.get("correlation_expected_vs_realized")},
        {"metric": "completion_only_mean_return_%", "value": all_diag.get("mean_%")},
        {"metric": "completion_only_median_return_%", "value": all_diag.get("median_%")},
        {"metric": "completion_only_mean_to_std", "value": all_diag.get("cross_sectional_mean_to_std")},
        {"metric": "completion_only_sortino_0pct_mar", "value": all_diag.get("sortino_0pct_mar")},
        {"metric": "completion_only_ex_top_mean_return_%", "value": ex_top.get("mean_%")},
        {"metric": "completion_only_negative_trade_count", "value": all_diag.get("negative_trade_count")},
        {"metric": "completion_only_history_trade_count", "value": completion_history.get("trade_count")},
        {"metric": "completion_only_active_day_sharpe", "value": completion_active.get("annualized_sharpe_0rf")},
        {"metric": "completion_only_calendar_day_sharpe", "value": completion_calendar.get("annualized_sharpe_0rf")},
        {"metric": "all_tradable_history_trade_count", "value": all_history.get("trade_count")},
        {"metric": "all_tradable_active_day_sharpe", "value": all_active.get("annualized_sharpe_0rf")},
        {"metric": "all_tradable_calendar_day_sharpe", "value": all_calendar.get("annualized_sharpe_0rf")},
    ]
    _write_table(out / "06_strategy_result_key_metrics.csv", metrics)

    order = [
        ("all_trades", "All trades"),
        ("enter", "ENTER"),
        ("reverse", "REVERSE"),
        ("all_ex_largest_winner", "Ex-top winner"),
    ]
    rows = []
    for key, label in order:
        block = diagnostics.get(key, {})
        if not block:
            continue
        rows.append({
            "sample": label,
            "n": block.get("n"),
            "mean_%": block.get("mean_%"),
            "median_%": block.get("median_%"),
            "std_%": block.get("std_%"),
            "negative_trade_count": block.get("negative_trade_count"),
            "hit_rate_%": block.get("hit_rate_%"),
            "cross_sectional_mean_to_std": block.get("cross_sectional_mean_to_std"),
            "sortino_0pct_mar": block.get("sortino_0pct_mar"),
        })
    _write_table(out / "06_strategy_result_completion_only_diagnostics.csv", rows)
    _render_risk_png(
        diagnostics,
        str(out / "06_strategy_result_completion_only_diagnostics.png"),
    )

    event_daily = _read_csv(strategy_event_daily_path)
    if not event_daily.empty:
        event_daily["date"] = pd.to_datetime(event_daily["date"], errors="coerce")
        event_daily["daily_return"] = pd.to_numeric(
            event_daily["daily_return"], errors="coerce"
        )
        event_daily["notional"] = pd.to_numeric(
            event_daily["notional"], errors="coerce"
        )
        event_daily = event_daily.dropna(
            subset=["event_id", "date", "daily_return", "notional"]
        )
        event_daily["active_notional"] = event_daily.groupby("date")[
            "notional"
        ].transform("sum")
        event_daily["daily_portfolio_contribution"] = (
            event_daily["notional"]
            / event_daily["active_notional"]
            * event_daily["daily_return"]
        )
        event_curve = (
            event_daily.groupby("event_id", as_index=False)
            .agg(
                close_date=("date", "max"),
                signal=("signal", "first"),
                actual_outcome=("actual_outcome", "first"),
                return_source=("return_source", "first"),
                notional=("notional", "first"),
                event_return=("daily_return", "sum"),
                portfolio_return_contribution=(
                    "daily_portfolio_contribution", "sum"
                ),
            )
            .sort_values(["close_date", "event_id"])
            .reset_index(drop=True)
        )
        target_map = sig.set_index("event_id").get("target", pd.Series(dtype=object))
        event_curve["target"] = event_curve["event_id"].map(target_map)
        event_curve["event_number"] = np.arange(1, len(event_curve) + 1)
        event_curve["event_return_%"] = event_curve["event_return"] * 100.0
        event_curve["portfolio_return_contribution_%"] = (
            event_curve["portfolio_return_contribution"] * 100.0
        )
        event_curve["cumulative_return_%"] = event_curve[
            "portfolio_return_contribution_%"
        ].cumsum()
        event_curve.to_csv(
            out / "06_strategy_result_event_cumulative_return.csv", index=False
        )

        fig, ax = plt.subplots(figsize=(7.2, 3.6))
        ax.plot(
            event_curve["event_number"],
            event_curve["cumulative_return_%"],
            color="#4C78A8",
            linewidth=2.5,
            marker="o",
            markersize=4.5,
        )
        ax.set_title("Cumulative capacity-weighted contribution by event")
        ax.set_xlabel("event number (ordered by close date)")
        ax.set_ylabel("additive cumulative portfolio return (%)")
        ax.set_xticks(event_curve["event_number"])
        ax.margins(y=0.12)
        ax.grid(alpha=0.25)
        final_event_return = float(event_curve["cumulative_return_%"].iloc[-1])
        ax.annotate(
            f"{final_event_return:.1f}%",
            xy=(event_curve["event_number"].iloc[-1], final_event_return),
            xytext=(-8, 8),
            textcoords="offset points",
            ha="right",
            color="#003057",
            weight="bold",
        )
        fig.tight_layout()
        fig.savefig(
            out / "06_strategy_result_event_cumulative_return.png",
            dpi=180,
        )
        plt.close(fig)

    strategy_daily = _read_csv(strategy_daily_path)
    if not strategy_daily.empty:
        if "scope" in strategy_daily.columns:
            all_tradable_daily = strategy_daily[
                strategy_daily["scope"].astype(str).eq("all_tradable")
            ].copy()
            if all_tradable_daily.empty:
                all_tradable_daily = strategy_daily.copy()
        else:
            all_tradable_daily = strategy_daily.copy()
        all_tradable_daily["date"] = pd.to_datetime(
            all_tradable_daily["date"], errors="coerce"
        )
        all_tradable_daily["capacity_weighted_return"] = pd.to_numeric(
            all_tradable_daily["capacity_weighted_return"], errors="coerce"
        )
        time_curve = (
            all_tradable_daily.dropna(
                subset=["date", "capacity_weighted_return"]
            )
            .sort_values("date")
            .reset_index(drop=True)
        )
        time_curve["daily_return_%"] = (
            time_curve["capacity_weighted_return"] * 100.0
        )
        time_curve["cumulative_return_%"] = (
            time_curve["capacity_weighted_return"].cumsum() * 100.0
        )
        time_curve.to_csv(
            out / "06_strategy_result_time_cumulative_return.csv", index=False
        )

        fig, ax = plt.subplots(figsize=(7.2, 3.6))
        ax.plot(
            time_curve["date"],
            time_curve["cumulative_return_%"],
            color="#59A14F",
            linewidth=2.4,
        )
        ax.set_title("Cumulative capacity-weighted portfolio return over time")
        ax.set_xlabel("calendar date")
        ax.set_ylabel("additive cumulative portfolio return (%)")
        ax.margins(y=0.12)
        ax.grid(alpha=0.25)
        final_time_return = float(time_curve["cumulative_return_%"].iloc[-1])
        ax.annotate(
            f"{final_time_return:.1f}%",
            xy=(time_curve["date"].iloc[-1], final_time_return),
            xytext=(-8, 8),
            textcoords="offset points",
            ha="right",
            color="#003057",
            weight="bold",
        )
        fig.tight_layout()
        fig.savefig(
            out / "06_strategy_result_time_cumulative_return.png",
            dpi=180,
        )
        plt.close(fig)

    historical_rows = [
        {
            "scope": "Completion only",
            "trade_count": completion_history.get("trade_count"),
            "active_day_sharpe": completion_active.get("annualized_sharpe_0rf"),
            "calendar_day_sharpe": completion_calendar.get("annualized_sharpe_0rf"),
            "active_day_sortino": completion_active.get("annualized_sortino_0pct_mar"),
            "calendar_day_sortino": completion_calendar.get("annualized_sortino_0pct_mar"),
            "calendar_max_drawdown_%": completion_calendar.get("max_drawdown_additive_%"),
        },
        {
            "scope": "All tradable",
            "trade_count": all_history.get("trade_count"),
            "active_day_sharpe": all_active.get("annualized_sharpe_0rf"),
            "calendar_day_sharpe": all_calendar.get("annualized_sharpe_0rf"),
            "active_day_sortino": all_active.get("annualized_sortino_0pct_mar"),
            "calendar_day_sortino": all_calendar.get("annualized_sortino_0pct_mar"),
            "calendar_max_drawdown_%": all_calendar.get("max_drawdown_additive_%"),
        },
    ]
    history_table = _write_table(out / "06_strategy_result_historical_sharpe.csv", historical_rows)

    daily_performance_rows = [
        {
            "sample": "Active-position days",
            "days": completion_active.get("day_count"),
            "mean_daily_return_%": completion_active.get("mean_daily_return_%"),
            "annualized_volatility_%": completion_active.get("annualized_volatility_%"),
            "annualized_sharpe_0rf": completion_active.get("annualized_sharpe_0rf"),
            "annualized_sortino_0pct_mar": completion_active.get("annualized_sortino_0pct_mar"),
            "cumulative_additive_return_%": completion_active.get("cumulative_additive_return_%"),
            "max_drawdown_additive_%": completion_active.get("max_drawdown_additive_%"),
        },
        {
            "sample": "Full calendar",
            "days": completion_calendar.get("day_count"),
            "mean_daily_return_%": completion_calendar.get("mean_daily_return_%"),
            "annualized_volatility_%": completion_calendar.get("annualized_volatility_%"),
            "annualized_sharpe_0rf": completion_calendar.get("annualized_sharpe_0rf"),
            "annualized_sortino_0pct_mar": completion_calendar.get("annualized_sortino_0pct_mar"),
            "cumulative_additive_return_%": completion_calendar.get("cumulative_additive_return_%"),
            "max_drawdown_additive_%": completion_calendar.get("max_drawdown_additive_%"),
        },
    ]
    daily_performance = _write_table(
        out / "06_strategy_result_daily_performance.csv",
        daily_performance_rows,
    )
    if not daily_performance.empty:
        display_cols = [
            ("days", "Days"),
            ("mean_daily_return_%", "Mean daily %"),
            ("annualized_volatility_%", "Ann. vol %"),
            ("annualized_sharpe_0rf", "Sharpe"),
            ("annualized_sortino_0pct_mar", "Sortino"),
            ("cumulative_additive_return_%", "Cum. return %"),
            ("max_drawdown_additive_%", "Max DD %"),
        ]

        def _daily_fmt(value: Any, key: str) -> str:
            if value is None or pd.isna(value):
                return "—"
            if key == "days":
                return f"{int(value):,}"
            return f"{float(value):.2f}"

        cell_text = [
            [_daily_fmt(row[key], key) for key, _ in display_cols]
            for _, row in daily_performance.iterrows()
        ]
        fig, ax = plt.subplots(figsize=(11.3, 2.6))
        ax.axis("off")
        table = ax.table(
            cellText=cell_text,
            rowLabels=daily_performance["sample"].tolist(),
            colLabels=[label for _, label in display_cols],
            colWidths=[0.09, 0.15, 0.13, 0.10, 0.11, 0.16, 0.13],
            loc="center",
            cellLoc="center",
        )
        table.auto_set_font_size(False)
        table.set_fontsize(10.5)
        table.scale(1, 1.65)
        for column_index in range(len(display_cols)):
            table[0, column_index].set_facecolor("#003057")
            table[0, column_index].set_text_props(color="white", weight="bold")
        ax.set_title(
            "Historical daily strategy performance (18 reconstructed paths)",
            fontsize=13,
            weight="bold",
            color="#003057",
            pad=14,
        )
        fig.text(
            0.5,
            0.03,
            "Capacity-weighted across concurrent positions; 0% risk-free rate and MAR. "
            "Full calendar inserts zero-return inactive business days.",
            ha="center",
            fontsize=8,
            style="italic",
            color="#434545",
        )
        fig.tight_layout()
        fig.savefig(
            out / "06_strategy_result_daily_performance.png",
            dpi=180,
            bbox_inches="tight",
        )
        plt.close(fig)

    sharpe_plot = pd.DataFrame([
        {"series": "Completion: active days", "sharpe": completion_active.get("annualized_sharpe_0rf")},
        {"series": "Completion: full calendar", "sharpe": completion_calendar.get("annualized_sharpe_0rf")},
        {"series": "All tradable: active days", "sharpe": all_active.get("annualized_sharpe_0rf")},
        {"series": "All tradable: full calendar", "sharpe": all_calendar.get("annualized_sharpe_0rf")},
    ])
    if not history_table.empty:
        _save_bar(
            out / "06_strategy_result_historical_sharpe.png",
            sharpe_plot["series"],
            pd.to_numeric(sharpe_plot["sharpe"], errors="coerce"),
            "Historical strategy Sharpe by scope",
            "annualized Sharpe (0% risk-free rate)",
            "#4C78A8",
        )

    validation = {
        "headline_use": (
            "Use expected PnL, capacity, and nested-OOS probability metrics for the forward-looking strategy."
        ),
        "completion_only_use": (
            "Use realized-return statistics only as cross-sectional sensitivity on completed election outcomes."
        ),
        "not_claimed": (
            "No annualized strategy Sharpe or investable Sortino is claimed because broken deals lack realized election demand."
        ),
        "sortino_definition": (
            "Mean realized return divided by downside deviation relative to a 0% minimum acceptable return."
        ),
        "historical_sharpe_method": history.get("method"),
        "historical_scope_comparison": history.get("scope_comparison_note"),
        "historical_outcome_counts": history.get("actual_outcome_counts"),
        "historical_return_source_counts": history.get("return_source_counts"),
        "source_note": diagnostics.get("note"),
    }
    _write_json(out / "06_strategy_result_validation_boundary.json", validation)
    return {
        "layer": "strategy_results",
        "status": "ok",
        "trade_count": summary.get("trade_count"),
        "covered_realized_trade_count": realized.get("covered_trade_count"),
    }


def write_manifest(output_dir: Any = MATERIAL_DIR) -> None:
    out = _ensure_dir(output_dir)
    write_layer_map(out)
    files = sorted(p for p in out.iterdir() if p.is_file())
    index = []
    for p in files:
        if p.name == "material_index.csv":
            continue
        index.append({
            "file": p.name,
            "suffix": p.suffix.lstrip("."),
            "size_bytes": p.stat().st_size,
        })
    pd.DataFrame(index).to_csv(out / "material_index.csv", index=False)

    layer_lines = "\n".join(
        f"- {row['order']}. {row['layer']}: {row['role']} Performance: {row['performance']}"
        for row in LAYER_MAP
    )
    file_lines = "\n".join(f"- `{p.name}`" for p in files)
    manifest = f"""# Slide Material Manifest

This folder centralizes outputs for the layers after the Monte Carlo payoff engine.

## Layers
{layer_lines}

## Files
{file_lines}
"""
    (out / "material_manifest.md").write_text(manifest, encoding="utf-8")


def _safe_export(output_dir: Any, fn: Callable[..., Dict[str, Any]], *args: Any, **kwargs: Any) -> Dict[str, Any]:
    out = _ensure_dir(output_dir)
    try:
        return fn(output_dir=out, *args, **kwargs)
    except Exception as exc:
        error = {"layer": fn.__name__, "status": "error", "error": str(exc)}
        with (out / "material_errors.log").open("a", encoding="utf-8") as f:
            f.write(json.dumps(error, sort_keys=True) + "\n")
        return error


def export_after_arb_run(output_dir: Any = MATERIAL_DIR) -> List[Dict[str, Any]]:
    results = [
        _safe_export(output_dir, export_mc_material),
    ]
    write_manifest(output_dir)
    return results


def export_after_outcome(output_dir: Any = MATERIAL_DIR) -> List[Dict[str, Any]]:
    results = [
        _safe_export(output_dir, export_outcome_material),
    ]
    write_manifest(output_dir)
    return results


def export_after_signal(output_dir: Any = MATERIAL_DIR) -> List[Dict[str, Any]]:
    results = [
        _safe_export(output_dir, export_holder_material),
        _safe_export(output_dir, export_capacity_material),
        _safe_export(output_dir, export_signal_material),
        _safe_export(output_dir, export_strategy_material),
        _safe_export(output_dir, export_strategy_result_material),
    ]
    write_manifest(output_dir)
    return results


def build_all_material(output_dir: Any = MATERIAL_DIR) -> List[Dict[str, Any]]:
    results = [
        _safe_export(output_dir, export_mc_material),
        _safe_export(output_dir, export_outcome_material),
        _safe_export(output_dir, export_holder_material),
        _safe_export(output_dir, export_capacity_material),
        _safe_export(output_dir, export_signal_material),
        _safe_export(output_dir, export_strategy_material),
        _safe_export(output_dir, export_strategy_result_material),
    ]
    write_manifest(output_dir)
    return results


if __name__ == "__main__":
    results = build_all_material()
    print(json.dumps(_json_ready(results), indent=2, sort_keys=True))
