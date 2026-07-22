#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Deal-outcome probability model for M&A arbitrage.

This module deliberately stays independent from election/proration value
estimation.  It uses only pre-outcome deal, ownership, and market fields to
predict whether a deal completes, is mutually terminated, or is withdrawn.
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional, Tuple

import pandas as pd


OUTCOME_LABELS = ["completed", "terminated", "withdrawn"]

NUMERIC_FEATURES = [
    "Announced Total Value (mil.)",
    "TV/EBITDA",
    "target_price",
    "acquirer_price",
    "entry_target_price",
    "entry_acquirer_price",
    "deal_spread",
    "deal_spread_pct",
    "target_adv20",
    "target_dollar_adv20",
    "target_bid_ask_spread_pct",
    "target_market_cap",
    "target_shares_outstanding",
    "acquirer_adv20",
    "acquirer_bid_ask_spread_pct",
    "short_volume_ratio",
    "short_volume_ratio_20d_avg",
    "passive_control_percent",
    "etf_ownership_percent",
    "cash_consideration_per_share_num",
    "exchange_ratio_num",
    "cash_cap_fraction",
    "stock_cap_fraction",
]

CATEGORICAL_FEATURES = [
    "Payment Type",
    "payment_type",
    "default_rule",
    "non_election_default_rule",
    "is_election_menu",
]


def clean_str(value: Any) -> str:
    if value is None:
        return ""
    try:
        if pd.isna(value):
            return ""
    except Exception:
        pass
    text = str(value).strip()
    if text.lower() in {"", "nan", "none", "null", "not_found", "not applicable", "n/a"}:
        return ""
    return text


def numeric_value(value: Any) -> Optional[float]:
    if value is None:
        return None
    try:
        if pd.isna(value):
            return None
    except Exception:
        pass
    if isinstance(value, (int, float)):
        x = float(value)
        return x if math.isfinite(x) else None
    text = clean_str(value).replace(",", "")
    if not text:
        return None
    match = re.search(r"-?\d+(?:\.\d+)?", text)
    if not match:
        return None
    try:
        x = float(match.group(0))
        return x if math.isfinite(x) else None
    except Exception:
        return None


def parse_fraction(value: Any) -> Optional[float]:
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
    text = clean_str(value).lower().replace(",", "")
    if not text:
        return None
    pct = re.search(r"([0-9]+(?:\.[0-9]+)?)\s*%", text)
    if pct:
        return float(pct.group(1)) / 100.0
    frac = re.search(r"\b([0-9]+(?:\.[0-9]+)?)\s*/\s*([0-9]+(?:\.[0-9]+)?)\b", text)
    if frac and float(frac.group(2)) != 0:
        return float(frac.group(1)) / float(frac.group(2))
    decimal = re.search(r"\b0\.[0-9]+\b", text)
    if decimal:
        return float(decimal.group(0))
    return None


def normalize_outcome_label(row: pd.Series) -> Optional[str]:
    texts = [
        clean_str(row.get("deal_completion_or_break")),
        clean_str(row.get("Deal Status")),
        clean_str(row.get("deal_status")),
        clean_str(row.get("status")),
    ]
    joined = " ".join(t.lower() for t in texts if t)
    if not joined:
        return None

    if any(w in joined for w in ["withdrawn", "withdraw", "acquirer withdrew", "offer withdrawn"]):
        return "withdrawn"
    if any(w in joined for w in ["terminated", "termination", "mutual termination", "abandoned", "cancelled", "canceled"]):
        return "terminated"
    if any(w in joined for w in ["completed", "complete", "closed", "closing", "effective time", "consummated"]):
        return "completed"
    return None


def normalize_category(value: Any) -> str:
    text = clean_str(value).lower()
    if not text:
        return "__missing__"
    text = re.sub(r"\s+", " ", text)
    if "cash or stock" in text:
        return "cash_or_stock"
    if "cash and stock" in text:
        return "cash_and_stock"
    if "cash" in text and "stock" not in text:
        return "cash"
    if "stock" in text and "cash" not in text:
        return "stock"
    if "mixed" in text or ("cash" in text and "stock" in text):
        return "mixed"
    return text[:80]


def numeric_feature_value(row: pd.Series, feature: str) -> Optional[float]:
    if feature.endswith("_fraction") and feature not in row.index:
        base = feature.replace("_fraction", "")
        value = parse_fraction(row.get(base))
    else:
        value = numeric_value(row.get(feature))
    if value is None:
        return None
    if feature.endswith("_pct") or "percent" in feature or feature.endswith("_fraction"):
        return value
    # Keep large-scale fields from dominating Gaussian likelihoods.
    return math.copysign(math.log1p(abs(value)), value)


def categorical_feature_value(row: pd.Series, feature: str) -> str:
    if feature == "default_rule":
        source = clean_str(row.get("default_rule")) or clean_str(row.get("non_election_default_rule"))
        return normalize_category(source)
    if feature == "is_election_menu":
        value = row.get(feature)
        if isinstance(value, bool):
            return "true" if value else "false"
        text = clean_str(value).lower()
        if text in {"true", "1", "yes"}:
            return "true"
        if text in {"false", "0", "no"}:
            return "false"
        menu = clean_str(row.get("consideration_menu")).lower()
        payment = clean_str(row.get("Payment Type") or row.get("payment_type")).lower()
        is_menu = "cash or stock" in payment or ("cash election" in menu and "stock election" in menu)
        return "true" if is_menu else "false"
    return normalize_category(row.get(feature))


@dataclass
class DealOutcomeModel:
    priors: Dict[str, float]
    numeric_stats: Dict[str, Dict[str, Tuple[float, float]]]
    categorical_counts: Dict[str, Dict[str, Dict[str, float]]]
    category_values: Dict[str, List[str]]
    fit_n: int
    fit_effective_labels: Dict[str, int]
    global_numeric_stats: Dict[str, Tuple[float, float]]
    alpha: float = 1.0

    def predict_proba(self, row: pd.Series) -> Dict[str, float]:
        scores: Dict[str, float] = {}
        for label in OUTCOME_LABELS:
            score = math.log(max(self.priors.get(label, 0.0), 1e-12))
            for feature in NUMERIC_FEATURES:
                value = numeric_feature_value(row, feature)
                if value is None:
                    continue
                mean, var = self.numeric_stats.get(label, {}).get(
                    feature,
                    self.global_numeric_stats.get(feature, (0.0, 1.0)),
                )
                var = max(var, 1e-4)
                score += -0.5 * (math.log(2.0 * math.pi * var) + ((value - mean) ** 2) / var)
            for feature in CATEGORICAL_FEATURES:
                value = categorical_feature_value(row, feature)
                values = self.category_values.get(feature, ["__missing__"])
                counts = self.categorical_counts.get(label, {}).get(feature, {})
                denom = sum(counts.values()) + self.alpha * max(1, len(values))
                score += math.log((counts.get(value, 0.0) + self.alpha) / max(denom, 1e-12))
            scores[label] = score

        max_score = max(scores.values())
        exp_scores = {k: math.exp(v - max_score) for k, v in scores.items()}
        total = sum(exp_scores.values())
        if total <= 0:
            return dict(self.priors)
        return {k: exp_scores[k] / total for k in OUTCOME_LABELS}


def fit_deal_outcome_model(
    rows: Iterable[pd.Series],
    default_probs: Optional[Dict[str, float]] = None,
    alpha: float = 1.0,
) -> DealOutcomeModel:
    labeled: List[Tuple[pd.Series, str]] = []
    for row in rows:
        label = normalize_outcome_label(row)
        if label in OUTCOME_LABELS:
            labeled.append((row, label))

    default_probs = default_probs or {"completed": 0.90, "terminated": 0.07, "withdrawn": 0.03}
    total_default = sum(max(0.0, default_probs.get(label, 0.0)) for label in OUTCOME_LABELS)
    if total_default <= 0:
        default_probs = {"completed": 0.90, "terminated": 0.07, "withdrawn": 0.03}
        total_default = 1.0
    default_probs = {label: max(0.0, default_probs.get(label, 0.0)) / total_default for label in OUTCOME_LABELS}

    counts = {label: 0 for label in OUTCOME_LABELS}
    for _, label in labeled:
        counts[label] += 1
    fit_n = len(labeled)
    prior_total = fit_n + alpha * len(OUTCOME_LABELS)
    priors = {
        label: (counts[label] + alpha * default_probs[label]) / max(prior_total, 1e-12)
        for label in OUTCOME_LABELS
    }
    norm = sum(priors.values())
    priors = {k: v / norm for k, v in priors.items()}

    global_numeric_stats: Dict[str, Tuple[float, float]] = {}
    numeric_stats: Dict[str, Dict[str, Tuple[float, float]]] = {label: {} for label in OUTCOME_LABELS}
    for feature in NUMERIC_FEATURES:
        all_values = [numeric_feature_value(row, feature) for row, _ in labeled]
        all_values = [v for v in all_values if v is not None]
        if all_values:
            g_mean = sum(all_values) / len(all_values)
            g_var = sum((v - g_mean) ** 2 for v in all_values) / max(1, len(all_values) - 1)
            global_numeric_stats[feature] = (g_mean, max(g_var, 1e-4))
        for label in OUTCOME_LABELS:
            values = [numeric_feature_value(row, feature) for row, y in labeled if y == label]
            values = [v for v in values if v is not None]
            if len(values) >= 2:
                mean = sum(values) / len(values)
                var = sum((v - mean) ** 2 for v in values) / max(1, len(values) - 1)
                numeric_stats[label][feature] = (mean, max(var, 1e-4))

    categorical_counts: Dict[str, Dict[str, Dict[str, float]]] = {label: {} for label in OUTCOME_LABELS}
    category_values: Dict[str, List[str]] = {}
    for feature in CATEGORICAL_FEATURES:
        values = sorted({categorical_feature_value(row, feature) for row, _ in labeled} | {"__missing__"})
        category_values[feature] = values
        for label in OUTCOME_LABELS:
            feature_counts: Dict[str, float] = {}
            for row, y in labeled:
                if y != label:
                    continue
                value = categorical_feature_value(row, feature)
                feature_counts[value] = feature_counts.get(value, 0.0) + 1.0
            categorical_counts[label][feature] = feature_counts

    return DealOutcomeModel(
        priors=priors,
        numeric_stats=numeric_stats,
        categorical_counts=categorical_counts,
        category_values=category_values,
        fit_n=fit_n,
        fit_effective_labels=counts,
        global_numeric_stats=global_numeric_stats,
        alpha=alpha,
    )


def outcome_probability_row(
    row: pd.Series,
    train_rows: Iterable[pd.Series],
    min_fit_events: int,
    default_completed_prob: float,
    default_terminated_prob: float,
    default_withdrawn_prob: float,
) -> Dict[str, Any]:
    default_probs = {
        "completed": default_completed_prob,
        "terminated": default_terminated_prob,
        "withdrawn": default_withdrawn_prob,
    }
    model = fit_deal_outcome_model(train_rows, default_probs=default_probs)
    if model.fit_n < min_fit_events:
        probs = fit_deal_outcome_model([], default_probs=default_probs).priors
        source = "default_probabilities_insufficient_history"
    else:
        probs = model.predict_proba(row)
        source = "rolling_naive_bayes"
    return {
        "deal_completed_probability": probs.get("completed", 0.0),
        "deal_terminated_probability": probs.get("terminated", 0.0),
        "deal_withdrawn_probability": probs.get("withdrawn", 0.0),
        "deal_break_probability": probs.get("terminated", 0.0) + probs.get("withdrawn", 0.0),
        "deal_outcome_model_source": source,
        "deal_outcome_fit_n": model.fit_n,
        "deal_outcome_train_completed": model.fit_effective_labels.get("completed", 0),
        "deal_outcome_train_terminated": model.fit_effective_labels.get("terminated", 0),
        "deal_outcome_train_withdrawn": model.fit_effective_labels.get("withdrawn", 0),
        "actual_deal_outcome": normalize_outcome_label(row) or "",
    }
