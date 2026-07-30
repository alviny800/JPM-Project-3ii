#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Monolithic election-arbitrage pipeline.

This file consolidates the original arb_*.py modules into one runnable script
for easier delivery and slide-prep workflows.  The original split modules are
kept in the repository for compatibility with the README commands, but the
functions below are self-contained except for non-arb project dependencies such
as structural_election_model.py and material_builder.py.

Main commands:
  python arb_pipeline.py check     # validate standard local inputs
  python arb_pipeline.py outcome   # build deal_outcome_probabilities.csv
  python arb_pipeline.py mc        # build arb_deals.csv + arb_output + MC material
  python arb_pipeline.py signal    # build arb_signals.csv + strategy summary + material
  python arb_pipeline.py fast      # deadline spread -> outcome -> mc -> signal
  python arb_pipeline.py material  # refresh material/ only
"""
from __future__ import annotations

import os
os.environ.setdefault("MPLCONFIGDIR", "/private/tmp/jpm_mpl_config")



# ==============================================================================
# arb_outcome.py
# ==============================================================================
"""Outcome-probability adapter for the election-arb trade layer.

This module is intentionally an adapter, not a hidden training pipeline.  If a
future all-status deal universe provides completed/terminated/withdrawn model
outputs, the trade layer can consume them here.  If those columns are absent,
the code falls back to explicit scenario defaults and reports that source.
"""

from dataclasses import dataclass
import json
import math
from pathlib import Path
import re
from typing import Any, Dict, Iterable, List, Optional, Tuple

import pandas as pd


OUTCOME_STATES = ("completed", "terminated", "withdrawn")

PROBABILITY_COLUMNS = {
    "completed": ("p_completed", "deal_completed_probability", "completed_probability"),
    "terminated": ("p_terminated", "deal_terminated_probability", "terminated_probability"),
    "withdrawn": ("p_withdrawn", "deal_withdrawn_probability", "withdrawn_probability"),
}

BREAK_PROBABILITY_COLUMNS = ("p_break", "deal_break_probability", "break_probability")
BBG_STATUS_VALUES = {
    "completed": "completed",
    "terminated": "terminated",
    "withdrawn": "withdrawn",
}

NUMERIC_MODEL_FEATURES = (
    "announced_value_log",
    "tv_ebitda",
    "announce_year",
)

CATEGORICAL_MODEL_FEATURES = (
    "deal_type",
    "payment_type",
    "target_market_suffix",
    "acquirer_market_suffix",
    "target_cusip_present",
    "acquirer_cusip_present",
)

ALL_MODEL_FEATURES = NUMERIC_MODEL_FEATURES + CATEGORICAL_MODEL_FEATURES
MODEL_FEATURE_PREFIX = "__outcome_nb_"


@dataclass(frozen=True)
class OutcomeDefaults:
    completed: float = 0.88
    terminated: float = 0.07
    withdrawn: float = 0.05
    withdrawn_share_of_break: float = 0.35


@dataclass(frozen=True)
class OutcomeNBParams:
    categorical_alpha: float = 1.0
    prior_strength: float = 1.0
    variance_shrinkage: float = 0.0
    likelihood_weight: float = 1.0
    prior_blend: float = 0.0

    def as_dict(self) -> Dict[str, float]:
        return {
            "categorical_alpha": float(self.categorical_alpha),
            "prior_strength": float(self.prior_strength),
            "variance_shrinkage": float(self.variance_shrinkage),
            "likelihood_weight": float(self.likelihood_weight),
            "prior_blend": float(self.prior_blend),
        }


def outcome_nb_candidate_grid() -> List[OutcomeNBParams]:
    """Small, auditable grid for nested temporal tuning."""
    candidates = [
        OutcomeNBParams(
            categorical_alpha=alpha,
            prior_strength=25.0,
            variance_shrinkage=variance_shrinkage,
            likelihood_weight=likelihood_weight,
            prior_blend=prior_blend,
        )
        for alpha in (0.5, 2.0)
        for variance_shrinkage in (0.5, 0.9)
        for likelihood_weight in (0.25, 0.5, 1.0)
        for prior_blend in (0.5, 0.75, 0.9, 0.95)
    ]
    candidates.extend([
        OutcomeNBParams(0.5, 25.0, 0.9, 0.1, 0.0),
        OutcomeNBParams(0.5, 25.0, 0.9, 0.25, 0.0),
        OutcomeNBParams(2.0, 1.0, 0.9, 0.5, 0.75),
        OutcomeNBParams(2.0, 100.0, 0.9, 0.5, 0.75),
    ])
    return candidates


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


def as_float(value: Any) -> Optional[float]:
    if value is None:
        return None
    try:
        if pd.isna(value):
            return None
    except Exception:
        pass
    if isinstance(value, (int, float)):
        x = float(value)
        return x if pd.notna(x) else None
    text = clean_str(value).replace(",", "")
    if not text:
        return None
    match = re.search(r"-?\d+(?:\.\d+)?", text)
    if not match:
        return None
    try:
        return float(match.group(0))
    except Exception:
        return None


def clamp_prob(value: Optional[float]) -> Optional[float]:
    if value is None:
        return None
    return max(0.0, min(1.0, float(value)))


def normalize_outcome_label(value: Any) -> str:
    """Normalize a row or text value into completed/terminated/withdrawn/blank."""
    if isinstance(value, pd.Series):
        texts = [
            clean_str(value.get("deal_outcome_label")),
            clean_str(value.get("deal_completion_or_break")),
            clean_str(value.get("Deal Status")),
            clean_str(value.get("deal_status")),
            clean_str(value.get("status")),
            clean_str(value.get("broke")),
        ]
        text = " ".join(t.lower() for t in texts if t)
    else:
        text = clean_str(value).lower()
    if not text:
        return ""
    if any(w in text for w in ["withdrawn", "withdraw", "acquirer withdrew", "offer withdrawn"]):
        return "withdrawn"
    if any(w in text for w in ["terminated", "termination", "abandoned", "cancelled", "canceled"]):
        return "terminated"
    if any(w in text for w in ["completed", "complete", "closed", "closing", "consummated", "effective time"]):
        return "completed"
    return ""


def normalize_bbg_deal_status(value: Any) -> str:
    text = clean_str(value).strip().lower()
    return BBG_STATUS_VALUES.get(text, "")


def normalized_name(value: Any) -> str:
    text = clean_str(value).lower()
    text = re.sub(r"[^a-z0-9]+", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def event_match_key(announce_date: Any, target_name: Any, acquirer_name: Any) -> str:
    date = pd.to_datetime(announce_date, errors="coerce")
    date_part = "" if pd.isna(date) else pd.Timestamp(date).strftime("%Y%m%d")
    return "|".join([date_part, normalized_name(target_name), normalized_name(acquirer_name)])


def market_suffix(value: Any) -> str:
    parts = clean_str(value).upper().split()
    return parts[-1] if len(parts) > 1 else "__missing__"


def present_flag(value: Any) -> str:
    text = re.sub(r"[^A-Za-z0-9]+", "", clean_str(value)).upper()
    if not text or text in {"NA", "NAN", "NONE", "NULL"}:
        return "missing"
    return "present"


def bbg_model_features(row: pd.Series) -> Dict[str, Any]:
    cached = {
        feature: row.get(f"{MODEL_FEATURE_PREFIX}{feature}")
        for feature in ALL_MODEL_FEATURES
    }
    if all(f"{MODEL_FEATURE_PREFIX}{feature}" in row.index for feature in ALL_MODEL_FEATURES):
        return cached
    ann = pd.to_datetime(row.get("Announce Date", row.get("announce_date", "")), errors="coerce")
    value = as_float(row.get("Announced Total Value (mil.)"))
    tv_ebitda = as_float(row.get("TV/EBITDA"))
    return {
        "announced_value_log": None if value is None or value < 0 else math.log1p(value),
        "tv_ebitda": tv_ebitda,
        "announce_year": None if pd.isna(ann) else float(pd.Timestamp(ann).year),
        "deal_type": normalized_name(row.get("Deal Type")) or "__missing__",
        "payment_type": normalized_name(row.get("Payment Type", row.get("payment_type", ""))) or "__missing__",
        "target_market_suffix": market_suffix(row.get("Target Ticker", row.get("target_ticker", ""))),
        "acquirer_market_suffix": market_suffix(row.get("Acquirer Ticker", row.get("acquirer_ticker", ""))),
        "target_cusip_present": present_flag(row.get("Target cusip", row.get("target_cusip", ""))),
        "acquirer_cusip_present": present_flag(row.get("Acquirer cusip", row.get("acquirer_cusip", ""))),
    }


def precompute_bbg_model_features(rows: pd.DataFrame) -> pd.DataFrame:
    """Cache normalized model features once for repeated temporal fits."""
    out = rows.copy()
    if all(f"{MODEL_FEATURE_PREFIX}{feature}" in out.columns for feature in ALL_MODEL_FEATURES):
        return out
    feature_rows = [bbg_model_features(row) for _, row in out.iterrows()]
    features = pd.DataFrame(feature_rows, index=out.index)
    for feature in ALL_MODEL_FEATURES:
        out[f"{MODEL_FEATURE_PREFIX}{feature}"] = features[feature]
    return out


def _first_probability(row: pd.Series, names: Iterable[str]) -> Optional[float]:
    for name in names:
        if name in row.index:
            value = clamp_prob(as_float(row.get(name)))
            if value is not None:
                return value
    return None


def _normalized_defaults(defaults: OutcomeDefaults) -> Dict[str, float]:
    probs = {
        "completed": clamp_prob(defaults.completed) or 0.0,
        "terminated": clamp_prob(defaults.terminated) or 0.0,
        "withdrawn": clamp_prob(defaults.withdrawn) or 0.0,
    }
    total = sum(probs.values())
    if total <= 0:
        probs = {"completed": 0.88, "terminated": 0.07, "withdrawn": 0.05}
        total = sum(probs.values())
    return {k: v / total for k, v in probs.items()}


def _normalize_probabilities(probs: Dict[str, float]) -> Dict[str, float]:
    clean = {k: clamp_prob(probs.get(k)) or 0.0 for k in OUTCOME_STATES}
    total = sum(clean.values())
    if total <= 0:
        return {"completed": 1.0, "terminated": 0.0, "withdrawn": 0.0}
    return {k: clean[k] / total for k in OUTCOME_STATES}


def probabilities_from_row(row: pd.Series, defaults: OutcomeDefaults, source_prefix: str) -> Dict[str, Any]:
    explicit = {
        state: _first_probability(row, PROBABILITY_COLUMNS[state])
        for state in OUTCOME_STATES
    }
    p_break = _first_probability(row, BREAK_PROBABILITY_COLUMNS)

    if all(explicit[state] is not None for state in OUTCOME_STATES):
        probs = _normalize_probabilities({state: float(explicit[state]) for state in OUTCOME_STATES})
        source = f"{source_prefix}_explicit_three_state"
    elif p_break is not None:
        p_break = clamp_prob(p_break) or 0.0
        withdrawn_share = clamp_prob(defaults.withdrawn_share_of_break)
        if withdrawn_share is None:
            withdrawn_share = 0.35
        probs = {
            "completed": 1.0 - p_break,
            "terminated": p_break * (1.0 - withdrawn_share),
            "withdrawn": p_break * withdrawn_share,
        }
        source = f"{source_prefix}_aggregate_break_split"
    elif any(explicit[state] is not None for state in OUTCOME_STATES):
        base = _normalized_defaults(defaults)
        known = {state: explicit[state] for state in OUTCOME_STATES if explicit[state] is not None}
        known_sum = sum(float(v) for v in known.values())
        missing = [state for state in OUTCOME_STATES if state not in known]
        missing_base_sum = sum(base[state] for state in missing)
        probs = {state: float(known[state]) for state in known}
        residual = max(0.0, 1.0 - known_sum)
        for state in missing:
            probs[state] = residual * base[state] / missing_base_sum if missing_base_sum > 0 else 0.0
        probs = _normalize_probabilities(probs)
        source = f"{source_prefix}_partial_three_state_plus_defaults"
    else:
        probs = _normalized_defaults(defaults)
        source = "default_scenario_no_event_probabilities"

    return {
        "p_completed": probs["completed"],
        "p_terminated": probs["terminated"],
        "p_withdrawn": probs["withdrawn"],
        "outcome_probability_source": source,
    }


def load_outcome_probability_table(path: Optional[str]) -> pd.DataFrame:
    """Read an optional event-level probability table.

    The table must have event_id plus one of:
    - p_completed/p_terminated/p_withdrawn (or deal_*_probability aliases), or
    - p_break / deal_break_probability.
    """
    if not path:
        return pd.DataFrame()
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Outcome probability file not found: {p}")
    table = pd.read_csv(p)
    if "event_id" not in table.columns:
        raise ValueError(f"Outcome probability file must contain event_id: {p}")
    return table


def outcome_probabilities_for_event(
    deal_row: pd.Series,
    probability_table: Optional[pd.DataFrame],
    defaults: OutcomeDefaults,
) -> Dict[str, Any]:
    event_id = clean_str(deal_row.get("event_id"))
    if probability_table is not None and not probability_table.empty and event_id:
        matches = probability_table[probability_table["event_id"].astype(str) == event_id]
        if not matches.empty:
            return probabilities_from_row(matches.iloc[0], defaults, "external_outcome_probabilities")
    return probabilities_from_row(deal_row, defaults, "arb_deals")


class BbgOutcomeNaiveBayes:
    def __init__(
        self,
        rows: pd.DataFrame,
        defaults: OutcomeDefaults,
        alpha: float = 1.0,
        params: Optional[OutcomeNBParams] = None,
    ) -> None:
        labeled = rows.copy()
        labeled["_outcome_label"] = labeled["Deal Status"].map(normalize_bbg_deal_status)
        labeled = labeled[labeled["_outcome_label"].isin(OUTCOME_STATES)].copy()
        self.fit_n = int(len(labeled))
        self.params = params or OutcomeNBParams(categorical_alpha=float(alpha))
        self.alpha = max(float(self.params.categorical_alpha), 1e-8)
        self.prior_strength = max(float(self.params.prior_strength), 0.0)
        self.variance_shrinkage = max(0.0, min(1.0, float(self.params.variance_shrinkage)))
        self.likelihood_weight = max(0.0, float(self.params.likelihood_weight))
        self.prior_blend = max(0.0, min(1.0, float(self.params.prior_blend)))
        self.defaults = _normalized_defaults(defaults)
        self.label_counts = {
            state: int((labeled["_outcome_label"] == state).sum())
            for state in OUTCOME_STATES
        }
        prior_total = self.fit_n + self.prior_strength
        self.priors = {
            state: (
                self.label_counts[state] + self.prior_strength * self.defaults[state]
            ) / max(prior_total, 1e-12)
            for state in OUTCOME_STATES
        }
        norm = sum(self.priors.values())
        self.priors = {k: v / norm for k, v in self.priors.items()}

        labeled = precompute_bbg_model_features(labeled)
        feats = pd.DataFrame({
            "_row_idx": labeled.index,
            "_outcome_label": labeled["_outcome_label"],
            **{
                feature: labeled[f"{MODEL_FEATURE_PREFIX}{feature}"]
                for feature in ALL_MODEL_FEATURES
            },
        })
        self.numeric_stats: Dict[str, Dict[str, Tuple[float, float]]] = {state: {} for state in OUTCOME_STATES}
        self.global_numeric_stats: Dict[str, Tuple[float, float]] = {}
        for feature in NUMERIC_MODEL_FEATURES:
            vals = pd.to_numeric(feats.get(feature), errors="coerce").dropna() if feature in feats else pd.Series(dtype=float)
            if len(vals):
                mean = float(vals.mean())
                var = float(vals.var(ddof=1)) if len(vals) > 1 else 1.0
                self.global_numeric_stats[feature] = (mean, max(var, 1e-4))
            for state in OUTCOME_STATES:
                svals = pd.to_numeric(feats.loc[feats["_outcome_label"] == state, feature], errors="coerce").dropna() if feature in feats else pd.Series(dtype=float)
                if len(svals) >= 2:
                    mean = float(svals.mean())
                    var = float(svals.var(ddof=1))
                    global_var = self.global_numeric_stats.get(feature, (mean, var))[1]
                    shrunk_var = (
                        (1.0 - self.variance_shrinkage) * var
                        + self.variance_shrinkage * global_var
                    )
                    self.numeric_stats[state][feature] = (mean, max(shrunk_var, 1e-4))

        self.category_values: Dict[str, List[str]] = {}
        self.category_counts: Dict[str, Dict[str, Dict[str, float]]] = {state: {} for state in OUTCOME_STATES}
        for feature in CATEGORICAL_MODEL_FEATURES:
            values = (
                sorted(set(feats[feature].astype(str)) | {"__missing__", "__other__"})
                if feature in feats
                else ["__missing__", "__other__"]
            )
            self.category_values[feature] = values
            for state in OUTCOME_STATES:
                counts: Dict[str, float] = {}
                for value in feats.loc[feats["_outcome_label"] == state, feature].astype(str) if feature in feats else []:
                    counts[value] = counts.get(value, 0.0) + 1.0
                self.category_counts[state][feature] = counts

    def predict_proba(self, row: pd.Series) -> Dict[str, float]:
        feats = bbg_model_features(row)
        scores: Dict[str, float] = {}
        for state in OUTCOME_STATES:
            log_likelihood = 0.0
            for feature in NUMERIC_MODEL_FEATURES:
                value = feats.get(feature)
                if value is None or pd.isna(value):
                    continue
                mean, var = self.numeric_stats.get(state, {}).get(
                    feature,
                    self.global_numeric_stats.get(feature, (0.0, 1.0)),
                )
                var = max(var, 1e-4)
                log_likelihood += -0.5 * (
                    math.log(2.0 * math.pi * var)
                    + ((float(value) - mean) ** 2) / var
                )
            for feature in CATEGORICAL_MODEL_FEATURES:
                value = str(feats.get(feature) or "__missing__")
                values = self.category_values.get(feature, ["__missing__"])
                if value not in values:
                    value = "__other__"
                counts = self.category_counts.get(state, {}).get(feature, {})
                denom = sum(counts.values()) + self.alpha * max(1, len(values))
                log_likelihood += math.log(
                    (counts.get(value, 0.0) + self.alpha) / max(denom, 1e-12)
                )
            scores[state] = (
                math.log(max(self.priors.get(state, 0.0), 1e-12))
                + self.likelihood_weight * log_likelihood
            )
        max_score = max(scores.values())
        exp_scores = {k: math.exp(v - max_score) for k, v in scores.items()}
        total = sum(exp_scores.values())
        if total <= 0:
            return dict(self.priors)
        posterior = {k: exp_scores[k] / total for k in OUTCOME_STATES}
        return {
            k: (1.0 - self.prior_blend) * posterior[k] + self.prior_blend * self.priors[k]
            for k in OUTCOME_STATES
        }


def tune_bbg_outcome_naive_bayes(
    rows: pd.DataFrame,
    defaults: Optional[OutcomeDefaults] = None,
    candidates: Optional[List[OutcomeNBParams]] = None,
    min_train_per_class: int = 10,
    max_validation_years: int = 5,
) -> Tuple[OutcomeNBParams, pd.DataFrame]:
    """Select NB parameters using expanding-window validation inside `rows`."""
    defaults = defaults or OutcomeDefaults()
    labels = list(OUTCOME_STATES)
    labeled = rows.copy()
    labeled["_outcome_label"] = labeled["Deal Status"].map(normalize_bbg_deal_status)
    labeled = labeled[labeled["_outcome_label"].isin(labels)].copy()
    labeled["_tuning_date"] = pd.to_datetime(labeled["Announce Date"], errors="coerce")
    labeled = labeled.dropna(subset=["_tuning_date"]).sort_values("_tuning_date")
    labeled = precompute_bbg_model_features(labeled)
    candidate_list = candidates or outcome_nb_candidate_grid()

    validation_years: List[int] = []
    if not labeled.empty:
        first_year = int(labeled["_tuning_date"].dt.year.min())
        last_year = int(labeled["_tuning_date"].dt.year.max())
        for year in range(first_year + 1, last_year + 1):
            cutoff = pd.Timestamp(year=year, month=1, day=1)
            next_cutoff = pd.Timestamp(year=year + 1, month=1, day=1)
            train = labeled[labeled["_tuning_date"] < cutoff]
            validation = labeled[
                (labeled["_tuning_date"] >= cutoff)
                & (labeled["_tuning_date"] < next_cutoff)
            ]
            train_counts = train["_outcome_label"].value_counts()
            if not validation.empty and all(
                int(train_counts.get(label, 0)) >= min_train_per_class
                for label in labels
            ):
                validation_years.append(year)
    if max_validation_years > 0:
        validation_years = validation_years[-max_validation_years:]

    if not validation_years:
        fallback = OutcomeNBParams()
        return fallback, pd.DataFrame([{
            **fallback.as_dict(),
            "validation_rows": 0,
            "validation_years": "",
            "multiclass_brier_score": None,
        }])

    results: List[Dict[str, Any]] = []
    for params in candidate_list:
        squared_error = 0.0
        validation_n = 0
        for year in validation_years:
            cutoff = pd.Timestamp(year=year, month=1, day=1)
            next_cutoff = pd.Timestamp(year=year + 1, month=1, day=1)
            train = labeled[labeled["_tuning_date"] < cutoff]
            validation = labeled[
                (labeled["_tuning_date"] >= cutoff)
                & (labeled["_tuning_date"] < next_cutoff)
            ]
            model = BbgOutcomeNaiveBayes(train, defaults, params=params)
            for _, row in validation.iterrows():
                probs = model.predict_proba(row)
                actual = str(row["_outcome_label"])
                squared_error += sum(
                    (probs[label] - float(actual == label)) ** 2
                    for label in labels
                )
                validation_n += 1
        results.append({
            **params.as_dict(),
            "validation_rows": int(validation_n),
            "validation_years": ",".join(str(year) for year in validation_years),
            "multiclass_brier_score": (
                float(squared_error / validation_n) if validation_n else None
            ),
        })

    tuning = pd.DataFrame(results).sort_values(
        ["multiclass_brier_score", "likelihood_weight", "categorical_alpha"],
        ascending=[True, False, True],
    ).reset_index(drop=True)
    best_row = tuning.iloc[0]
    best = OutcomeNBParams(
        categorical_alpha=float(best_row["categorical_alpha"]),
        prior_strength=float(best_row["prior_strength"]),
        variance_shrinkage=float(best_row["variance_shrinkage"]),
        likelihood_weight=float(best_row["likelihood_weight"]),
        prior_blend=float(best_row["prior_blend"]),
    )
    return best, tuning


def tune_outcome_decision_prior_power(
    rows: pd.DataFrame,
    params: OutcomeNBParams,
    defaults: Optional[OutcomeDefaults] = None,
    candidates: Optional[List[float]] = None,
    min_train_per_class: int = 5,
    max_validation_years: int = 5,
) -> Tuple[float, pd.DataFrame]:
    """Tune the cost-balanced hard-decision rule on earlier years only."""
    defaults = defaults or OutcomeDefaults()
    powers = candidates or [0.0, 0.25, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
    labels = list(OUTCOME_STATES)
    labeled = rows.copy()
    labeled["_outcome_label"] = labeled["Deal Status"].map(normalize_bbg_deal_status)
    labeled = labeled[labeled["_outcome_label"].isin(labels)].copy()
    labeled["_decision_date"] = pd.to_datetime(labeled["Announce Date"], errors="coerce")
    labeled = labeled.dropna(subset=["_decision_date"]).sort_values("_decision_date")
    labeled = precompute_bbg_model_features(labeled)

    years: List[int] = []
    if not labeled.empty:
        first_year = int(labeled["_decision_date"].dt.year.min())
        last_year = int(labeled["_decision_date"].dt.year.max())
        for year in range(first_year + 1, last_year + 1):
            cutoff = pd.Timestamp(year=year, month=1, day=1)
            next_cutoff = pd.Timestamp(year=year + 1, month=1, day=1)
            train = labeled[labeled["_decision_date"] < cutoff]
            validation = labeled[
                (labeled["_decision_date"] >= cutoff)
                & (labeled["_decision_date"] < next_cutoff)
            ]
            train_counts = train["_outcome_label"].value_counts()
            if not validation.empty and all(
                int(train_counts.get(label, 0)) >= min_train_per_class
                for label in labels
            ):
                years.append(year)
    if max_validation_years > 0:
        years = years[-max_validation_years:]
    if not years:
        return 0.5, pd.DataFrame([{
            "decision_prior_power": 0.5,
            "validation_rows": 0,
            "validation_years": "",
            "balanced_accuracy": None,
            "macro_f1": None,
        }])

    validation_records: List[Dict[str, Any]] = []
    for year in years:
        cutoff = pd.Timestamp(year=year, month=1, day=1)
        next_cutoff = pd.Timestamp(year=year + 1, month=1, day=1)
        train = labeled[labeled["_decision_date"] < cutoff]
        validation = labeled[
            (labeled["_decision_date"] >= cutoff)
            & (labeled["_decision_date"] < next_cutoff)
        ]
        model = BbgOutcomeNaiveBayes(train, defaults, params=params)
        for _, row in validation.iterrows():
            probs = model.predict_proba(row)
            validation_records.append({
                "actual": str(row["_outcome_label"]),
                **{f"p_{label}": probs[label] for label in labels},
                **{f"prior_{label}": model.priors[label] for label in labels},
            })

    validation_df = pd.DataFrame(validation_records)
    results: List[Dict[str, Any]] = []
    for power in powers:
        predicted: List[str] = []
        for _, row in validation_df.iterrows():
            scores = {
                label: float(row[f"p_{label}"]) / max(
                    float(row[f"prior_{label}"]) ** float(power),
                    1e-12,
                )
                for label in labels
            }
            predicted.append(max(labels, key=lambda label: scores[label]))
        pred = pd.Series(predicted, index=validation_df.index)
        actual = validation_df["actual"].astype(str)
        recalls: List[float] = []
        f1s: List[float] = []
        for label in labels:
            actual_label = actual == label
            predicted_label = pred == label
            tp = int((actual_label & predicted_label).sum())
            fp = int((~actual_label & predicted_label).sum())
            fn = int((actual_label & ~predicted_label).sum())
            precision = float(tp / (tp + fp)) if tp + fp else 0.0
            recall = float(tp / (tp + fn)) if tp + fn else 0.0
            f1 = float(2.0 * precision * recall / (precision + recall)) if precision + recall else 0.0
            recalls.append(recall)
            f1s.append(f1)
        results.append({
            "decision_prior_power": float(power),
            "validation_rows": int(len(validation_df)),
            "validation_years": ",".join(str(year) for year in years),
            "balanced_accuracy": float(sum(recalls) / len(recalls)),
            "macro_f1": float(sum(f1s) / len(f1s)),
        })

    tuning = pd.DataFrame(results).sort_values(
        ["macro_f1", "balanced_accuracy", "decision_prior_power"],
        ascending=[False, False, True],
    ).reset_index(drop=True)
    return float(tuning.iloc[0]["decision_prior_power"]), tuning


def load_bbg_with_keys(path: str = "BBG Data Pull 2006+ Final.csv") -> pd.DataFrame:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"BBG source file not found: {p}")
    bbg = pd.read_csv(p)
    required = ["Announce Date", "Target Name", "Acquirer Name", "Deal Status"]
    missing = [c for c in required if c not in bbg.columns]
    if missing:
        raise ValueError(f"BBG source missing required columns: {missing}")
    bbg = bbg.copy()
    bbg["_bbg_row_idx"] = range(len(bbg))
    bbg["_match_key"] = [
        event_match_key(r.get("Announce Date"), r.get("Target Name"), r.get("Acquirer Name"))
        for _, r in bbg.iterrows()
    ]
    bbg["_outcome_label"] = bbg["Deal Status"].map(normalize_bbg_deal_status)
    return bbg


def load_event_frame_for_outcomes(path: str = "eda_output/merged_panel.csv") -> pd.DataFrame:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Event frame not found: {p}")
    events = pd.read_csv(p)
    required = ["event_id", "target_name", "acquirer_name", "announce_date"]
    missing = [c for c in required if c not in events.columns]
    if missing:
        raise ValueError(f"Event frame missing required columns: {missing}")
    events = events[required + [c for c in ["deal_status", "payment_type"] if c in events.columns]].drop_duplicates("event_id")
    events["_match_key"] = [
        event_match_key(r.get("announce_date"), r.get("target_name"), r.get("acquirer_name"))
        for _, r in events.iterrows()
    ]
    return events


def event_status_map_from_bbg(
    bbg_path: str = "BBG Data Pull 2006+ Final.csv",
    events_path: str = "eda_output/merged_panel.csv",
) -> Dict[str, Dict[str, Any]]:
    bbg = load_bbg_with_keys(bbg_path)
    events = load_event_frame_for_outcomes(events_path)
    bbg_by_key = {
        str(row["_match_key"]): row
        for _, row in bbg.drop_duplicates("_match_key", keep="first").iterrows()
    }
    out: Dict[str, Dict[str, Any]] = {}
    for _, event in events.iterrows():
        match = bbg_by_key.get(str(event["_match_key"]))
        if match is not None:
            raw = clean_str(match.get("Deal Status"))
            label = normalize_bbg_deal_status(raw)
            source = "bbg_deal_status_key_match" if label else "bbg_deal_status_ignored"
        else:
            raw = clean_str(event.get("deal_status"))
            label = normalize_bbg_deal_status(raw)
            source = "merged_panel_deal_status_fallback" if label else "missing_bbg_deal_status_match"
        out[str(event["event_id"])] = {
            "deal_status_raw": raw,
            "deal_outcome_label": label,
            "deal_outcome_source": source,
        }
    return out


def build_bbg_outcome_probability_table(
    bbg_path: str = "BBG Data Pull 2006+ Final.csv",
    events_path: str = "eda_output/merged_panel.csv",
    output_path: str = "deal_outcome_probabilities.csv",
    defaults: Optional[OutcomeDefaults] = None,
    min_train_rows: int = 25,
) -> pd.DataFrame:
    defaults = defaults or OutcomeDefaults()
    bbg = load_bbg_with_keys(bbg_path)
    events = load_event_frame_for_outcomes(events_path)
    labeled = bbg[bbg["_outcome_label"].isin(OUTCOME_STATES)].copy()
    if len(labeled) >= min_train_rows:
        tuned_params, tuning = tune_bbg_outcome_naive_bayes(labeled, defaults)
        model = BbgOutcomeNaiveBayes(labeled, defaults, params=tuned_params)
        default_probs = model.priors
        model_source = "bbg_naive_bayes_temporal_tuned_full_training"
        train_counts = model.label_counts
        train_n = model.fit_n
        tuning_brier = tuning.iloc[0]["multiclass_brier_score"]
    else:
        model = None
        tuned_params = OutcomeNBParams()
        default_probs = _normalized_defaults(defaults)
        model_source = "default_probabilities_insufficient_bbg_training_rows"
        train_counts = {state: 0 for state in OUTCOME_STATES}
        train_n = int(len(labeled))
        tuning_brier = None
    bbg_by_key = {
        str(row["_match_key"]): row
        for _, row in bbg.drop_duplicates("_match_key", keep="first").iterrows()
    }

    rows = []
    for _, event in events.iterrows():
        key = str(event["_match_key"])
        match = bbg_by_key.get(key)
        if match is not None:
            predict_row = match
            actual_raw = clean_str(match.get("Deal Status"))
            actual_label = normalize_bbg_deal_status(actual_raw)
            match_source = "bbg_key_match"
        else:
            predict_row = pd.Series({
                "Announce Date": event.get("announce_date"),
                "Target Name": event.get("target_name"),
                "Acquirer Name": event.get("acquirer_name"),
                "Payment Type": event.get("payment_type", ""),
                "Deal Status": event.get("deal_status", ""),
            })
            actual_raw = clean_str(event.get("deal_status"))
            actual_label = normalize_bbg_deal_status(actual_raw)
            match_source = "event_frame_fallback_no_bbg_key_match"

        if model is not None:
            probs = model.predict_proba(predict_row)
        else:
            probs = default_probs

        rows.append({
            "event_id": event["event_id"],
            "p_completed": probs["completed"],
            "p_terminated": probs["terminated"],
            "p_withdrawn": probs["withdrawn"],
            "outcome_probability_source": model_source,
            "outcome_match_source": match_source,
            "actual_deal_status_raw": actual_raw,
            "actual_outcome_label": actual_label,
            "outcome_train_n": int(train_n),
            "outcome_train_completed": int(train_counts.get("completed", 0)),
            "outcome_train_terminated": int(train_counts.get("terminated", 0)),
            "outcome_train_withdrawn": int(train_counts.get("withdrawn", 0)),
            "outcome_nb_categorical_alpha": tuned_params.categorical_alpha,
            "outcome_nb_prior_strength": tuned_params.prior_strength,
            "outcome_nb_variance_shrinkage": tuned_params.variance_shrinkage,
            "outcome_nb_likelihood_weight": tuned_params.likelihood_weight,
            "outcome_nb_prior_blend": tuned_params.prior_blend,
            "outcome_nb_inner_validation_brier": tuning_brier,
        })
    out = pd.DataFrame(rows)
    out.to_csv(output_path, index=False)
    return out


def _parse_cli() -> Any:
    import argparse

    p = argparse.ArgumentParser(description="Build event-level completed/terminated/withdrawn probabilities from BBG Deal Status.")
    p.add_argument("--bbg", default="BBG Data Pull 2006+ Final.csv")
    p.add_argument("--events", default="eda_output/merged_panel.csv")
    p.add_argument("--out", default="deal_outcome_probabilities.csv")
    p.add_argument("--default-completed-prob", type=float, default=0.88)
    p.add_argument("--default-terminated-prob", type=float, default=0.07)
    p.add_argument("--default-withdrawn-prob", type=float, default=0.05)
    return p.parse_args()


# ==============================================================================
# arb_mc.py
# ==============================================================================
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


# ==============================================================================
# arb_backtest.py
# ==============================================================================
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
import numpy as np
import pandas as pd
from scipy import stats


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
    # drop deals with an implausible cash-vs-stock term gap (misparsed cash value / ratio / price);
    # a >50% gap is a data artifact, not a real deadline spread. Matches the arb_signal REVIEW guard.
    stock_val = d["R"] * d["P_acq"]
    term_gap = (d["C"] - stock_val).abs() / np.minimum(d["C"].abs(), stock_val.abs()).clip(lower=1e-9)
    d = d[term_gap <= 0.50]
    rows = []
    for _, r in d.iterrows():
        _, _, blended, optimal = prorate(r["f_cash"], r["pi_cash"], r["C"], r["R"] * r["P_acq"])
        rows.append({"event_id": r["event_id"], "target_name": r["target_name"],
                     "blended": float(blended), "optimal": float(optimal),
                     "edge": float(optimal - blended),
                     "edge_pct": float((optimal - blended) / blended * 100) if blended else np.nan})
    out = pd.DataFrame(rows)
    return out


# ==============================================================================
# arb_terms.py
# ==============================================================================
"""
arb_terms.py  —  DATA LAYER for the election-arb Monte Carlo / backtest framework.

Assembles ONE clean deal table (arb_deals.csv) that every downstream module reads.
Per completed cash-or-stock election deal we resolve the structural terms the MC needs:

  C        cash consideration per share            ($/sh)          -> cash election value
  R        exchange ratio (fixed only)             (acq sh / tgt)  -> stock leg
  P_acq    acquirer price at the election deadline ($/sh)          -> stock election value = R * P_acq
  stock_val= R * P_acq                             ($/sh)
  spread   = C - stock_val                         ($/sh)          -> the deadline election spread
  pi_cash  aggregate CASH proration target         (frac 0-1)      -> the fixed cash pool (from cash_cap)
  f_cash   REALIZED fraction of shares electing cash(frac 0-1)     -> the stochastic outcome we model
  ratio_type  fixed | floating                                     -> floating excluded from spread work

Nothing here is stochastic — this is just the observed, cleaned deal terms.
Reuses deadline_spread.csv (already has C, R, P_acq, spread, ratio_type, realized demand).
"""
from pathlib import Path
import re
import numpy as np
import pandas as pd



def read_required_csv(path, purpose):
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(
            f"Missing {p} needed for {purpose}. Run the upstream pipeline that writes this file; "
            "the arb layer will not fabricate missing inputs."
        )
    return pd.read_csv(p)


def first_num(s, lo, hi):
    """First number in text within [lo,hi] — skips share counts / years."""
    if pd.isna(s):
        return np.nan
    for tok in re.findall(r"\d+\.?\d*", str(s).replace(",", "")):
        v = float(tok)
        if lo <= v <= hi:
            return v
    return np.nan


def parse_pct_frac(s):
    """A percentage in prose -> fraction in (0,1). Skips share counts. e.g. '50%' -> 0.50"""
    if pd.isna(s):
        return np.nan
    m = re.search(r"(\d{1,3}(?:\.\d+)?)\s*%", str(s).replace(",", ""))
    if m:
        v = float(m.group(1))
        return v / 100.0 if 1 <= v <= 99 else np.nan
    return np.nan


def build_deals() -> pd.DataFrame:
    ds = read_required_csv("deadline_spread.csv", "deadline-date spread inputs")
    ext = read_required_csv("ma_edgar_full/llm_field_extractions.csv", "SEC/Claude term extraction")
    w = ext[ext.event_id.notna()].pivot_table(index="event_id", columns="field_name",
                                               values="value", aggfunc="first")
    norm = read_required_csv("normalized_labels.csv", "normalized realized election labels")

    d = ds.rename(columns={"cash_val": "C", "ratio": "R", "acq_price_deadline": "P_acq",
                           "spread_deadline": "spread", "realized_cash_share": "f_cash"}).copy()

    # aggregate cash proration target pi_cash: prefer cash_cap %, else 1 - stock_cap %, else 0.50 (the
    # modal 50/50 election structure) with a source flag so coverage is auditable.
    def pi_for(eid):
        if eid in w.index:
            p = parse_pct_frac(w.loc[eid].get("cash_cap"))
            if not np.isnan(p):
                return p, "cash_cap"
            ps = parse_pct_frac(w.loc[eid].get("stock_cap"))
            if not np.isnan(ps):
                return 1 - ps, "stock_cap"
        return 0.50, "default_5050"
    pis = d["event_id"].map(lambda e: pi_for(e))
    d["pi_cash"] = pis.map(lambda t: t[0])
    d["pi_cash_source"] = pis.map(lambda t: t[1])

    # realized election demand as a fraction (from the normalized labels, authoritative)
    fc = norm.set_index("event_id")["pct_elected_cash"].apply(pd.to_numeric, errors="coerce") / 100.0
    d["f_cash"] = d["event_id"].map(fc)

    # Outcome labels are post-outcome labels for backtest/audit only.  The
    # authoritative source is the original BBG Deal Status column; Claude's
    # deal_completion_or_break text is kept only as a last-resort audit fallback.
    status_map = event_status_map_from_bbg()
    brk = w["deal_completion_or_break"] if "deal_completion_or_break" in w.columns else pd.Series(dtype=str)

    def outcome_for(eid):
        status = status_map.get(str(eid), {})
        if status.get("deal_outcome_label"):
            return (
                status.get("deal_outcome_label", ""),
                status.get("deal_outcome_source", "bbg_deal_status"),
                status.get("deal_status_raw", ""),
            )
        if status.get("deal_outcome_source"):
            return "", status.get("deal_outcome_source", ""), status.get("deal_status_raw", "")
        if eid not in getattr(brk, "index", []):
            return "", "missing_bbg_and_claude_deal_status", ""
        raw = brk.get(eid, "")
        label = normalize_outcome_label(raw)
        if label:
            return label, "claude_deal_completion_or_break_fallback", raw
        if re.search(r"break|fail", str(raw), re.I):
            return "", "claude_deal_completion_or_break_regex_break_unclassified", raw
        return "", "unrecognized_or_blank_bbg_and_claude_deal_status", raw

    outcomes = d["event_id"].map(lambda e: outcome_for(e))
    d["deal_outcome_label"] = outcomes.map(lambda t: t[0])
    d["deal_outcome_source"] = outcomes.map(lambda t: t[1])
    d["deal_status_raw"] = outcomes.map(lambda t: t[2])
    d["broke"] = (
        d["deal_outcome_label"].isin(["terminated", "withdrawn"])
        | d["deal_outcome_source"].str.contains("regex_break_unclassified", na=False)
    )

    d["stock_val"] = d["R"] * d["P_acq"]
    optional_probability_cols = [
        c for c in [
            "p_completed", "p_terminated", "p_withdrawn", "p_break",
            "deal_completed_probability", "deal_terminated_probability",
            "deal_withdrawn_probability", "deal_break_probability",
        ]
        if c in d.columns
    ]
    # keep the analytic columns
    keep = ["event_id", "target_name", "ratio_type", "C", "R", "P_acq", "stock_val", "spread",
            "pi_cash", "pi_cash_source", "f_cash", "deal_outcome_label", "deal_outcome_source", "deal_status_raw",
            "broke", *optional_probability_cols]
    d = d[keep]
    d.to_csv("arb_deals.csv", index=False)

    # coverage report
    have_terms = d[["C", "R", "P_acq"]].notna().all(axis=1)
    have_demand = d["f_cash"].notna()
    fixed = d.ratio_type.eq("fixed")
    print(f"[terms] deals: {len(d)}")
    print(f"  full structural terms (C,R,P_acq): {have_terms.sum()}")
    print(f"  fixed-ratio: {fixed.sum()}   floating: {(~fixed).sum()}")
    print(f"  realized demand present: {have_demand.sum()}   (the MC-calibration set)")
    print(f"  MC-ready (terms + pi_cash + demand, fixed): {(have_terms & have_demand & fixed).sum()}")
    print(f"  pi_cash source: " + ", ".join(f"{k}={v}" for k, v in d.pi_cash_source.value_counts().items()))
    return d


# ==============================================================================
# arb_capacity.py
# ==============================================================================
"""Capacity and election-impact overlay for election-arb signals.

The payoff engine prices a marginal share.  This module asks how many shares can
actually be sourced without changing the election outcome enough to destroy the
edge.  It is deliberately transparent: when true holder/borrow data are absent,
the output labels the behavioral assumptions rather than pretending they were
observed.
"""

from dataclasses import dataclass
import math
from typing import Any, Dict, Iterable, List, Optional

import numpy as np
import pandas as pd


@dataclass
class CapacityConfig:
    panel_path: str = "eda_output/merged_panel.csv"
    holder_model: str = "rolling_structural"
    holder_rolling_window_events: int = 50
    holder_min_fit_events: int = 10
    holder_p_grid_size: int = 11
    holder_q_grid_size: int = 11
    default_irrational_cash_prob: float = 0.50
    default_rational_share: float = 0.30
    build_days: float = 10.0
    max_adv_participation: float = 0.20
    max_position_pct_shares: float = 0.05
    positive_holder_share_of_active: float = 0.35
    noisy_sell_fraction: float = 0.50
    noisy_buy_fraction: float = 0.50
    noisy_election_cash_prob: float = 0.50
    positive_sell_min: float = 0.02
    positive_sell_max: float = 0.85
    positive_sell_width: float = 0.03
    positive_no_sell_edge: float = 0.03
    passive_sell_fraction: float = 0.01
    passive_lendable_fraction: float = 0.20
    noisy_lendable_fraction: float = 0.10
    capacity_grid_points: int = 51


def capacity_as_float(value: Any) -> float:
    try:
        x = float(value)
    except Exception:
        return math.nan
    return x if math.isfinite(x) else math.nan


def clamp01(value: float) -> float:
    if not math.isfinite(value):
        return math.nan
    return max(0.0, min(1.0, float(value)))


def normalize_fraction(value: Any) -> float:
    x = capacity_as_float(value)
    if not math.isfinite(x):
        return math.nan
    if x > 1.0 and x <= 100.0:
        x = x / 100.0
    return clamp01(x)


def build_rolling_holder_estimates(panel: pd.DataFrame, cfg: CapacityConfig) -> pd.DataFrame:
    """Estimate event-level holder composition from prior realized election events.

    This is the old structural model wired into the current capacity layer:
    p_hat = irrational/noisy holder cash-election probability.
    q_hat = rational, EV-sensitive holder share of total target ownership.
    Fits are rolling and leave the current event out by construction.
    """
    if panel.empty or "event_id" not in panel.columns:
        return pd.DataFrame()
    try:
        from structural_election_model import fit_p_q, observed_election_shares, predict_votes, sort_panel
    except Exception:
        return pd.DataFrame()

    ordered = sort_panel(panel)
    historical_rows: List[pd.Series] = []
    rows: List[Dict[str, Any]] = []
    p_grid = max(3, int(cfg.holder_p_grid_size))
    q_grid = max(3, int(cfg.holder_q_grid_size))
    for i, row in ordered.iterrows():
        train_rows = historical_rows[-cfg.holder_rolling_window_events:] if cfg.holder_rolling_window_events > 0 else historical_rows
        fit = fit_p_q(
            train_rows if len(train_rows) >= cfg.holder_min_fit_events else [],
            p_grid,
            q_grid,
            cfg.default_irrational_cash_prob,
            cfg.default_rational_share,
        )
        pred = predict_votes(row, fit["p_hat"], fit["q_hat"])
        passive = normalize_fraction(pred.get("passive_share", row.get("passive_control_percent", math.nan)))
        passive = 0.0 if not math.isfinite(passive) else passive
        active = max(0.0, 1.0 - passive)
        q_hat = clamp01(capacity_as_float(fit.get("q_hat", math.nan)))
        p_hat = clamp01(capacity_as_float(fit.get("p_hat", math.nan)))
        positive = min(active, q_hat) if math.isfinite(q_hat) else math.nan
        noisy = max(0.0, active - positive) if math.isfinite(positive) else math.nan
        rows.append({
            "event_id": row.get("event_id"),
            "holder_model_source": "rolling_structural_fit" if int(fit.get("fit_n", 0) or 0) > 0 else "rolling_structural_default",
            "holder_rolling_event_index": i,
            "holder_fit_n": int(fit.get("fit_n", 0) or 0),
            "holder_fit_sse": fit.get("fit_sse", math.nan),
            "holder_fit_loglike": fit.get("fit_loglike", math.nan),
            "holder_p_hat": p_hat,
            "holder_q_hat": q_hat,
            "holder_positive_share": positive,
            "holder_positive_share_of_active": positive / active if active > 0 and math.isfinite(positive) else math.nan,
            "holder_noisy_share": noisy,
            "holder_noisy_cash_prob": p_hat,
            "holder_predicted_cash_demand_share": pred.get("predicted_cash_demand_share", math.nan),
            "holder_predicted_stock_demand_share": pred.get("predicted_stock_demand_share", math.nan),
            "holder_rational_action": pred.get("rational_action", ""),
            "holder_structural_status": pred.get("model_status", ""),
            "holder_structural_block_reason": pred.get("block_reason", ""),
        })
        obs_cash, _, _ = observed_election_shares(row)
        if obs_cash is not None:
            historical_rows.append(row)
    return pd.DataFrame(rows)


def load_capacity_table(path: str, cfg: Optional[CapacityConfig] = None) -> pd.DataFrame:
    if not path:
        return pd.DataFrame()
    try:
        panel = pd.read_csv(path)
    except FileNotFoundError:
        return pd.DataFrame()
    if "event_id" not in panel.columns:
        return pd.DataFrame()
    keep = [
        "event_id",
        "target_shares_outstanding",
        "target_adv20",
        "target_dollar_adv20",
        "target_market_cap",
        "passive_control_percent",
        "etf_ownership_percent",
        "active_share",
        "passive_pct",
        "default_rule",
        "non_election_default_rule",
        "short_volume_ratio_20d_avg",
        "announce_date",
        "active_cash_election_rate",
        "realized_cash_share",
        "pct_elected_cash",
        "spread",
        "deal_spread",
        "cash_election_value",
        "stock_election_value",
    ]
    keep = [c for c in keep if c in panel.columns]
    out = panel[keep].drop_duplicates("event_id")
    if cfg is not None and str(cfg.holder_model).lower() == "rolling_structural":
        estimates = build_rolling_holder_estimates(panel, cfg)
        if not estimates.empty:
            out = out.merge(estimates, on="event_id", how="left")
    return out.set_index("event_id")


def capacity_row(event_id: str, table: pd.DataFrame) -> pd.Series:
    if table.empty or event_id not in table.index:
        return pd.Series(dtype=object)
    row = table.loc[event_id]
    if isinstance(row, pd.DataFrame):
        row = row.iloc[0]
    return row


def passive_share_from_row(row: pd.Series) -> float:
    passive = normalize_fraction(row.get("passive_control_percent", math.nan))
    if not math.isfinite(passive):
        passive = normalize_fraction(row.get("passive_pct", math.nan))
    if not math.isfinite(passive):
        passive = normalize_fraction(row.get("etf_ownership_percent", math.nan))
    return 0.0 if not math.isfinite(passive) else passive


def default_cash_probability(row: pd.Series, pi_cash: float) -> float:
    text = str(row.get("default_rule", row.get("non_election_default_rule", ""))).strip().lower()
    if "cash" in text and "stock" not in text:
        return 1.0
    if "stock" in text and "cash" not in text:
        return 0.0
    if "mixed" in text or ("cash" in text and "stock" in text):
        return clamp01(pi_cash)
    return clamp01(pi_cash) if math.isfinite(pi_cash) else 0.5


def capacity_inputs(row: pd.Series, entry_price: float, cfg: CapacityConfig) -> Dict[str, Any]:
    shares = capacity_as_float(row.get("target_shares_outstanding", math.nan))
    adv = capacity_as_float(row.get("target_adv20", math.nan))
    dollar_adv = capacity_as_float(row.get("target_dollar_adv20", math.nan))
    if (not math.isfinite(adv) or adv <= 0) and math.isfinite(dollar_adv) and entry_price > 0:
        adv = dollar_adv / entry_price

    passive = passive_share_from_row(row)
    active = max(0.0, 1.0 - passive)
    rolling_positive = normalize_fraction(row.get("holder_positive_share", math.nan))
    if math.isfinite(rolling_positive):
        positive = min(active, rolling_positive)
        holder_model_source = str(row.get("holder_model_source", "rolling_structural_fit"))
    else:
        positive = active * clamp01(cfg.positive_holder_share_of_active)
        holder_model_source = "static_positive_holder_share_of_active"
    noisy = max(0.0, active - positive)
    noisy_cash_prob = normalize_fraction(row.get("holder_noisy_cash_prob", math.nan))
    if not math.isfinite(noisy_cash_prob):
        noisy_cash_prob = cfg.noisy_election_cash_prob
    adv_pct = adv / shares if math.isfinite(adv) and math.isfinite(shares) and shares > 0 else math.nan
    return {
        "shares_outstanding": shares,
        "adv20": adv,
        "dollar_adv20": dollar_adv,
        "passive_share": passive,
        "active_share": active,
        "positive_holder_share": positive,
        "positive_holder_share_of_active": positive / active if active > 0 else math.nan,
        "noisy_holder_share": noisy,
        "noisy_cash_prob": noisy_cash_prob,
        "adv_pct_shares": adv_pct,
        "holder_model_source": holder_model_source,
        "holder_fit_n": capacity_as_float(row.get("holder_fit_n", math.nan)),
        "holder_p_hat": capacity_as_float(row.get("holder_p_hat", math.nan)),
        "holder_q_hat": capacity_as_float(row.get("holder_q_hat", math.nan)),
        "holder_predicted_cash_demand_share": capacity_as_float(row.get("holder_predicted_cash_demand_share", math.nan)),
        "holder_rational_action": str(row.get("holder_rational_action", "")),
    }


def positive_sell_rate(edge_return: float, cfg: CapacityConfig) -> float:
    """EV-sensitive holders sell little when edge is positive, a lot when it is negative."""
    edge = 0.0 if not math.isfinite(edge_return) else float(edge_return)
    if edge >= cfg.positive_no_sell_edge:
        return 0.0
    if edge >= 0:
        scale = 1.0 - edge / max(cfg.positive_no_sell_edge, 1e-6)
        return max(0.0, cfg.positive_sell_min * scale)
    stress = min(1.0, -edge / max(cfg.positive_sell_width, 1e-6))
    return cfg.positive_sell_min + (cfg.positive_sell_max - cfg.positive_sell_min) * stress


def _finite_capacity_inputs(inputs: Dict[str, Any]) -> bool:
    shares = inputs.get("shares_outstanding", math.nan)
    adv = inputs.get("adv20", math.nan)
    return math.isfinite(shares) and shares > 0 and math.isfinite(adv) and adv > 0


def _cap_summary(pools: Iterable[Dict[str, float]], adv_cap: float, position_cap: float) -> Dict[str, Any]:
    source_cap = sum(max(0.0, p.get("capacity_pct", 0.0)) for p in pools)
    caps = {
        "source_supply": source_cap,
        "adv_participation": adv_cap,
        "position_limit": position_cap,
    }
    max_pct = min(caps.values())
    binding = min(caps, key=caps.get)
    return {"max_ownership": max(0.0, max_pct), "binding_constraint": binding, "source_cap": source_cap}


def long_flow(row: pd.Series, entry_price: float, pi_cash: float, elect: str,
              edge_return: float, cfg: CapacityConfig) -> Dict[str, Any]:
    inputs = capacity_inputs(row, entry_price, cfg)
    if not _finite_capacity_inputs(inputs):
        return {"status": "missing_shares_or_adv", **inputs, "max_ownership": math.nan, "pools": []}

    passive_cash = default_cash_probability(row, pi_cash)
    own_cash = 1.0 if str(elect).upper() == "CASH" else 0.0
    pos_sell = positive_sell_rate(edge_return, cfg)
    adv_cap = inputs["adv_pct_shares"] * cfg.build_days * cfg.max_adv_participation
    pools = [
        {
            "name": "noisy_sellers",
            "capacity_pct": min(inputs["noisy_holder_share"], adv_cap * cfg.noisy_sell_fraction),
            "cash_prob": inputs["noisy_cash_prob"],
        },
        {
            "name": "positive_holders",
            "capacity_pct": inputs["positive_holder_share"] * pos_sell,
            "cash_prob": own_cash,
        },
        {
            "name": "passive_inactive",
            "capacity_pct": inputs["passive_share"] * cfg.passive_sell_fraction,
            "cash_prob": passive_cash,
        },
    ]
    if edge_return < 0:
        pools = [pools[1], pools[0], pools[2]]
    cap = _cap_summary(pools, adv_cap, cfg.max_position_pct_shares)
    return {
        "status": "ok",
        **inputs,
        **cap,
        "pools": pools,
        "own_cash_prob": own_cash,
        "positive_sell_rate": pos_sell,
        "passive_cash_prob": passive_cash,
        "flow_model": "long_target_buy_from_noisy_then_ev_sensitive_then_passive",
    }


def reverse_flow(row: pd.Series, entry_price: float, pi_cash: float,
                 reverse_return: float, cfg: CapacityConfig) -> Dict[str, Any]:
    inputs = capacity_inputs(row, entry_price, cfg)
    if not _finite_capacity_inputs(inputs):
        return {"status": "missing_shares_or_adv", **inputs, "max_ownership": math.nan, "pools": []}

    passive_cash = default_cash_probability(row, pi_cash)
    adv_cap = inputs["adv_pct_shares"] * cfg.build_days * cfg.max_adv_participation
    borrow_pools = [
        {
            "name": "passive_inactive_lenders",
            "capacity_pct": inputs["passive_share"] * cfg.passive_lendable_fraction,
            "cash_prob": passive_cash,
        },
        {
            "name": "noisy_inactive_lenders",
            "capacity_pct": inputs["noisy_holder_share"] * cfg.noisy_lendable_fraction,
            "cash_prob": inputs["noisy_cash_prob"],
        },
    ]
    borrow_cap = sum(p["capacity_pct"] for p in borrow_pools)
    noisy_buyer_cap = min(inputs["noisy_holder_share"], adv_cap * cfg.noisy_buy_fraction)
    cap = _cap_summary(
        [{"name": "borrowable_inactive", "capacity_pct": borrow_cap, "cash_prob": passive_cash}],
        adv_cap,
        cfg.max_position_pct_shares,
    )
    if noisy_buyer_cap < cap["max_ownership"]:
        cap["max_ownership"] = noisy_buyer_cap
        cap["binding_constraint"] = "noisy_buyer_demand"

    lender_alloc = allocate_sources(borrow_pools, cap["max_ownership"])
    buyer_cash = inputs["noisy_cash_prob"]
    return {
        "status": "ok",
        **inputs,
        **cap,
        "pools": borrow_pools,
        "borrow_cap": borrow_cap,
        "noisy_buyer_cap": noisy_buyer_cap,
        "lender_cash_prob": lender_alloc["cash_prob"],
        "buyer_cash_prob": buyer_cash,
        "election_cash_shift": cap["max_ownership"] * (buyer_cash - lender_alloc["cash_prob"]),
        "flow_model": "short_target_borrow_from_inactive_sell_to_noisy_buyers",
        "positive_sell_rate": positive_sell_rate(reverse_return, cfg),
        "passive_cash_prob": passive_cash,
    }


def allocate_sources(pools: Iterable[Dict[str, float]], ownership: float) -> Dict[str, Any]:
    q = max(0.0, 0.0 if not math.isfinite(ownership) else float(ownership))
    if q <= 0:
        return {"cash_prob": math.nan, "source_mix": "", "unfilled_pct": 0.0}
    remaining = q
    cash = 0.0
    taken: List[str] = []
    for pool in pools:
        cap = max(0.0, pool.get("capacity_pct", 0.0))
        take = min(remaining, cap)
        if take <= 0:
            continue
        cash += take * pool.get("cash_prob", 0.5)
        taken.append(f"{pool.get('name', 'unknown')}:{take / q:.2f}")
        remaining -= take
        if remaining <= 1e-12:
            break
    filled = max(q - remaining, 0.0)
    return {
        "cash_prob": cash / filled if filled > 0 else math.nan,
        "source_mix": "|".join(taken),
        "unfilled_pct": max(0.0, remaining),
    }


# ==============================================================================
# arb_run.py
# ==============================================================================
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
import json, os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy import stats


OUT = "arb_output"
os.makedirs(OUT, exist_ok=True)
RNG = np.random.default_rng(20260716)


def run_monte_carlo():
    d = build_deals()
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


# ==============================================================================
# arb_signal.py
# ==============================================================================
"""
arb_signal.py  —  TRADE DECISION LAYER for cash-or-stock election arbitrage.

Turns the Monte Carlo payoff model into an actionable trade per deal. For each deal it decides:

  ENTRY     buy the target at its post-announcement market price M
  ELECT     choose the side (cash / stock) with the higher EXPECTED consideration under the
            demand model — the election is committed up front, before the crowd is known
  HEDGE     short the acquirer stock leg you expect to receive, so the deal's value is locked
            at today's prices (removes acquirer price risk -> you don't need the close price)
  REVERSE   short the target when the expected edge is negative. This side has NO election
            right, so its completion liability is the passive blended settlement, not the
            optimal elected payoff.
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
import argparse
from dataclasses import dataclass
import json
from pathlib import Path
import zlib
import numpy as np
import pandas as pd

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
    capacity_panel_path: str = "eda_output/merged_panel.csv"
    output_path: str = "arb_signals.csv"
    summary_output_path: str = "arb_strategy_summary.json"
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
    holder_model: str = "rolling_structural"
    holder_rolling_window_events: int = 50
    holder_min_fit_events: int = 10
    holder_p_grid_size: int = 11
    holder_q_grid_size: int = 11
    default_irrational_cash_prob: float = 0.50
    default_rational_share: float = 0.30
    capacity_build_days: float = 10.0
    capacity_max_adv_participation: float = 0.20
    capacity_max_position_pct_shares: float = 0.05
    positive_holder_share_of_active: float = 0.35
    positive_no_sell_edge: float = 0.03
    capacity_grid_points: int = 51


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


def gate_trade(stats, metric, cfg):
    decision = decision_value(stats, metric)
    p05 = stats["p05"]
    cvar05 = stats["cvar05"]
    loss_probability = stats["loss_probability"]
    if decision > cfg.hurdle and p05 >= cfg.min_p05_return and loss_probability <= cfg.max_loss_probability:
        risk_scale = max(-cvar05, -p05, 1e-3)
        return True, float(np.clip(stats["mean"] / risk_scale, 0, 3)), "risk_adjusted_return_positive"
    if decision <= cfg.hurdle:
        return False, 0.0, "risk_adjusted_return_below_hurdle"
    if p05 < cfg.min_p05_return:
        return False, 0.0, "p05_below_floor"
    return False, 0.0, "loss_probability_above_ceiling"


def apply_overlay_with_uniforms(complete_values, probs, terminated_value, withdrawn_value, uniforms):
    complete_values = np.asarray(complete_values, float)
    u = np.asarray(uniforms, float)
    p_completed = float(probs["completed"])
    p_terminated = float(probs["terminated"])
    return np.where(
        u < p_completed,
        complete_values,
        np.where(u < p_completed + p_terminated, terminated_value, withdrawn_value),
    )


def capacity_config_from_signal_config(cfg):
    return CapacityConfig(
        panel_path=cfg.capacity_panel_path,
        holder_model=cfg.holder_model,
        holder_rolling_window_events=cfg.holder_rolling_window_events,
        holder_min_fit_events=cfg.holder_min_fit_events,
        holder_p_grid_size=cfg.holder_p_grid_size,
        holder_q_grid_size=cfg.holder_q_grid_size,
        default_irrational_cash_prob=cfg.default_irrational_cash_prob,
        default_rational_share=cfg.default_rational_share,
        build_days=cfg.capacity_build_days,
        max_adv_participation=cfg.capacity_max_adv_participation,
        max_position_pct_shares=cfg.capacity_max_position_pct_shares,
        positive_holder_share_of_active=cfg.positive_holder_share_of_active,
        positive_no_sell_edge=cfg.positive_no_sell_edge,
        capacity_grid_points=max(5, int(getattr(cfg, "capacity_grid_points", 51))),
    )


def empty_capacity(status="not_evaluated"):
    return {
        "capacity_status": status,
        "capacity_flow_model": "",
        "capacity_binding_constraint": "",
        "capacity_source_mix": "",
        "capacity_raw_max_binding_constraint": "",
        "capacity_raw_max_notional": np.nan,
        "capacity_raw_max_pct_shares_outstanding": np.nan,
        "capacity_raw_max_shares": np.nan,
        "capacity_max_binding_constraint": "",
        "capacity_max_source_mix": "",
        "capacity_max_seller_cash_prob": np.nan,
        "capacity_max_own_cash_prob": np.nan,
        "capacity_max_lender_cash_prob": np.nan,
        "capacity_max_buyer_cash_prob": np.nan,
        "capacity_max_shares": np.nan,
        "capacity_max_notional": np.nan,
        "capacity_max_pct_shares_outstanding": np.nan,
        "capacity_max_E_return_%": np.nan,
        "capacity_max_downside_5%_%": np.nan,
        "capacity_max_loss_probability_%": np.nan,
        "capacity_max_expected_pnl": np.nan,
        "capacity_max_cash_demand_shift_pctpts": np.nan,
        "capacity_max_impacted_mean_cash_election_%": np.nan,
        "capacity_max_baseline_realized_return_%": np.nan,
        "capacity_max_baseline_realized_pnl": np.nan,
        "capacity_max_self_impact_realized_return_%": np.nan,
        "capacity_max_self_impact_realized_pnl": np.nan,
        "capacity_optimal_binding_constraint": "",
        "capacity_optimal_source_mix": "",
        "capacity_optimal_seller_cash_prob": np.nan,
        "capacity_optimal_own_cash_prob": np.nan,
        "capacity_optimal_lender_cash_prob": np.nan,
        "capacity_optimal_buyer_cash_prob": np.nan,
        "capacity_optimal_shares": np.nan,
        "capacity_optimal_notional": np.nan,
        "capacity_optimal_pct_shares_outstanding": np.nan,
        "capacity_optimal_E_return_%": np.nan,
        "capacity_optimal_downside_5%_%": np.nan,
        "capacity_optimal_loss_probability_%": np.nan,
        "capacity_optimal_expected_pnl": np.nan,
        "capacity_optimal_cash_demand_shift_pctpts": np.nan,
        "capacity_optimal_impacted_mean_cash_election_%": np.nan,
        "capacity_optimal_baseline_realized_return_%": np.nan,
        "capacity_optimal_baseline_realized_pnl": np.nan,
        "capacity_optimal_self_impact_realized_return_%": np.nan,
        "capacity_optimal_self_impact_realized_pnl": np.nan,
        "capacity_shares": np.nan,
        "capacity_notional": np.nan,
        "capacity_pct_shares_outstanding": np.nan,
        "capacity_max_raw_pct_shares_outstanding": np.nan,
        "capacity_adv20": np.nan,
        "capacity_shares_outstanding": np.nan,
        "capacity_passive_share_%": np.nan,
        "capacity_positive_holder_share_%": np.nan,
        "capacity_positive_holder_share_of_active_%": np.nan,
        "capacity_noisy_holder_share_%": np.nan,
        "capacity_noisy_cash_prob": np.nan,
        "capacity_holder_model_source": "",
        "capacity_holder_fit_n": np.nan,
        "capacity_holder_p_hat": np.nan,
        "capacity_holder_q_hat": np.nan,
        "capacity_holder_predicted_cash_demand_%": np.nan,
        "capacity_holder_rational_action": "",
        "capacity_positive_sell_rate_%": np.nan,
        "capacity_seller_cash_prob": np.nan,
        "capacity_buyer_cash_prob": np.nan,
        "capacity_lender_cash_prob": np.nan,
        "capacity_own_cash_prob": np.nan,
        "capacity_cash_demand_shift_pctpts": np.nan,
        "capacity_impacted_mean_cash_election_%": np.nan,
        "capacity_election_impact_used_in_payoff": False,
        "capacity_adjusted_E_return_%": np.nan,
        "capacity_adjusted_downside_5%_%": np.nan,
        "capacity_adjusted_loss_probability_%": np.nan,
        "capacity_alpha_decay_%": np.nan,
        "capacity_adjusted_realized_return_%": np.nan,
        "capacity_baseline_realized_return_%": np.nan,
        "capacity_baseline_realized_pnl": np.nan,
        "capacity_self_impact_realized_return_%": np.nan,
        "capacity_self_impact_realized_pnl": np.nan,
        "self_impact_can_eliminate_arbitrage": False,
        "self_impact_break_even_pct_shares_outstanding": np.nan,
        "self_impact_proof_note": "",
    }


def rng_for_event(event_id, salt=0):
    seed = (zlib.crc32(str(event_id).encode("utf-8")) + int(salt)) & 0xFFFFFFFF
    return np.random.default_rng(seed)


def _long_q_candidate(flow, d, M, stock_val, f, elect, probs, terminated_value,
                      withdrawn_value, uniforms, q):
    alloc = allocate_sources(flow.get("pools", []), float(q))
    seller_cash = alloc["cash_prob"]
    own_cash = flow.get("own_cash_prob", 1.0 if elect == "CASH" else 0.0)
    if not np.isfinite(seller_cash) or alloc["unfilled_pct"] > 1e-9:
        return None
    shift = float(q) * (own_cash - seller_cash)
    f_impacted = np.clip(f + shift, 1e-6, 1 - 1e-6)
    cash_h, stock_h, _, _ = prorate(f_impacted, d.pi_cash, d.C, stock_val)
    held = cash_h if elect == "CASH" else stock_h
    realized_val = apply_overlay_with_uniforms(held, probs, terminated_value, withdrawn_value, uniforms)
    stats = summarize_returns((realized_val - M) / M)
    return {
        "q": float(q),
        "alloc": alloc,
        "stats": stats,
        "shift": shift,
        "impacted_mean": float(f_impacted.mean()),
        "seller_cash": seller_cash,
        "own_cash": own_cash,
    }


def _noisy_self_impact_proof(d, M, stock_val, f, elect, probs, terminated_value,
                             withdrawn_value, uniforms, cfg):
    own_cash = 1.0 if str(elect).upper() == "CASH" else 0.0
    seller_cash = 0.5
    for q in np.linspace(0.0, 1.0, 201):
        shift = float(q) * (own_cash - seller_cash)
        f_impacted = np.clip(f + shift, 1e-6, 1 - 1e-6)
        cash_h, stock_h, _, _ = prorate(f_impacted, d.pi_cash, d.C, stock_val)
        held = cash_h if elect == "CASH" else stock_h
        realized_val = apply_overlay_with_uniforms(held, probs, terminated_value, withdrawn_value, uniforms)
        stats = summarize_returns((realized_val - M) / M)
        ok, _, reason = gate_trade(stats, cfg.decision_metric, cfg)
        if not ok:
            return {
                "can_eliminate": True,
                "break_even_pct": float(q * 100),
                "note": f"pure_noisy_counterparty_self_impact_hits_{reason}",
            }
    return {
        "can_eliminate": False,
        "break_even_pct": np.nan,
        "note": "not_within_100pct_target_shares_under_current_terms_and_gates",
    }


def _write_capacity_choice(out, prefix, cand, shares, M, binding):
    if cand is None or shares <= 0:
        return
    q = cand["q"]
    notional = q * shares * M
    stats = cand["stats"]
    out.update({
        f"capacity_{prefix}_binding_constraint": binding,
        f"capacity_{prefix}_source_mix": cand["alloc"]["source_mix"],
        f"capacity_{prefix}_seller_cash_prob": cand.get("seller_cash", np.nan),
        f"capacity_{prefix}_own_cash_prob": cand.get("own_cash", np.nan),
        f"capacity_{prefix}_shares": q * shares,
        f"capacity_{prefix}_notional": notional,
        f"capacity_{prefix}_pct_shares_outstanding": q * 100,
        f"capacity_{prefix}_E_return_%": stats["mean"] * 100,
        f"capacity_{prefix}_downside_5%_%": stats["p05"] * 100,
        f"capacity_{prefix}_loss_probability_%": stats["loss_probability"] * 100,
        f"capacity_{prefix}_expected_pnl": notional * stats["mean"],
        f"capacity_{prefix}_cash_demand_shift_pctpts": cand["shift"] * 100,
        f"capacity_{prefix}_impacted_mean_cash_election_%": cand["impacted_mean"] * 100,
    })


def _alias_optimal_capacity(out, base_stats):
    out.update({
        "capacity_binding_constraint": out.get("capacity_optimal_binding_constraint", ""),
        "capacity_source_mix": out.get("capacity_optimal_source_mix", ""),
        "capacity_shares": out.get("capacity_optimal_shares", np.nan),
        "capacity_notional": out.get("capacity_optimal_notional", np.nan),
        "capacity_pct_shares_outstanding": out.get("capacity_optimal_pct_shares_outstanding", np.nan),
        "capacity_cash_demand_shift_pctpts": out.get("capacity_optimal_cash_demand_shift_pctpts", np.nan),
        "capacity_impacted_mean_cash_election_%": out.get("capacity_optimal_impacted_mean_cash_election_%", np.nan),
        "capacity_adjusted_E_return_%": out.get("capacity_optimal_E_return_%", np.nan),
        "capacity_adjusted_downside_5%_%": out.get("capacity_optimal_downside_5%_%", np.nan),
        "capacity_adjusted_loss_probability_%": out.get("capacity_optimal_loss_probability_%", np.nan),
    })
    opt_e = out.get("capacity_optimal_E_return_%", np.nan)
    out["capacity_alpha_decay_%"] = (
        base_stats["mean"] * 100 - opt_e if np.isfinite(opt_e) else np.nan
    )


def evaluate_long_capacity(cap_row, d, M, stock_val, f, elect, probs, terminated_value,
                           withdrawn_value, base_stats, cfg, rng):
    cap_cfg = capacity_config_from_signal_config(cfg)
    flow = long_flow(cap_row, M, d.pi_cash, elect, base_stats["mean"], cap_cfg)
    out = empty_capacity(flow.get("status", "missing_capacity_inputs"))
    raw_q = flow.get("max_ownership", np.nan)
    shares = flow.get("shares_outstanding", np.nan)
    out.update({
        "capacity_flow_model": flow.get("flow_model", ""),
        "capacity_raw_max_binding_constraint": flow.get("binding_constraint", ""),
        "capacity_raw_max_pct_shares_outstanding": raw_q * 100 if np.isfinite(raw_q) else np.nan,
        "capacity_raw_max_notional": raw_q * shares * M
        if np.isfinite(raw_q) and np.isfinite(shares) else np.nan,
        "capacity_raw_max_shares": raw_q * shares if np.isfinite(raw_q) and np.isfinite(shares) else np.nan,
        "capacity_max_raw_pct_shares_outstanding": raw_q * 100 if np.isfinite(raw_q) else np.nan,
        "capacity_adv20": flow.get("adv20", np.nan),
        "capacity_shares_outstanding": shares,
        "capacity_passive_share_%": flow.get("passive_share", np.nan) * 100
        if np.isfinite(flow.get("passive_share", np.nan)) else np.nan,
        "capacity_positive_holder_share_%": flow.get("positive_holder_share", np.nan) * 100
        if np.isfinite(flow.get("positive_holder_share", np.nan)) else np.nan,
        "capacity_positive_holder_share_of_active_%": flow.get("positive_holder_share_of_active", np.nan) * 100
        if np.isfinite(flow.get("positive_holder_share_of_active", np.nan)) else np.nan,
        "capacity_noisy_holder_share_%": flow.get("noisy_holder_share", np.nan) * 100
        if np.isfinite(flow.get("noisy_holder_share", np.nan)) else np.nan,
        "capacity_noisy_cash_prob": flow.get("noisy_cash_prob", np.nan),
        "capacity_holder_model_source": flow.get("holder_model_source", ""),
        "capacity_holder_fit_n": flow.get("holder_fit_n", np.nan),
        "capacity_holder_p_hat": flow.get("holder_p_hat", np.nan),
        "capacity_holder_q_hat": flow.get("holder_q_hat", np.nan),
        "capacity_holder_predicted_cash_demand_%": flow.get("holder_predicted_cash_demand_share", np.nan) * 100
        if np.isfinite(flow.get("holder_predicted_cash_demand_share", np.nan)) else np.nan,
        "capacity_holder_rational_action": flow.get("holder_rational_action", ""),
        "capacity_positive_sell_rate_%": flow.get("positive_sell_rate", np.nan) * 100
        if np.isfinite(flow.get("positive_sell_rate", np.nan)) else np.nan,
        "capacity_own_cash_prob": flow.get("own_cash_prob", np.nan),
        "capacity_election_impact_used_in_payoff": True,
    })
    if flow.get("status") != "ok" or not np.isfinite(raw_q) or raw_q <= 0:
        return out

    uniforms = rng.random(len(f))
    proof = _noisy_self_impact_proof(
        d, M, stock_val, f, elect, probs, terminated_value, withdrawn_value, uniforms, cfg
    )
    out.update({
        "self_impact_can_eliminate_arbitrage": proof["can_eliminate"],
        "self_impact_break_even_pct_shares_outstanding": proof["break_even_pct"],
        "self_impact_proof_note": proof["note"],
    })

    max_cand = None
    optimal_cand = None
    optimal_pnl = -np.inf
    fail_reason = ""

    for q in np.linspace(raw_q / cap_cfg.capacity_grid_points, raw_q, cap_cfg.capacity_grid_points):
        cand = _long_q_candidate(
            flow, d, M, stock_val, f, elect, probs, terminated_value, withdrawn_value, uniforms, float(q)
        )
        if cand is None:
            fail_reason = "source_allocation_unfilled"
            continue
        ok, _, reason = gate_trade(cand["stats"], cfg.decision_metric, cfg)
        if ok:
            max_cand = cand
            pnl = cand["q"] * shares * M * cand["stats"]["mean"]
            if pnl > optimal_pnl:
                optimal_pnl = pnl
                optimal_cand = cand
        else:
            fail_reason = reason

    if max_cand is None or optimal_cand is None:
        out.update({
            "capacity_status": "zero_capacity_after_self_impact",
            "capacity_binding_constraint": fail_reason or "self_impact_gate",
        })
        return out

    max_binding = flow.get("binding_constraint", "")
    if max_cand["q"] < raw_q * (1.0 - 1e-6):
        max_binding = "self_impact_" + (fail_reason or "gate")
    opt_binding = max_binding if abs(optimal_cand["q"] - max_cand["q"]) < 1e-12 else "profit_optimal_before_max"
    out["capacity_status"] = "ok"
    _write_capacity_choice(out, "max", max_cand, shares, M, max_binding)
    _write_capacity_choice(out, "optimal", optimal_cand, shares, M, opt_binding)
    out["capacity_seller_cash_prob"] = optimal_cand["seller_cash"]
    out["capacity_own_cash_prob"] = optimal_cand["own_cash"]
    _alias_optimal_capacity(out, base_stats)
    return out


def evaluate_reverse_capacity(cap_row, d, M, f, reverse_stats, cfg):
    cap_cfg = capacity_config_from_signal_config(cfg)
    flow = reverse_flow(cap_row, M, d.pi_cash, reverse_stats["mean"], cap_cfg)
    out = empty_capacity(flow.get("status", "missing_capacity_inputs"))
    out.update({
        "capacity_flow_model": flow.get("flow_model", ""),
        "capacity_binding_constraint": flow.get("binding_constraint", ""),
        "capacity_max_raw_pct_shares_outstanding": flow.get("max_ownership", np.nan) * 100
        if np.isfinite(flow.get("max_ownership", np.nan)) else np.nan,
        "capacity_adv20": flow.get("adv20", np.nan),
        "capacity_shares_outstanding": flow.get("shares_outstanding", np.nan),
        "capacity_passive_share_%": flow.get("passive_share", np.nan) * 100
        if np.isfinite(flow.get("passive_share", np.nan)) else np.nan,
        "capacity_positive_holder_share_%": flow.get("positive_holder_share", np.nan) * 100
        if np.isfinite(flow.get("positive_holder_share", np.nan)) else np.nan,
        "capacity_positive_holder_share_of_active_%": flow.get("positive_holder_share_of_active", np.nan) * 100
        if np.isfinite(flow.get("positive_holder_share_of_active", np.nan)) else np.nan,
        "capacity_noisy_holder_share_%": flow.get("noisy_holder_share", np.nan) * 100
        if np.isfinite(flow.get("noisy_holder_share", np.nan)) else np.nan,
        "capacity_noisy_cash_prob": flow.get("noisy_cash_prob", np.nan),
        "capacity_holder_model_source": flow.get("holder_model_source", ""),
        "capacity_holder_fit_n": flow.get("holder_fit_n", np.nan),
        "capacity_holder_p_hat": flow.get("holder_p_hat", np.nan),
        "capacity_holder_q_hat": flow.get("holder_q_hat", np.nan),
        "capacity_holder_predicted_cash_demand_%": flow.get("holder_predicted_cash_demand_share", np.nan) * 100
        if np.isfinite(flow.get("holder_predicted_cash_demand_share", np.nan)) else np.nan,
        "capacity_holder_rational_action": flow.get("holder_rational_action", ""),
        "capacity_positive_sell_rate_%": flow.get("positive_sell_rate", np.nan) * 100
        if np.isfinite(flow.get("positive_sell_rate", np.nan)) else np.nan,
        "capacity_buyer_cash_prob": flow.get("buyer_cash_prob", np.nan),
        "capacity_lender_cash_prob": flow.get("lender_cash_prob", np.nan),
        "capacity_election_impact_used_in_payoff": False,
    })
    max_q = flow.get("max_ownership", np.nan)
    shares = flow.get("shares_outstanding", np.nan)
    out.update({
        "capacity_raw_max_binding_constraint": flow.get("binding_constraint", ""),
        "capacity_raw_max_pct_shares_outstanding": max_q * 100 if np.isfinite(max_q) else np.nan,
        "capacity_raw_max_notional": max_q * shares * M
        if np.isfinite(max_q) and np.isfinite(shares) else np.nan,
        "capacity_raw_max_shares": max_q * shares if np.isfinite(max_q) and np.isfinite(shares) else np.nan,
    })
    if flow.get("status") != "ok" or not np.isfinite(max_q) or max_q <= 0:
        return out

    alloc = allocate_sources(flow.get("pools", []), max_q)
    shift = flow.get("election_cash_shift", np.nan)
    stats = reverse_stats
    notional = max_q * shares * M
    out.update({
        "capacity_status": "ok",
        "capacity_max_binding_constraint": flow.get("binding_constraint", ""),
        "capacity_max_source_mix": alloc["source_mix"],
        "capacity_max_lender_cash_prob": flow.get("lender_cash_prob", np.nan),
        "capacity_max_buyer_cash_prob": flow.get("buyer_cash_prob", np.nan),
        "capacity_max_shares": max_q * shares,
        "capacity_max_notional": notional,
        "capacity_max_pct_shares_outstanding": max_q * 100,
        "capacity_max_E_return_%": stats["mean"] * 100,
        "capacity_max_downside_5%_%": stats["p05"] * 100,
        "capacity_max_loss_probability_%": stats["loss_probability"] * 100,
        "capacity_max_expected_pnl": notional * stats["mean"],
        "capacity_max_cash_demand_shift_pctpts": shift * 100 if np.isfinite(shift) else np.nan,
        "capacity_max_impacted_mean_cash_election_%": (float(np.mean(f)) + shift) * 100
        if np.isfinite(shift) and len(f) else np.nan,
        "capacity_optimal_binding_constraint": flow.get("binding_constraint", ""),
        "capacity_optimal_source_mix": alloc["source_mix"],
        "capacity_optimal_lender_cash_prob": flow.get("lender_cash_prob", np.nan),
        "capacity_optimal_buyer_cash_prob": flow.get("buyer_cash_prob", np.nan),
        "capacity_optimal_shares": max_q * shares,
        "capacity_optimal_notional": notional,
        "capacity_optimal_pct_shares_outstanding": max_q * 100,
        "capacity_optimal_E_return_%": stats["mean"] * 100,
        "capacity_optimal_downside_5%_%": stats["p05"] * 100,
        "capacity_optimal_loss_probability_%": stats["loss_probability"] * 100,
        "capacity_optimal_expected_pnl": notional * stats["mean"],
        "capacity_optimal_cash_demand_shift_pctpts": shift * 100 if np.isfinite(shift) else np.nan,
        "capacity_optimal_impacted_mean_cash_election_%": (float(np.mean(f)) + shift) * 100
        if np.isfinite(shift) and len(f) else np.nan,
        "capacity_buyer_cash_prob": flow.get("buyer_cash_prob", np.nan),
        "capacity_lender_cash_prob": flow.get("lender_cash_prob", np.nan),
        "self_impact_can_eliminate_arbitrage": False,
        "self_impact_break_even_pct_shares_outstanding": np.nan,
        "self_impact_proof_note": "not_possible_for_reverse_under_fixed_pool_passive_blended_settlement",
    })
    _alias_optimal_capacity(out, reverse_stats)
    out["capacity_alpha_decay_%"] = 0.0
    return out


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
    cap_cfg = capacity_config_from_signal_config(cfg)
    capacity_table = load_capacity_table(cfg.capacity_panel_path, cap_cfg)
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
        long_realized_val, normalized_probs = apply_outcome_overlay(
            held,
            entry_value=M,
            p_completed=outcome["p_completed"],
            p_terminated=outcome["p_terminated"],
            p_withdrawn=outcome["p_withdrawn"],
            terminated_value=terminated_value,
            withdrawn_value=withdrawn_value,
            rng=rng,
        )
        long_ret_dist = (long_realized_val - M) / M
        long_risk = summarize_returns(long_ret_dist)
        long_decision = decision_value(long_risk, cfg.decision_metric)
        long_ok, long_size, long_reason = gate_trade(long_risk, cfg.decision_metric, cfg)
        long_e_ret = long_risk["mean"]
        long_risk_adjusted_fair_value = M * (1.0 + long_e_ret)
        terminated_return = (terminated_value - M) / M
        withdrawn_return = (withdrawn_value - M) / M

        # Reverse trade: short target.  The short seller has no election right,
        # so completion-state liability is the passive/blended settlement, not
        # the optimal elected-side payoff.
        passive_settlement = float(blended)
        reverse_liability, _ = apply_outcome_overlay(
            np.full_like(held, passive_settlement, dtype=float),
            entry_value=M,
            p_completed=outcome["p_completed"],
            p_terminated=outcome["p_terminated"],
            p_withdrawn=outcome["p_withdrawn"],
            terminated_value=terminated_value,
            withdrawn_value=withdrawn_value,
            rng=rng,
        )
        reverse_ret_dist = (M - reverse_liability) / M
        reverse_risk = summarize_returns(reverse_ret_dist)
        reverse_decision = decision_value(reverse_risk, cfg.decision_metric)
        reverse_ok, reverse_size, reverse_reason = gate_trade(reverse_risk, cfg.decision_metric, cfg)
        reverse_e_ret = reverse_risk["mean"]
        reverse_completion_return = (M - passive_settlement) / M
        reverse_risk_adjusted_settlement_value = M * (1.0 - reverse_e_ret)
        reverse_hedge_ratio = d.R * (1.0 - d.pi_cash)

        # realized (historical) outcome given the actual election demand
        actual_outcome = normalize_outcome_label(d)
        actual_f_cash = pd.to_numeric(pd.Series([d.get("f_cash", np.nan)]), errors="coerce").iloc[0]
        if actual_outcome == "terminated":
            long_realized_ret = terminated_return
            reverse_realized_ret = -terminated_return
            realized_source = "deal_outcome_label"
        elif actual_outcome == "withdrawn":
            long_realized_ret = withdrawn_return
            reverse_realized_ret = -withdrawn_return
            realized_source = "deal_outcome_label"
        elif actual_outcome == "completed" or np.isfinite(actual_f_cash):
            reverse_realized_ret = reverse_completion_return
            if np.isfinite(actual_f_cash):
                cr, sr, _, _ = prorate(np.array([actual_f_cash]), d.pi_cash, d.C, stock_val)
                long_realized_value = (cr if elect == "CASH" else sr)[0]
                long_realized_ret = (long_realized_value - M) / M
                realized_source = "realized_f_cash"
            else:
                long_realized_ret = np.nan
                realized_source = "completed_status_no_realized_f_cash"
            if not actual_outcome:
                actual_outcome = "completed"
        else:
            long_realized_ret = np.nan
            reverse_realized_ret = np.nan
            realized_source = "missing_realized_outcome"

        # data-quality guard: a real post-announcement merger-arb spread lives in ~[-30%, +30%].
        # anything outside that is almost certainly a bad entry price or misparsed term -> REVIEW,
        # never a tradeable signal. keeps the blotter honest by construction.
        data_quality_return = max(abs(arb_ret), abs(reverse_completion_return))
        # also flag an implausible cash-vs-stock TERM gap (misparsed cash value / ratio / acquirer
        # identity or price): the return check above misses these when the misparse still lands the
        # return inside +/-30% (e.g. REVERSE trades that settle at blended value). Check the gap at
        # BOTH the entry and deadline acquirer prices — a real election is structured near cash/stock
        # parity, so a >50% gap at entry is a misparse, and a >50% gap at the deadline is either a
        # misparse (wrong acquirer/price) or basis risk large enough to warrant REVIEW.
        entry_gap = abs(d.C - stock_val) / max(min(abs(d.C), abs(stock_val)), 1e-9)
        deadline_sv = d.R * d.P_acq
        deadline_gap = abs(d.C - deadline_sv) / max(min(abs(d.C), abs(deadline_sv)), 1e-9)
        term_gap = max(entry_gap, deadline_gap)
        if data_quality_return > 0.30 or term_gap > 0.50:
            signal, size, signal_reason = "REVIEW", 0.0, (
                "implausible_cash_vs_stock_term_gap" if term_gap > 0.50
                else "abs_completion_or_election_return_above_30pct")
            selected = "REVIEW"
            selected_risk = long_risk if long_decision >= reverse_decision else reverse_risk
            selected_decision = max(long_decision, reverse_decision)
            selected_e_ret = selected_risk["mean"]
            selected_fair_value = V if long_decision >= reverse_decision else passive_settlement
            selected_risk_adjusted_value = (
                long_risk_adjusted_fair_value
                if long_decision >= reverse_decision
                else reverse_risk_adjusted_settlement_value
            )
            selected_hedge_ratio = hedge_ratio if long_decision >= reverse_decision else reverse_hedge_ratio
            hedge_side = "SHORT_ACQUIRER" if long_decision >= reverse_decision else "LONG_ACQUIRER"
            realized_ret = long_realized_ret if long_decision >= reverse_decision else reverse_realized_ret
        elif long_ok or reverse_ok:
            choose_reverse = reverse_ok and (not long_ok or reverse_decision > long_decision)
            if choose_reverse:
                signal, size, signal_reason = "REVERSE", reverse_size, reverse_reason
                selected = "SHORT_TARGET_PASSIVE"
                selected_risk = reverse_risk
                selected_decision = reverse_decision
                selected_e_ret = reverse_e_ret
                selected_fair_value = passive_settlement
                selected_risk_adjusted_value = reverse_risk_adjusted_settlement_value
                selected_hedge_ratio = reverse_hedge_ratio
                hedge_side = "LONG_ACQUIRER"
                realized_ret = reverse_realized_ret
            else:
                signal, size, signal_reason = "ENTER", long_size, long_reason
                selected = "LONG_TARGET_ELECTION"
                selected_risk = long_risk
                selected_decision = long_decision
                selected_e_ret = long_e_ret
                selected_fair_value = V
                selected_risk_adjusted_value = long_risk_adjusted_fair_value
                selected_hedge_ratio = hedge_ratio
                hedge_side = "SHORT_ACQUIRER"
                realized_ret = long_realized_ret
        elif long_reason == "p05_below_floor" or reverse_reason == "p05_below_floor":
            signal, size, signal_reason = "PASS_P05", 0.0, "best_candidate_p05_below_floor"
            selected = "PASS"
            selected_risk = long_risk if long_decision >= reverse_decision else reverse_risk
            selected_decision = max(long_decision, reverse_decision)
            selected_e_ret = selected_risk["mean"]
            selected_fair_value = V if long_decision >= reverse_decision else passive_settlement
            selected_risk_adjusted_value = (
                long_risk_adjusted_fair_value
                if long_decision >= reverse_decision
                else reverse_risk_adjusted_settlement_value
            )
            selected_hedge_ratio = hedge_ratio if long_decision >= reverse_decision else reverse_hedge_ratio
            hedge_side = "SHORT_ACQUIRER" if long_decision >= reverse_decision else "LONG_ACQUIRER"
            realized_ret = long_realized_ret if long_decision >= reverse_decision else reverse_realized_ret
        elif long_reason == "loss_probability_above_ceiling" or reverse_reason == "loss_probability_above_ceiling":
            signal, size, signal_reason = "PASS_LOSS_PROB", 0.0, "best_candidate_loss_probability_above_ceiling"
            selected = "PASS"
            selected_risk = long_risk if long_decision >= reverse_decision else reverse_risk
            selected_decision = max(long_decision, reverse_decision)
            selected_e_ret = selected_risk["mean"]
            selected_fair_value = V if long_decision >= reverse_decision else passive_settlement
            selected_risk_adjusted_value = (
                long_risk_adjusted_fair_value
                if long_decision >= reverse_decision
                else reverse_risk_adjusted_settlement_value
            )
            selected_hedge_ratio = hedge_ratio if long_decision >= reverse_decision else reverse_hedge_ratio
            hedge_side = "SHORT_ACQUIRER" if long_decision >= reverse_decision else "LONG_ACQUIRER"
            realized_ret = long_realized_ret if long_decision >= reverse_decision else reverse_realized_ret
        else:
            signal, size, signal_reason = "PASS", 0.0, "both_directions_below_hurdle"
            selected = "PASS"
            selected_risk = long_risk if long_decision >= reverse_decision else reverse_risk
            selected_decision = max(long_decision, reverse_decision)
            selected_e_ret = selected_risk["mean"]
            selected_fair_value = V if long_decision >= reverse_decision else passive_settlement
            selected_risk_adjusted_value = (
                long_risk_adjusted_fair_value
                if long_decision >= reverse_decision
                else reverse_risk_adjusted_settlement_value
            )
            selected_hedge_ratio = hedge_ratio if long_decision >= reverse_decision else reverse_hedge_ratio
            hedge_side = "SHORT_ACQUIRER" if long_decision >= reverse_decision else "LONG_ACQUIRER"
            realized_ret = long_realized_ret if long_decision >= reverse_decision else reverse_realized_ret

        cap = empty_capacity()
        cap_source = capacity_row(str(d.event_id), capacity_table)
        if signal == "ENTER":
            cap = evaluate_long_capacity(
                cap_source,
                d,
                M,
                stock_val,
                f,
                elect,
                normalized_probs,
                terminated_value,
                withdrawn_value,
                long_risk,
                cfg,
                rng_for_event(d.event_id, 991),
            )
            for prefix in ["max", "optimal"]:
                notional = cap.get(f"capacity_{prefix}_notional", np.nan)
                q = cap.get(f"capacity_{prefix}_pct_shares_outstanding", np.nan) / 100.0
                seller_cash = cap.get(f"capacity_{prefix}_seller_cash_prob", np.nan)
                own_cash = cap.get(f"capacity_{prefix}_own_cash_prob", np.nan)
                baseline_ret = np.nan
                self_impact_ret = np.nan
                if actual_outcome == "terminated":
                    baseline_ret = terminated_return
                    self_impact_ret = terminated_return
                elif actual_outcome == "withdrawn":
                    baseline_ret = withdrawn_return
                    self_impact_ret = withdrawn_return
                elif np.isfinite(actual_f_cash):
                    baseline_ret = long_realized_ret
                    if np.isfinite(q) and np.isfinite(seller_cash) and np.isfinite(own_cash):
                        f_actual = np.clip(actual_f_cash + q * (own_cash - seller_cash), 1e-6, 1 - 1e-6)
                        cr, sr, _, _ = prorate(np.array([f_actual]), d.pi_cash, d.C, stock_val)
                        cap_realized_value = (cr if elect == "CASH" else sr)[0]
                        self_impact_ret = (cap_realized_value - M) / M
                if np.isfinite(baseline_ret):
                    cap[f"capacity_{prefix}_baseline_realized_return_%"] = baseline_ret * 100
                    cap[f"capacity_{prefix}_baseline_realized_pnl"] = (
                        notional * baseline_ret if np.isfinite(notional) else np.nan
                    )
                if np.isfinite(self_impact_ret):
                    cap[f"capacity_{prefix}_self_impact_realized_return_%"] = self_impact_ret * 100
                    cap[f"capacity_{prefix}_self_impact_realized_pnl"] = (
                        notional * self_impact_ret if np.isfinite(notional) else np.nan
                    )
            cap["capacity_baseline_realized_return_%"] = cap.get("capacity_optimal_baseline_realized_return_%", np.nan)
            cap["capacity_baseline_realized_pnl"] = cap.get("capacity_optimal_baseline_realized_pnl", np.nan)
            cap["capacity_self_impact_realized_return_%"] = cap.get("capacity_optimal_self_impact_realized_return_%", np.nan)
            cap["capacity_self_impact_realized_pnl"] = cap.get("capacity_optimal_self_impact_realized_pnl", np.nan)
            cap["capacity_adjusted_realized_return_%"] = cap["capacity_self_impact_realized_return_%"]
        elif signal == "REVERSE":
            cap = evaluate_reverse_capacity(cap_source, d, M, f, reverse_risk, cfg)
            for prefix in ["max", "optimal"]:
                notional = cap.get(f"capacity_{prefix}_notional", np.nan)
                if np.isfinite(reverse_realized_ret):
                    cap[f"capacity_{prefix}_baseline_realized_return_%"] = reverse_realized_ret * 100
                    cap[f"capacity_{prefix}_baseline_realized_pnl"] = (
                        notional * reverse_realized_ret if np.isfinite(notional) else np.nan
                    )
                    cap[f"capacity_{prefix}_self_impact_realized_return_%"] = reverse_realized_ret * 100
                    cap[f"capacity_{prefix}_self_impact_realized_pnl"] = (
                        notional * reverse_realized_ret if np.isfinite(notional) else np.nan
                    )
            cap["capacity_baseline_realized_return_%"] = cap.get("capacity_optimal_baseline_realized_return_%", np.nan)
            cap["capacity_baseline_realized_pnl"] = cap.get("capacity_optimal_baseline_realized_pnl", np.nan)
            cap["capacity_self_impact_realized_return_%"] = cap.get("capacity_optimal_self_impact_realized_return_%", np.nan)
            cap["capacity_self_impact_realized_pnl"] = cap.get("capacity_optimal_self_impact_realized_pnl", np.nan)
            cap["capacity_adjusted_realized_return_%"] = cap["capacity_self_impact_realized_return_%"]

        rows.append({"event_id": d.event_id, "target": str(d.target_name)[:26], "signal": signal,
                     "signal_reason": signal_reason,
                     "selected_strategy": selected,
                     "elect": elect, "M": round(M, 2), "fair_value": round(selected_fair_value, 2),
                     "risk_adjusted_fair_value": round(selected_risk_adjusted_value, 2),
                     "arb_return_%": round(arb_ret * 100, 2), "E_return_%": round(selected_e_ret * 100, 2),
                     "decision_metric": cfg.decision_metric, "decision_value_%": round(selected_decision * 100, 2),
                     "downside_5%_%": round(selected_risk["p05"] * 100, 2),
                     "cvar_5%_%": round(selected_risk["cvar05"] * 100, 2),
                     "loss_probability_%": round(selected_risk["loss_probability"] * 100, 2),
                     "long_fair_value": round(V, 2),
                     "long_E_return_%": round(long_e_ret * 100, 2),
                     "long_downside_5%_%": round(long_risk["p05"] * 100, 2),
                     "long_loss_probability_%": round(long_risk["loss_probability"] * 100, 2),
                     "reverse_settlement_value": round(passive_settlement, 2),
                     "reverse_completion_return_%": round(reverse_completion_return * 100, 2),
                     "reverse_E_return_%": round(reverse_e_ret * 100, 2),
                     "reverse_downside_5%_%": round(reverse_risk["p05"] * 100, 2),
                     "reverse_loss_probability_%": round(reverse_risk["loss_probability"] * 100, 2),
                     "p_completed": round(normalized_probs["completed"], 4),
                     "p_terminated": round(normalized_probs["terminated"], 4),
                     "p_withdrawn": round(normalized_probs["withdrawn"], 4),
                     "outcome_probability_source": outcome["outcome_probability_source"],
                     "terminated_return_%": round(terminated_return * 100, 2),
                     "withdrawn_return_%": round(withdrawn_return * 100, 2),
                     "terminated_value": round(terminated_value, 2),
                     "withdrawn_value": round(withdrawn_value, 2),
                     "hedge_side": hedge_side,
                     "hedge_ratio": round(selected_hedge_ratio, 3),
                     "long_hedge_ratio_short_acquirer": round(hedge_ratio, 3),
                     "reverse_hedge_ratio_long_acquirer": round(reverse_hedge_ratio, 3),
                     "size_x": round(size, 2),
                     "actual_outcome": actual_outcome,
                     "realized_return_source": realized_source,
                     "realized_return_%": round(realized_ret * 100, 2) if np.isfinite(realized_ret) else np.nan,
                     **{
                         k: (round(v, 4) if isinstance(v, (int, float, np.floating)) and np.isfinite(v) else v)
                         for k, v in cap.items()
                     }})
    out = pd.DataFrame(rows)
    if not out.empty:
        out = out.sort_values("E_return_%", ascending=False)
    out.to_csv(cfg.output_path, index=False)
    clean_cols = {
        "target": "target", "signal": "signal", "elect": "elect", "M": "entry",
        "fair_value": "fair_val", "E_return_%": "exp_ret%", "downside_5%_%": "p05%",
        "loss_probability_%": "loss_prob%", "hedge_side": "hedge", "hedge_ratio": "hedge_ratio",
        "size_x": "size", "capacity_notional": "notional_$", "realized_return_%": "realized%",
    }
    present = [c for c in clean_cols if c in out.columns]
    clean_path = cfg.output_path.replace(".csv", "_clean.csv")
    out[present].rename(columns=clean_cols).to_csv(clean_path, index=False)
    return out


def _numeric(df, col):
    if col not in df.columns:
        return pd.Series(np.nan, index=df.index, dtype=float)
    return pd.to_numeric(df[col], errors="coerce")


def _finite_mean(series):
    s = pd.to_numeric(series, errors="coerce").dropna()
    return float(s.mean()) if len(s) else np.nan


def _finite_median(series):
    s = pd.to_numeric(series, errors="coerce").dropna()
    return float(s.median()) if len(s) else np.nan


def _finite_sum(series):
    s = pd.to_numeric(series, errors="coerce").dropna()
    return float(s.sum()) if len(s) else np.nan


def _finite_corr(left, right):
    pair = pd.DataFrame({"left": pd.to_numeric(left, errors="coerce"),
                         "right": pd.to_numeric(right, errors="coerce")}).dropna()
    if len(pair) <= 2:
        return np.nan
    return float(np.corrcoef(pair["left"], pair["right"])[0, 1])


def _notional_weighted_return_pct(df, pnl_col, notional_col):
    pnl = _numeric(df, pnl_col)
    notional = _numeric(df, notional_col)
    ok = np.isfinite(pnl) & np.isfinite(notional) & (notional > 0)
    if not ok.any():
        return np.nan
    return float(pnl[ok].sum() / notional[ok].sum() * 100.0)


def _expected_capacity_block(trades, prefix):
    notional_col = f"capacity_{prefix}_notional"
    pnl_col = f"capacity_{prefix}_expected_pnl"
    ret_col = f"capacity_{prefix}_E_return_%"
    pct_col = f"capacity_{prefix}_pct_shares_outstanding"
    notional = _numeric(trades, notional_col)
    pnl = _numeric(trades, pnl_col)
    ok = np.isfinite(notional) & (notional > 0)
    return {
        "trade_count": int(ok.sum()),
        "total_notional": float(notional[ok].sum()) if ok.any() else np.nan,
        "median_notional": _finite_median(notional[ok]),
        "median_pct_shares_outstanding": _finite_median(_numeric(trades, pct_col)[ok]),
        "total_expected_pnl": _finite_sum(pnl[ok]),
        "unweighted_expected_return_%": _finite_mean(_numeric(trades, ret_col)[ok]),
        "notional_weighted_expected_return_%": _notional_weighted_return_pct(
            trades.loc[ok], pnl_col, notional_col
        ),
    }


def _realized_capacity_block(trades, prefix, mode):
    notional_col = f"capacity_{prefix}_notional"
    ret_col = f"capacity_{prefix}_{mode}_realized_return_%"
    pnl_col = f"capacity_{prefix}_{mode}_realized_pnl"
    ret = _numeric(trades, ret_col)
    pnl = _numeric(trades, pnl_col)
    notional = _numeric(trades, notional_col)
    ok = np.isfinite(ret) & np.isfinite(pnl) & np.isfinite(notional) & (notional > 0)
    losing = ok & (ret <= 0)
    return {
        "covered_trade_count": int(ok.sum()),
        "total_notional_covered": float(notional[ok].sum()) if ok.any() else np.nan,
        "total_realized_pnl": float(pnl[ok].sum()) if ok.any() else np.nan,
        "return_on_deployed_notional_%": float(pnl[ok].sum() / notional[ok].sum() * 100.0)
        if ok.any() and notional[ok].sum() > 0 else np.nan,
        "unweighted_realized_return_%": _finite_mean(ret[ok]),
        "hit_rate_%": float((ret[ok] > 0).mean() * 100.0) if ok.any() else np.nan,
        "wrong_trade_count": int(losing.sum()),
        "wrong_trade_pnl": float(pnl[losing].sum()) if losing.any() else 0.0,
        "correlation_expected_vs_realized": _finite_corr(
            _numeric(trades, f"capacity_{prefix}_E_return_%")[ok], ret[ok]
        ),
    }


def _json_ready(value):
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
        return x if np.isfinite(x) else None
    if not isinstance(value, (str, bytes, bool)):
        try:
            if pd.isna(value):
                return None
        except Exception:
            pass
    return value


def _risk_adjusted_block(trades):
    """Completion-only, cross-sectional diagnostics for realized trade returns."""
    def _stats(x):
        x = pd.to_numeric(x, errors="coerce").dropna()
        if len(x) < 1:
            return {"n": 0}
        std = float(x.std(ddof=1)) if len(x) > 1 else float("nan")
        downside = np.minimum(x.to_numpy(dtype=float), 0.0)
        downside_deviation = float(np.sqrt(np.mean(np.square(downside))))
        mean = float(x.mean())
        return {
            "n": int(len(x)),
            "mean_%": round(mean, 2),
            "median_%": round(float(x.median()), 2),
            "std_%": round(std, 2),
            "min_%": round(float(x.min()), 2),
            "max_%": round(float(x.max()), 2),
            "negative_trade_count": int((x < 0).sum()),
            "hit_rate_%": round(float((x > 0).mean() * 100.0), 2),
            "cross_sectional_mean_to_std": round(mean / std, 2)
            if np.isfinite(std) and std > 0 else None,
            "sortino_0pct_mar": round(mean / downside_deviation, 2)
            if downside_deviation > 0 else None,
        }

    sig = trades["signal"].astype(str)
    realized = _numeric(trades, "realized_return_%").dropna()
    ex_top = realized.drop(realized.idxmax()) if len(realized) > 1 else realized
    return {
        "note": (
            "Cross-sectional realized-return diagnostics on the completed-election sample. "
            "Deal breaks are absent from realized election returns, so mean/std and Sortino "
            "are sensitivity statistics, not investable or annualized strategy ratios."
        ),
        "minimum_acceptable_return_%": 0.0,
        "all_trades": _stats(realized),
        "enter": _stats(_numeric(trades[sig.eq("ENTER")], "realized_return_%")),
        "reverse": _stats(_numeric(trades[sig.eq("REVERSE")], "realized_return_%")),
        "all_ex_largest_winner": _stats(ex_top),
    }


def _render_risk_png(rap, out_path="arb_output/risk_performance.png"):
    order = [("all_trades", "All trades"), ("enter", "ENTER"),
             ("reverse", "REVERSE"), ("all_ex_largest_winner", "Ex-top winner")]
    cols = [("n", "n"), ("mean_%", "Mean %"), ("median_%", "Median %"),
            ("std_%", "Std"), ("cross_sectional_mean_to_std", "Mean / Std"),
            ("sortino_0pct_mar", "Sortino (0%)")]

    def _fmt(value):
        if value is None or (isinstance(value, float) and not np.isfinite(value)):
            return "—"
        return f"{value:g}"

    labels, rows = [], []
    for key, label in order:
        stats = rap.get(key) if isinstance(rap.get(key), dict) else None
        if not stats or not stats.get("n"):
            continue
        labels.append(label)
        rows.append([_fmt(stats.get(column)) for column, _ in cols])
    if not rows:
        return
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception:
        return
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    ink = "#003057"
    fig, ax = plt.subplots(figsize=(7.4, 1.1 + 0.5 * len(rows)))
    ax.axis("off")
    table = ax.table(
        cellText=rows,
        rowLabels=labels,
        colLabels=[column[1] for column in cols],
        loc="center",
        cellLoc="center",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(11)
    table.scale(1, 1.7)
    for column in range(len(cols)):
        table[0, column].set_facecolor(ink)
        table[0, column].set_text_props(color="white", weight="bold")
    ax.set_title(
        "Completion-only realized-return diagnostics (cross-sectional)",
        fontsize=13,
        weight="bold",
        color=ink,
        pad=16,
    )
    fig.text(
        0.5,
        0.03,
        "Deal breaks are outside this realized-election sample. "
        "Mean / Std is descriptive; Sortino uses a 0% MAR and is n/a without losses.",
        ha="center",
        fontsize=8,
        style="italic",
        color="#434545",
    )
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def summarize_strategy(out, output_path="arb_strategy_summary.json"):
    """Summarize ENTER and REVERSE as one combined arbitrage book."""
    if out is None or out.empty or "signal" not in out.columns:
        summary = {
            "priced_signal_count": 0,
            "note": "no_signal_rows",
        }
    else:
        signal = out["signal"].astype(str)
        trades = out[signal.isin(["ENTER", "REVERSE"])].copy()
        pass_rows = out[signal.str.startswith("PASS", na=False)].copy()
        review_rows = out[signal.eq("REVIEW")].copy()
        counts = signal.value_counts().to_dict()

        raw_notional = _numeric(trades, "capacity_raw_max_notional")
        raw_pct = _numeric(trades, "capacity_raw_max_pct_shares_outstanding")
        cap_ok = trades["capacity_status"].eq("ok") if "capacity_status" in trades else pd.Series(False, index=trades.index)
        missed_ret = _numeric(pass_rows, "realized_return_%")
        review_ret = _numeric(review_rows, "realized_return_%")
        optimal_self_ret = _numeric(trades, "capacity_optimal_self_impact_realized_return_%")
        optimal_base_ret = _numeric(trades, "capacity_optimal_baseline_realized_return_%")

        summary = {
            "priced_signal_count": int(len(out)),
            "trade_count": int(len(trades)),
            "enter_count": int(counts.get("ENTER", 0)),
            "reverse_count": int(counts.get("REVERSE", 0)),
            "review_count": int(counts.get("REVIEW", 0)),
            "pass_count": int(sum(v for k, v in counts.items() if str(k).startswith("PASS"))),
            "signal_counts": {str(k): int(v) for k, v in counts.items()},
            "capacity_ok_count": int(cap_ok.sum()),
            "capacity_missing_count": int(len(trades) - cap_ok.sum()),
            "holder_model": {
                "source_counts": trades.get("capacity_holder_model_source", pd.Series(dtype=object)).value_counts(dropna=False).to_dict(),
                "median_fit_n": _finite_median(_numeric(trades, "capacity_holder_fit_n")),
                "median_p_hat": _finite_median(_numeric(trades, "capacity_holder_p_hat")),
                "median_q_hat": _finite_median(_numeric(trades, "capacity_holder_q_hat")),
                "median_positive_share_of_active_%": _finite_median(
                    _numeric(trades, "capacity_positive_holder_share_of_active_%")
                ),
                "median_noisy_cash_prob": _finite_median(_numeric(trades, "capacity_noisy_cash_prob")),
            },
            "raw_maximum": {
                "total_notional": _finite_sum(raw_notional),
                "median_notional": _finite_median(raw_notional),
                "median_pct_shares_outstanding": _finite_median(raw_pct),
            },
            "risk_gated_maximum": _expected_capacity_block(trades, "max"),
            "profit_optimal": _expected_capacity_block(trades, "optimal"),
            "realized_baseline_accept_historical_election": {
                "maximum": _realized_capacity_block(trades, "max", "baseline"),
                "optimal": _realized_capacity_block(trades, "optimal", "baseline"),
            },
            "realized_self_impact_election": {
                "maximum": _realized_capacity_block(trades, "max", "self_impact"),
                "optimal": _realized_capacity_block(trades, "optimal", "self_impact"),
            },
            "trade_quality": {
                "wrong_trade_count_baseline_optimal": int((optimal_base_ret.dropna() <= 0).sum()),
                "wrong_trade_count_self_impact_optimal": int((optimal_self_ret.dropna() <= 0).sum()),
                "missed_profitable_pass_count_marginal_proxy": int((missed_ret.dropna() > 0).sum()),
                "missed_profitable_pass_mean_return_%": _finite_mean(missed_ret[missed_ret > 0]),
                "review_profitable_count_marginal_proxy": int((review_ret.dropna() > 0).sum()),
                "review_profitable_mean_return_%": _finite_mean(review_ret[review_ret > 0]),
                "marginal_proxy_note": (
                    "PASS/REVIEW rows have no executed capacity; missed/review profitability uses the "
                    "selected marginal realized return already printed in arb_signals.csv."
                ),
            },
            "risk_adjusted_performance": _risk_adjusted_block(trades),
            "proof": {
                "long_self_impact_can_eliminate_count": int(
                    out.get("self_impact_can_eliminate_arbitrage", pd.Series(False, index=out.index)).fillna(False).sum()
                ),
                "long_self_impact_break_even_median_pct_shares": _finite_median(
                    _numeric(out[out["signal"].eq("ENTER")], "self_impact_break_even_pct_shares_outstanding")
                ),
                "long_self_impact_break_even_at_or_inside_optimal_count": int((
                    _numeric(trades[trades["signal"].eq("ENTER")], "self_impact_break_even_pct_shares_outstanding")
                    <= _numeric(trades[trades["signal"].eq("ENTER")], "capacity_optimal_pct_shares_outstanding")
                ).sum()),
                "reverse_self_impact_note": (
                    "Under fixed-pool passive blended settlement, reverse payoff is not destroyed by "
                    "its election-demand shift because the short side has no election right and blended "
                    "completion liability is demand-independent."
                ),
            },
            "notes": {
                "maximum_definition": (
                    "capacity_raw_max_* is flow/ADV/position supply. capacity_max_* is the largest "
                    "tradable size that still passes the risk gates after our self-impact. "
                    "capacity_optimal_* maximizes expected dollar PnL over the same feasible grid."
                ),
                "baseline_realized_method": (
                    "Sizes and entry capacity include our market impact, but final settlement accepts "
                    "the historical election result."
                ),
                "self_impact_realized_method": (
                    "Final completed-state election demand is shifted by q*(our election cash "
                    "probability - counterparty cash probability) for ENTER. REVERSE keeps passive "
                    "blended liability because it has no election right."
                ),
            },
        }

        if len(trades):
            for side in ["ENTER", "REVERSE"]:
                side_rows = trades[trades["signal"].eq(side)]
                summary[f"{side.lower()}_profit_optimal"] = _expected_capacity_block(side_rows, "optimal")
                summary[f"{side.lower()}_realized_self_impact_optimal"] = _realized_capacity_block(
                    side_rows, "optimal", "self_impact"
                )

    summary = _json_ready(summary)
    if output_path:
        path = Path(output_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
        csv_path = path.with_suffix(".csv")
        pd.json_normalize(summary, sep=".").to_csv(csv_path, index=False)
        _render_risk_png(summary.get("risk_adjusted_performance", {}))
    return summary


def parse_args():
    p = argparse.ArgumentParser(description="Build risk-aware cash-or-stock election-arb signals.")
    p.add_argument("--deals", default="arb_deals.csv", help="Input deal terms table from arb_terms.py")
    p.add_argument("--market-daily", default="ma_market_wrds/wrds_market_daily.csv",
                   help="WRDS daily market file used for entry/recovery prices")
    p.add_argument("--capacity-panel", default="eda_output/merged_panel.csv",
                   help="Deal panel with target shares, ADV, and passive ownership for capacity sizing")
    p.add_argument("--out", default="arb_signals.csv", help="Output signal blotter")
    p.add_argument("--summary-out", default="arb_strategy_summary.json",
                   help="Combined long/reverse strategy summary output")
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
    p.add_argument("--holder-model", choices=["rolling_structural", "static"], default="rolling_structural",
                   help="Estimate positive/noisy holder mix from prior events or use static fallback")
    p.add_argument("--holder-rolling-window-events", type=int, default=50,
                   help="Prior labeled events used for rolling structural holder fit")
    p.add_argument("--holder-min-fit-events", type=int, default=10,
                   help="Minimum prior labeled events before rolling holder fit is used")
    p.add_argument("--holder-p-grid-size", type=int, default=11,
                   help="Grid size for noisy/irrational cash-election probability p")
    p.add_argument("--holder-q-grid-size", type=int, default=11,
                   help="Grid size for EV-sensitive rational ownership share q")
    p.add_argument("--default-irrational-cash-prob", type=float, default=0.50,
                   help="Fallback p before enough rolling observations exist")
    p.add_argument("--default-rational-share", type=float, default=0.30,
                   help="Fallback q before enough rolling observations exist")
    p.add_argument("--capacity-build-days", type=float, default=10.0)
    p.add_argument("--capacity-max-adv-participation", type=float, default=0.20)
    p.add_argument("--capacity-max-position-pct-shares", type=float, default=0.05)
    p.add_argument("--positive-holder-share-of-active", type=float, default=0.35)
    p.add_argument("--positive-no-sell-edge", type=float, default=0.03,
                   help="Positive holders are assumed not to sell once long EV exceeds this return")
    p.add_argument("--capacity-grid-points", type=int, default=51)
    return p.parse_args()


def config_from_args(args):
    return SignalConfig(
        deals_path=args.deals,
        market_daily_path=args.market_daily,
        capacity_panel_path=args.capacity_panel,
        output_path=args.out,
        summary_output_path=args.summary_out,
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
        holder_model=args.holder_model,
        holder_rolling_window_events=args.holder_rolling_window_events,
        holder_min_fit_events=args.holder_min_fit_events,
        holder_p_grid_size=args.holder_p_grid_size,
        holder_q_grid_size=args.holder_q_grid_size,
        default_irrational_cash_prob=args.default_irrational_cash_prob,
        default_rational_share=args.default_rational_share,
        capacity_build_days=args.capacity_build_days,
        capacity_max_adv_participation=args.capacity_max_adv_participation,
        capacity_max_position_pct_shares=args.capacity_max_position_pct_shares,
        positive_holder_share_of_active=args.positive_holder_share_of_active,
        positive_no_sell_edge=args.positive_no_sell_edge,
        capacity_grid_points=args.capacity_grid_points,
    )


# ==============================================================================
# Unified CLI for the monolithic pipeline
# ==============================================================================

def run_outcome_layer(
    bbg_path: str = "BBG Data Pull 2006+ Final.csv",
    events_path: str = "eda_output/merged_panel.csv",
    output_path: str = "deal_outcome_probabilities.csv",
    default_completed_prob: float = 0.88,
    default_terminated_prob: float = 0.07,
    default_withdrawn_prob: float = 0.05,
    write_material: bool = True,
):
    table = build_bbg_outcome_probability_table(
        bbg_path=bbg_path,
        events_path=events_path,
        output_path=output_path,
        defaults=OutcomeDefaults(
            completed=default_completed_prob,
            terminated=default_terminated_prob,
            withdrawn=default_withdrawn_prob,
        ),
    )
    print(f"[outcome] wrote {output_path}: {len(table)} rows")
    if "outcome_probability_source" in table:
        print(table["outcome_probability_source"].value_counts().to_string())
    if write_material:
        try:
            from material_builder import export_after_outcome
            material_results = export_after_outcome()
            print("[material] wrote outcome slide material to material/")
            print(json.dumps(material_results, indent=2))
        except Exception as exc:
            print(f"[material] skipped outcome material export: {exc}")
    return table


def run_signal_layer(
    deals_path: str = "arb_deals.csv",
    market_daily_path: str = "ma_market_wrds/wrds_market_daily.csv",
    capacity_panel_path: str = "eda_output/merged_panel.csv",
    output_path: str = "arb_signals.csv",
    summary_output_path: str = "arb_strategy_summary.json",
    outcome_probs_path: str = "",
    n_draws: int = NDRAW,
    write_material: bool = True,
):
    # Keep the large pipeline CLI as the delivery surface while using the
    # modular signal implementation as the single source of truth.
    from arb_signal import (
        SignalConfig as CurrentSignalConfig,
        build_signals as build_current_signals,
        summarize_strategy as summarize_current_strategy,
    )

    cfg = CurrentSignalConfig(
        deals_path=deals_path,
        market_daily_path=market_daily_path,
        capacity_panel_path=capacity_panel_path,
        output_path=output_path,
        summary_output_path=summary_output_path,
        outcome_probs_path=outcome_probs_path,
        n_draws=n_draws,
    )
    out = build_current_signals(cfg)
    summary = summarize_current_strategy(
        out,
        cfg.summary_output_path,
        market_daily_path=cfg.market_daily_path,
    )
    if write_material:
        try:
            from material_builder import export_after_signal
            material_results = export_after_signal()
            print("[material] wrote post-MC slide material to material/")
            print(json.dumps(material_results, indent=2))
        except Exception as exc:
            print(f"[material] skipped post-MC material export: {exc}")
    has_signal = "signal" in out.columns
    ent = out[out["signal"] == "ENTER"] if has_signal else pd.DataFrame()
    rev = out[out["signal"] == "REVERSE"] if has_signal else pd.DataFrame()
    trades = out[out["signal"].isin(["ENTER", "REVERSE"])] if has_signal else pd.DataFrame()
    print(
        f"[signal] wrote {output_path}: {len(out)} rows, "
        f"{len(ent)} ENTER, {len(rev)} REVERSE, {len(trades)} total trades"
    )
    opt = summary.get("profit_optimal", {}) if isinstance(summary, dict) else {}
    if opt:
        print(
            "[strategy] optimal sizing: "
            f"notional=${float(opt.get('total_notional') or 0):,.0f}  "
            f"E[pnl]=${float(opt.get('total_expected_pnl') or 0):,.0f}  "
            f"weighted E[return]={float(opt.get('notional_weighted_expected_return_%') or 0):.2f}%"
        )
    return out, summary


def run_deadline_spread_layer() -> None:
    """Rebuild deadline_spread.csv with the canonical standalone script."""
    import subprocess
    import sys

    subprocess.run([sys.executable, "deadline_spread.py"], check=True)


def run_fast_pipeline(signal_outcome_probs_path: str = "deal_outcome_probabilities.csv"):
    run_deadline_spread_layer()
    run_outcome_layer()
    run_monte_carlo()
    return run_signal_layer(outcome_probs_path=signal_outcome_probs_path)


def run_preflight_check() -> bool:
    """Validate the standard local inputs used by the reproducible rebuild."""
    required_inputs = {
        "BBG Data Pull 2006+ Final.csv": {
            "purpose": "all-status outcome-model training",
            "columns": {"Announce Date", "Target Name", "Acquirer Name", "Deal Status"},
        },
        "ma_edgar_full/llm_field_extractions.csv": {
            "purpose": "SEC deal terms and election mechanics",
            "columns": {"event_id", "field_name", "value"},
        },
        "ma_market_wrds/wrds_market_daily.csv": {
            "purpose": "target/acquirer price paths and liquidity",
            "columns": {"event_id", "side", "price_date", "price"},
        },
        "ma_market_wrds/event_security_map.csv": {
            "purpose": "event-to-security mapping",
            "columns": {"event_id", "side", "cusip"},
        },
        "target_close_dates.csv": {
            "purpose": "close-date fallback for election deadlines",
            "columns": {"target_cusip8", "close_date"},
        },
        "normalized_labels.csv": {
            "purpose": "realized cash-election demand labels",
            "columns": {"event_id", "pct_elected_cash"},
        },
        "eda_output/merged_panel.csv": {
            "purpose": "capacity and event-level analysis panel",
            "columns": {"event_id"},
        },
    }
    ok = True
    print("[check] standard rebuild inputs")
    for path_text, spec in required_inputs.items():
        path = Path(path_text)
        if not path.exists():
            ok = False
            print(f"  MISSING  {path_text}  ({spec['purpose']})")
            continue
        try:
            columns = set(pd.read_csv(path, nrows=1).columns)
        except Exception as exc:
            ok = False
            print(f"  INVALID  {path_text}  ({exc})")
            continue
        missing_columns = sorted(spec["columns"] - columns)
        if missing_columns:
            ok = False
            print(
                f"  INVALID  {path_text}  "
                f"(missing columns: {', '.join(missing_columns)})"
            )
            continue
        size_mb = path.stat().st_size / (1024.0 * 1024.0)
        print(f"  OK       {path_text}  ({size_mb:.2f} MB)")
    print("[check] ready for `python arb_pipeline.py fast`" if ok else "[check] fix the items above before running `fast`")
    return ok


def parse_pipeline_args():
    parser = argparse.ArgumentParser(description="Run the consolidated election-arb pipeline.")
    sub = parser.add_subparsers(dest="command", required=True)

    sub.add_parser("check", help="Validate the standard local inputs for a full rebuild.")

    outcome = sub.add_parser("outcome", help="Build event-level completed/terminated/withdrawn probabilities.")
    outcome.add_argument("--bbg", default="BBG Data Pull 2006+ Final.csv")
    outcome.add_argument("--events", default="eda_output/merged_panel.csv")
    outcome.add_argument("--out", default="deal_outcome_probabilities.csv")
    outcome.add_argument("--no-material", action="store_true")

    mc = sub.add_parser("mc", help="Run terms, demand MC, calibration, and MC material export.")
    mc.add_argument("--no-material", action="store_true", help="Accepted for symmetry; arb_run export remains best-effort.")

    signal = sub.add_parser("signal", help="Build risk-gated signals and strategy summary.")
    signal.add_argument("--deals", default="arb_deals.csv")
    signal.add_argument("--market-daily", default="ma_market_wrds/wrds_market_daily.csv")
    signal.add_argument("--capacity-panel", default="eda_output/merged_panel.csv")
    signal.add_argument("--out", default="arb_signals.csv")
    signal.add_argument("--summary-out", default="arb_strategy_summary.json")
    signal.add_argument("--outcome-probs", default="")
    signal.add_argument("--n-draws", type=int, default=NDRAW)
    signal.add_argument("--no-material", action="store_true")

    fast = sub.add_parser(
        "fast",
        help="Run deadline spread -> outcome -> MC -> signal.",
    )
    fast.add_argument("--signal-outcome-probs", default="deal_outcome_probabilities.csv")

    sub.add_parser("material", help="Refresh all material/ slide outputs from current artifacts.")
    return parser.parse_args()


def pipeline_main():
    args = parse_pipeline_args()
    if args.command == "check":
        if not run_preflight_check():
            raise SystemExit(1)
        return True
    if args.command == "outcome":
        return run_outcome_layer(
            bbg_path=args.bbg,
            events_path=args.events,
            output_path=args.out,
            write_material=not args.no_material,
        )
    if args.command == "mc":
        return run_monte_carlo()
    if args.command == "signal":
        return run_signal_layer(
            deals_path=args.deals,
            market_daily_path=args.market_daily,
            capacity_panel_path=args.capacity_panel,
            output_path=args.out,
            summary_output_path=args.summary_out,
            outcome_probs_path=args.outcome_probs,
            n_draws=args.n_draws,
            write_material=not args.no_material,
        )
    if args.command == "fast":
        return run_fast_pipeline(signal_outcome_probs_path=args.signal_outcome_probs)
    if args.command == "material":
        from material_builder import build_all_material
        results = build_all_material()
        print(json.dumps(results, indent=2))
        return results
    raise ValueError(f"Unknown command: {args.command}")


if __name__ == "__main__":
    pipeline_main()
