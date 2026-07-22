#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Outcome-probability adapter for the election-arb trade layer.

This module is intentionally an adapter, not a hidden training pipeline.  If a
future all-status deal universe provides completed/terminated/withdrawn model
outputs, the trade layer can consume them here.  If those columns are absent,
the code falls back to explicit scenario defaults and reports that source.
"""
from __future__ import annotations

from dataclasses import dataclass
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


@dataclass(frozen=True)
class OutcomeDefaults:
    completed: float = 0.88
    terminated: float = 0.07
    withdrawn: float = 0.05
    withdrawn_share_of_break: float = 0.35


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
    def __init__(self, rows: pd.DataFrame, defaults: OutcomeDefaults, alpha: float = 1.0) -> None:
        labeled = rows.copy()
        labeled["_outcome_label"] = labeled["Deal Status"].map(normalize_bbg_deal_status)
        labeled = labeled[labeled["_outcome_label"].isin(OUTCOME_STATES)].copy()
        self.fit_n = int(len(labeled))
        self.alpha = float(alpha)
        self.defaults = _normalized_defaults(defaults)
        self.label_counts = {
            state: int((labeled["_outcome_label"] == state).sum())
            for state in OUTCOME_STATES
        }
        prior_total = self.fit_n + self.alpha * len(OUTCOME_STATES)
        self.priors = {
            state: (self.label_counts[state] + self.alpha * self.defaults[state]) / max(prior_total, 1e-12)
            for state in OUTCOME_STATES
        }
        norm = sum(self.priors.values())
        self.priors = {k: v / norm for k, v in self.priors.items()}

        feature_rows = []
        for idx, row in labeled.iterrows():
            feature_rows.append({"_row_idx": idx, "_outcome_label": row["_outcome_label"], **bbg_model_features(row)})
        feats = pd.DataFrame(feature_rows)
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
                    self.numeric_stats[state][feature] = (mean, max(var, 1e-4))

        self.category_values: Dict[str, List[str]] = {}
        self.category_counts: Dict[str, Dict[str, Dict[str, float]]] = {state: {} for state in OUTCOME_STATES}
        for feature in CATEGORICAL_MODEL_FEATURES:
            values = sorted(set(feats[feature].astype(str)) | {"__missing__"}) if feature in feats else ["__missing__"]
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
            score = math.log(max(self.priors.get(state, 0.0), 1e-12))
            for feature in NUMERIC_MODEL_FEATURES:
                value = feats.get(feature)
                if value is None:
                    continue
                mean, var = self.numeric_stats.get(state, {}).get(
                    feature,
                    self.global_numeric_stats.get(feature, (0.0, 1.0)),
                )
                var = max(var, 1e-4)
                score += -0.5 * (math.log(2.0 * math.pi * var) + ((float(value) - mean) ** 2) / var)
            for feature in CATEGORICAL_MODEL_FEATURES:
                value = str(feats.get(feature) or "__missing__")
                values = self.category_values.get(feature, ["__missing__"])
                counts = self.category_counts.get(state, {}).get(feature, {})
                denom = sum(counts.values()) + self.alpha * max(1, len(values))
                score += math.log((counts.get(value, 0.0) + self.alpha) / max(denom, 1e-12))
            scores[state] = score
        max_score = max(scores.values())
        exp_scores = {k: math.exp(v - max_score) for k, v in scores.items()}
        total = sum(exp_scores.values())
        if total <= 0:
            return dict(self.priors)
        return {k: exp_scores[k] / total for k in OUTCOME_STATES}


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
        model = BbgOutcomeNaiveBayes(labeled, defaults)
        default_probs = model.priors
        model_source = "bbg_naive_bayes_full_training"
        train_counts = model.label_counts
        train_n = model.fit_n
    else:
        model = None
        default_probs = _normalized_defaults(defaults)
        model_source = "default_probabilities_insufficient_bbg_training_rows"
        train_counts = {state: 0 for state in OUTCOME_STATES}
        train_n = int(len(labeled))
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


if __name__ == "__main__":
    args = _parse_cli()
    table = build_bbg_outcome_probability_table(
        bbg_path=args.bbg,
        events_path=args.events,
        output_path=args.out,
        defaults=OutcomeDefaults(
            completed=args.default_completed_prob,
            terminated=args.default_terminated_prob,
            withdrawn=args.default_withdrawn_prob,
        ),
    )
    print(f"[outcome] wrote {args.out}: {len(table)} rows")
    print(table["outcome_probability_source"].value_counts().to_string())
