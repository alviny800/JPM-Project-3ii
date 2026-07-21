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
from pathlib import Path
import re
from typing import Any, Dict, Iterable, Optional

import pandas as pd


OUTCOME_STATES = ("completed", "terminated", "withdrawn")

PROBABILITY_COLUMNS = {
    "completed": ("p_completed", "deal_completed_probability", "completed_probability"),
    "terminated": ("p_terminated", "deal_terminated_probability", "terminated_probability"),
    "withdrawn": ("p_withdrawn", "deal_withdrawn_probability", "withdrawn_probability"),
}

BREAK_PROBABILITY_COLUMNS = ("p_break", "deal_break_probability", "break_probability")


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
