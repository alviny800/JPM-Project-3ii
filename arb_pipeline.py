#!/usr/bin/env python3
"""Canonical Stage 0-10 orchestration for the election-arbitrage project.

The implementation lives in the stage modules. This file owns only run order,
standard paths, preflight validation, and the consolidated command-line
interface.

Commands:
  python arb_pipeline.py check
  python arb_pipeline.py outcome
  python arb_pipeline.py mc
  python arb_pipeline.py signal
  python arb_pipeline.py fast
  python arb_pipeline.py material
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
from typing import Any, Optional

import pandas as pd


os.environ.setdefault("MPLCONFIGDIR", "/private/tmp/jpm_mpl_config")

DEFAULT_OUTCOME_PATH = "deal_outcome_probabilities.csv"

STANDARD_INPUTS = {
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


def run_preflight_check() -> bool:
    """Validate every local input required by the normal offline rebuild."""
    ok = True
    print("[check] Stage 0-10 local inputs")
    for path_text, spec in STANDARD_INPUTS.items():
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
        missing = sorted(spec["columns"] - columns)
        if missing:
            ok = False
            print(f"  INVALID  {path_text}  (missing: {', '.join(missing)})")
            continue
        size_mb = path.stat().st_size / (1024.0 * 1024.0)
        print(f"  OK       {path_text}  ({size_mb:.2f} MB)")
    message = (
        "[check] ready for `python3 arb_pipeline.py fast`"
        if ok
        else "[check] repair the items above before running `fast`"
    )
    print(message)
    return ok


def run_deadline_spread_layer() -> None:
    """Rebuild election-deadline prices with the canonical terms script."""
    subprocess.run([sys.executable, "deadline_spread.py"], check=True)


def run_outcome_layer(
    bbg_path: str = "BBG Data Pull 2006+ Final.csv",
    events_path: str = "eda_output/merged_panel.csv",
    output_path: str = DEFAULT_OUTCOME_PATH,
    write_material: bool = True,
) -> pd.DataFrame:
    """Fit the temporally tuned three-state outcome model."""
    from arb_outcome import (
        OutcomeDefaults,
        build_bbg_outcome_probability_table,
    )

    table = build_bbg_outcome_probability_table(
        bbg_path=bbg_path,
        events_path=events_path,
        output_path=output_path,
        defaults=OutcomeDefaults(),
    )
    print(f"[outcome] wrote {output_path}: {len(table)} rows")
    if "outcome_probability_source" in table:
        print(table["outcome_probability_source"].value_counts().to_string())
    if write_material:
        from material_builder import export_after_outcome

        print(json.dumps(export_after_outcome(), indent=2))
    return table


def run_mc_layer(write_material: bool = True) -> None:
    """Build terms, demand calibration, proration MC, and realized-edge tests."""
    import arb_run

    arb_run.main(write_material=write_material)


def run_signal_layer(
    deals_path: str = "arb_deals.csv",
    market_daily_path: str = "ma_market_wrds/wrds_market_daily.csv",
    capacity_panel_path: str = "eda_output/merged_panel.csv",
    output_path: str = "arb_signals.csv",
    summary_output_path: str = "arb_strategy_summary.json",
    outcome_probs_path: str = DEFAULT_OUTCOME_PATH,
    n_draws: Optional[int] = None,
    write_material: bool = True,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Build risk-gated signals, capacity, and historical strategy diagnostics."""
    from arb_signal import (
        NDRAW,
        SignalConfig,
        build_signals,
        summarize_strategy,
    )

    config = SignalConfig(
        deals_path=deals_path,
        market_daily_path=market_daily_path,
        capacity_panel_path=capacity_panel_path,
        output_path=output_path,
        summary_output_path=summary_output_path,
        outcome_probs_path=outcome_probs_path,
        n_draws=NDRAW if n_draws is None else n_draws,
    )
    signals = build_signals(config)
    summary = summarize_strategy(
        signals,
        config.summary_output_path,
        market_daily_path=config.market_daily_path,
    )
    if write_material:
        from material_builder import export_after_signal

        print(json.dumps(export_after_signal(), indent=2))

    counts = signals.get("signal", pd.Series(dtype=str)).value_counts()
    print(
        f"[signal] wrote {output_path}: {len(signals)} rows, "
        f"{int(counts.get('ENTER', 0))} ENTER, "
        f"{int(counts.get('REVERSE', 0))} REVERSE"
    )
    return signals, summary


def run_material_layer() -> list[dict[str, Any]]:
    """Regenerate the complete disposable presentation-output directory."""
    from material_builder import build_all_material

    results = build_all_material()
    print(json.dumps(results, indent=2))
    return results


def run_fast_pipeline(
    outcome_probs_path: str = DEFAULT_OUTCOME_PATH,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Run the complete offline analytics chain exactly once per layer."""
    run_deadline_spread_layer()
    run_outcome_layer(output_path=outcome_probs_path, write_material=False)
    run_mc_layer(write_material=False)
    signals, summary = run_signal_layer(
        outcome_probs_path=outcome_probs_path,
        write_material=False,
    )
    run_material_layer()
    return signals, summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the canonical cash-or-stock election-arbitrage pipeline."
    )
    sub = parser.add_subparsers(dest="command", required=True)

    sub.add_parser("check", help="Validate all local inputs required by Stage 0-10.")

    outcome = sub.add_parser("outcome", help="Build Stage 4 outcome probabilities.")
    outcome.add_argument("--bbg", default="BBG Data Pull 2006+ Final.csv")
    outcome.add_argument("--events", default="eda_output/merged_panel.csv")
    outcome.add_argument("--out", default=DEFAULT_OUTCOME_PATH)
    outcome.add_argument("--no-material", action="store_true")

    mc = sub.add_parser("mc", help="Run demand calibration and proration Monte Carlo.")
    mc.add_argument("--no-material", action="store_true")

    signal = sub.add_parser(
        "signal",
        help="Run Stage 5-10 risk gates, capacity, and strategy history.",
    )
    signal.add_argument("--deals", default="arb_deals.csv")
    signal.add_argument(
        "--market-daily",
        default="ma_market_wrds/wrds_market_daily.csv",
    )
    signal.add_argument("--capacity-panel", default="eda_output/merged_panel.csv")
    signal.add_argument("--out", default="arb_signals.csv")
    signal.add_argument("--summary-out", default="arb_strategy_summary.json")
    signal.add_argument("--outcome-probs", default=DEFAULT_OUTCOME_PATH)
    signal.add_argument("--n-draws", type=int)
    signal.add_argument("--no-material", action="store_true")

    fast = sub.add_parser(
        "fast",
        help="Run deadline terms, outcome, MC, signals, strategy, and material.",
    )
    fast.add_argument("--outcome-probs", default=DEFAULT_OUTCOME_PATH)

    sub.add_parser("material", help="Regenerate disposable slide-ready outputs.")
    return parser.parse_args()


def main() -> Any:
    args = parse_args()
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
        return run_mc_layer(write_material=not args.no_material)
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
        return run_fast_pipeline(outcome_probs_path=args.outcome_probs)
    if args.command == "material":
        return run_material_layer()
    raise ValueError(f"unknown command: {args.command}")


if __name__ == "__main__":
    main()
