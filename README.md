# Cash-or-Stock Election Arbitrage

This repository contains the Stage 0-10 research code for cash-or-stock merger
elections. The workflow has two explicit execution boundaries:

1. **Credentialed data build (Stage 0-1):** starts from a manually supplied
   Bloomberg export, then uses separate SEC, WRDS, and LLM scripts plus manual
   identifier review.
2. **Offline analytics rebuild (Stage 2-10):** starts after the seven standard
   local inputs exist and is orchestrated by `arb_pipeline.py`.

A fresh clone does not contain the licensed Bloomberg/WRDS/SEC-derived inputs
and therefore cannot run Stage 2-10 until those files are restored or rebuilt.

The repository intentionally tracks only:

- current source code;
- lightweight configuration and reviewed public mappings;
- the normalized public election-demand labels;
- tests and current documentation.

Licensed/local source data and reproducible generated outputs are gitignored.
Running the pipeline creates `arb_output/`, `material/`, and the root-level
working CSV/JSON files.

## Current Reference Run

The latest local licensed-data run produces:

- 73 disclosed election-demand labels;
- 32 Monte Carlo-ready deals;
- temporal outcome-model Brier score 0.307 versus 0.317 prior-only;
- 88 priced signals: 10 ENTER, 10 REVERSE, 20 REVIEW, 48 PASS-family;
- $621.4 million optimal target-leg notional;
- $39.9 million expected P&L, or 6.42% of opportunity notional;
- 16 direct realized-settlement validations;
- 18 of 20 reconstructed daily trade paths;
- active-position Sharpe 1.08 and Sortino 3.07;
- full-calendar Sharpe 0.45 and Sortino 1.27.

These are reference diagnostics, not committed data artifacts. Rebuild them
from the standard local inputs described below.

## Quick Start

```bash
git clone https://github.com/alviny800/JPM-Project-3ii.git
cd JPM-Project-3ii

python3 -m venv .venv
source .venv/bin/activate
python3 -m pip install --upgrade pip
python3 -m pip install -r requirements.txt

# Restore or build the seven files listed under "Offline Analytics Inputs".
python3 arb_pipeline.py check
python3 arb_pipeline.py fast
```

`arb_pipeline.py` is the orchestration entry point for the offline Stage 2-10
chain. It does not download Bloomberg, SEC, WRDS, or LLM data. `fast` runs each
offline analytics layer once and regenerates the presentation material at the
end:

```text
deadline terms
  -> all-status outcome probabilities
  -> demand calibration and proration Monte Carlo
  -> risk gates and ENTER/REVERSE decisions
  -> holder structure and capacity
  -> realized settlement diagnostics
  -> historical daily strategy results
  -> material/
```

## Stage 0-10 Map

| Stage | Layer | Purpose | Current source | Main output |
|---:|---|---|---|---|
| 0 | Universe and identifiers | Review the Bloomberg universe and resolve historical CUSIP/ticker identities | `prepare_input_identifiers.py`, `cik_resolution.py` | reviewed election universe, CIK/CUSIP audits |
| 1 | SEC, market, and labels | Extract contractual terms and realized elections; download prices/ownership; normalize labels | `download_ma_edgar_files.py`, `download_wrds_market_data.py`, `download_ownership_etf_data.py`, `normalize_labels.py`, `election_arb_eda.py` | local SEC/WRDS files, `normalized_labels.csv`, merged panel |
| 2 | Demand distribution | Fit pooled Beta election demand and test calibration | `arb_mc.py`, `arb_backtest.py` | demand model and PIT diagnostics |
| 3 | Proration Monte Carlo | Apply deal terms and three-state outcome probabilities to simulated demand | `arb_terms.py`, `arb_run.py` | `arb_deals.csv`, `arb_output/` |
| 4 | Deal outcome | Estimate completed/terminated/withdrawn probabilities with temporally tuned Naive Bayes | `arb_outcome.py` | `deal_outcome_probabilities.csv` |
| 5 | Payoff overlay | Convert completion, termination, and withdrawal into return distributions | `arb_mc.py`, `arb_signal.py` | state-weighted payoff statistics |
| 6 | Risk-gated decisions | Choose ENTER, REVERSE, REVIEW, or PASS | `arb_signal.py` | `arb_signals.csv` |
| 7 | Holder structure | Estimate rolling noisy/EV-sensitive holder parameters | `structural_election_model.py`, `arb_capacity.py` | event-level p/q estimates |
| 8 | Capacity and self-impact | Apply supply, borrow, ADV, position, and proration-impact limits | `arb_capacity.py`, `arb_signal.py` | optimal/max capacity and expected P&L |
| 9 | Settlement validation | Compare expected payoffs with directly reconstructable completed settlements | `arb_signal.py` | realized-return diagnostics |
| 10 | Historical strategy | Reconstruct capacity-weighted daily target/acquirer hedged returns | `arb_signal.py` | daily/event return files and Sharpe/Sortino |

The conceptual stage order and execution order differ slightly: Stage 4
probabilities are generated before the Stage 3 portfolio overlay because the
Monte Carlo consumes those probabilities.

## Offline Analytics Inputs

`python3 arb_pipeline.py check` validates these exact paths and required columns:

| Path | Purpose | How it is produced |
|---|---|---|
| `BBG Data Pull 2006+ Final.csv` | all-status outcome-model training data | licensed Bloomberg export |
| `ma_edgar_full/llm_field_extractions.csv` | contractual terms, election deadlines, realized disclosures | `download_ma_edgar_files.py` |
| `ma_market_wrds/wrds_market_daily.csv` | target/acquirer prices, volume, and daily paths | `download_wrds_market_data.py`, then `fix_acquirer_prices.py` |
| `ma_market_wrds/event_security_map.csv` | event-to-security identifiers | `download_wrds_market_data.py` |
| `target_close_dates.csv` | close-date fallback | `build_close_dates.py` |
| `normalized_labels.csv` | clean cash-election demand labels | `normalize_labels.py` |
| `eda_output/merged_panel.csv` | capacity and diagnostic event panel | `election_arb_eda.py` |

All licensed or large inputs are local and gitignored. `normalized_labels.csv`
is the only small, reviewed analytic input committed to the repository.

## Normal Offline Rebuild

Use this path for code changes, model reruns, and refreshed presentation
material. It does not call Bloomberg, WRDS, SEC, or an LLM.

### 1. Validate inputs

```bash
python3 arb_pipeline.py check
```

The command fails before model work if a standard file or required column is
missing.

### 2. Run all offline analytics

```bash
python3 arb_pipeline.py fast
```

The command performs:

1. `deadline_spread.py`;
2. temporally tuned three-state outcome probabilities;
3. deal-term assembly, demand calibration, and proration Monte Carlo;
4. risk gates, trade direction, holder structure, capacity, and self-impact;
5. settlement and daily-path strategy diagnostics;
6. one final `material/` export.

### 3. Run a layer in isolation

```bash
python3 arb_pipeline.py outcome
python3 arb_pipeline.py mc
python3 arb_pipeline.py signal
python3 arb_pipeline.py material
```

Layer commands are for debugging. A normal rebuild should use `fast` so paths
and execution order remain consistent.

## Credentialed Data Build (Stage 0-1)

Run this only when the Bloomberg universe, SEC extraction, CRSP market data, or
ownership data must be replaced. It requires licensed data, credentials, API
access, and manual audit. This section is a sequence of commands, not an
unattended `arb_pipeline.py` mode.

### 1. Credentials

```bash
export WRDS_USERNAME="your_wrds_username"
export ANTHROPIC_API_KEY="your_anthropic_api_key"
export SEC_USER_AGENT="Your Name your.email@example.com"
```

Never store credentials in the repository.

### 2. Bloomberg universe and historical identifiers

Place the Bloomberg export at:

```text
BBG Data Pull 2006+ Final.csv
```

Download the CRSP security-name history:

```bash
python3 prepare_input_identifiers.py backfill-cusips \
  --input "BBG Data Pull 2006+ Final.csv" \
  --dump-stocknames \
  --wrds-username "$WRDS_USERNAME"
```

Build the reviewed election subset:

```bash
python3 prepare_input_identifiers.py backfill-cusips \
  --input "BBG Data Pull 2006+ Final.csv" \
  --out-dir cusip_backfill \
  --election-only
```

Review the generated CUSIP audit and unresolved rows. The downstream reviewed
universe is stored locally as `US_election_deals_for_analysis.csv`.

```bash
python3 prepare_input_identifiers.py clean-tickers \
  --input US_election_deals_for_analysis.csv
```

### 3. SEC CIK audit

```bash
python3 cik_resolution.py audit --user-agent "$SEC_USER_AGENT"
python3 cik_resolution.py build-overrides --user-agent "$SEC_USER_AGENT"
```

Review candidates before changing the committed `cik_manual_overrides.csv`.

### 4. Close-date fallback

```bash
python3 build_close_dates.py \
  --input US_election_deals_for_analysis.csv \
  --stocknames stocknames_cache.csv \
  --out target_close_dates.csv
```

### 5. SEC download and LLM field extraction

```bash
python3 download_ma_edgar_files.py \
  --input US_election_deals_for_analysis.csv \
  --output-dir ma_edgar_full \
  --user-agent "$SEC_USER_AGENT" \
  --cik-overrides cik_manual_overrides.csv \
  --close-dates target_close_dates.csv \
  --field-specs field_specs.json \
  --llm-stage batch \
  --max-batch-cost-usd 80
```

The canonical output is
`ma_edgar_full/llm_field_extractions.csv`. Raw filings, caches, request payloads,
and API response logs remain local.

### 6. WRDS market and ownership data

```bash
python3 download_wrds_market_data.py \
  --input US_election_deals_for_analysis.csv \
  --output-dir ma_market_wrds \
  --wrds-username "$WRDS_USERNAME" \
  --llm-extractions ma_edgar_full/llm_field_extractions.csv

python3 download_ownership_etf_data.py \
  --input US_election_deals_for_analysis.csv \
  --output-dir ma_ownership_wrds \
  --provider wrds \
  --wrds-username "$WRDS_USERNAME"
```

Apply the reviewed historical security overrides:

```bash
python3 fix_acquirer_prices.py
```

The overrides correct the Isle/MTR acquirer collision and the Sirius target
collision.

### 7. Normalize election-demand labels

```bash
python3 normalize_labels.py \
  --extractions ma_edgar_full/llm_field_extractions.csv \
  --out normalized_labels.csv
```

This separates the fraction shareholders **elected** from the shares they
ultimately **received after proration**.

### 8. Build the merged event panel

```bash
python3 election_arb_eda.py \
  --extractions ma_edgar_full/llm_field_extractions.csv \
  --ownership ma_ownership_wrds/ownership_mix_by_event.csv \
  --market ma_market_wrds/event_market_features.csv \
  --normalized normalized_labels.csv \
  --output-dir eda_output
```

Finish with the normal offline rebuild:

```bash
python3 arb_pipeline.py check
python3 arb_pipeline.py fast
```

## Output Lifecycle

Generated artifacts are intentionally not tracked:

| Path | Contents |
|---|---|
| `arb_output/` | MC summary, calibration/edge figures, and native model outputs |
| `material/` | Stage 2-10 slide-ready CSV/JSON/PNG exports and manifest |
| `arb_deals.csv` | assembled deterministic deal terms |
| `deal_outcome_probabilities.csv` | Stage 4 event probabilities |
| `arb_signals.csv` | complete risk-gated blotter |
| `arb_strategy_summary.json` | strategy, capacity, realized, and historical metrics |
| `arb_strategy_daily_returns.csv` | portfolio daily return series |
| `arb_strategy_event_daily_returns.csv` | event-level daily return series |

Delete any of these files safely and rerun `python3 arb_pipeline.py fast`.

## Code Layout

```text
arb_pipeline.py              canonical offline Stage 2-10 CLI and run order

prepare_input_identifiers.py Stage 0 Bloomberg/CUSIP preparation
cik_resolution.py            Stage 0 SEC identity audit
build_close_dates.py         Stage 1 close-date fallback
download_ma_edgar_files.py   Stage 1 SEC retrieval and LLM extraction
download_wrds_market_data.py Stage 1 market/security data
download_ownership_etf_data.py Stage 1 ownership/ETF data
event_csv_adapter.py         shared event-file normalization
fix_acquirer_prices.py       reviewed historical price corrections
normalize_labels.py          election-demand label normalization
election_arb_eda.py          merged event panel

arb_terms.py                 deterministic deal-term assembly
arb_mc.py                    demand and proration engine
arb_backtest.py              calibration and realized-edge tests
arb_run.py                   Stage 2-3 MC driver
arb_outcome.py               Stage 4 temporal Naive Bayes
structural_election_model.py Stage 7 rolling holder model
deal_outcome_model.py        shared outcome helpers for structural model
arb_capacity.py              Stage 7-8 capacity helpers
arb_signal.py                Stage 5-10 decisions and historical P&L
material_builder.py          disposable presentation-output exporter
```

The stage modules are the implementation and unit-test surface.
`arb_pipeline.py` contains no duplicated model implementation.

## Tests

```bash
python3 -m unittest -v test_outcome_tuning.py test_strategy_history.py
PYTHONPYCACHEPREFIX=/tmp/jpm_pycache python3 -m py_compile *.py
python3 arb_pipeline.py check
```

For an integration test with the standard local inputs:

```bash
python3 arb_pipeline.py fast
```

## Interpreting the Results

- `6.42%` is expected P&L divided by aggregate opportunity notional across the
  twenty selected trades. It is not a time-series cumulative return.
- `96.67%` is the additive sum of capacity-weighted daily returns across
  sequential opportunities. Capital is reused.
- Sixteen trades have directly reconstructable terminal settlement returns.
- Eighteen trades have enough common price history for daily reconstruction.
- Cross-sectional terminal Sortino is undefined because none of the sixteen
  terminal returns is negative.
- Daily Sortino is defined because profitable terminal trades still have
  negative mark-to-market days.
- Active-position ratios use only days with an open position.
- Full-calendar ratios retain inactive business days as zero returns.
- Completion-only and all-tradable historical results are currently identical
  because the executable reconstructed sample contains no broken deals.
- The three-class outcome model is evaluated with temporal Brier score, log
  loss, balanced accuracy, and macro-F1, not accuracy on the completed-only
  application frame.

## Known Limitations

- Only 73 deals disclose direct usable election demand.
- The all-status outcome model adds modest probability skill, not strong
  terminated/withdrawn classification.
- Historical strategy paths are completion-only and therefore survivorship
  biased.
- Transaction costs, financing, borrow fees, and locate failures are excluded.
- Some capacity flow and holder-behavior rates remain assumptions.
- Private/foreign acquirers and incomplete common price histories remain outside
  the fully hedged historical sample.

## Model Reference

- [ARB_FRAMEWORK.md](ARB_FRAMEWORK.md): model mechanics and validation scope.
