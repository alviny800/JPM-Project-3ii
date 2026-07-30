# Cash-or-Stock Election Arbitrage

This repository models cash-or-stock merger elections from raw deal terms through
trade sizing and historical strategy diagnostics.

In these deals, target shareholders choose cash or stock, but aggregate cash and
stock pools are capped. The average holder receives the fixed blended
consideration. A holder who elects the richer, under-subscribed side can receive
more than that average. The project estimates that proration-capture edge, applies
deal-outcome risk, converts it into ENTER/REVERSE/PASS decisions, sizes the trades,
and reconstructs daily hedged P&L.

Current reference run:

- 73 disclosed election-demand labels
- 32 Monte Carlo-ready deals
- 88 priced signals: 10 ENTER, 10 REVERSE, 20 REVIEW, 48 PASS-family
- $621.4 million aggregate target-leg opportunity notional
- $39.9 million ex-ante expected P&L, or 6.42% of opportunity notional
- 18/20 reconstructed daily paths
- Active-position Sharpe 1.08 and Sortino 3.07
- Full-calendar Sharpe 0.45 and Sortino 1.27

The numerical results above describe the current local licensed-data run. See
[ARB_FRAMEWORK.md](ARB_FRAMEWORK.md) for the modeling assumptions and limitations.

## Start Here

There are two ways to run the project.

### Rebuild from the existing local data

Use this path for normal research, code changes, result refreshes, and slide work.
It does not call WRDS, SEC, or an LLM.

```bash
git clone https://github.com/alviny800/JPM-Project-3ii.git
cd JPM-Project-3ii

python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt

# Copy the seven local inputs listed below into their standard paths first.
python arb_pipeline.py check
python arb_pipeline.py fast
```

`fast` runs the complete reproducible analytics chain:

```text
outcome probabilities
  -> deadline terms and demand Monte Carlo
  -> calibration and realized-edge diagnostics
  -> risk gates and ENTER/REVERSE decisions
  -> holder structure and capacity
  -> historical daily strategy results
  -> slide-ready material
```

To refresh only the presentation assets after the model outputs already exist:

```bash
python arb_pipeline.py material
```

Do not run `material_builder.py --help`. It is a build module, not a CLI; execute
it through `arb_pipeline.py material`.

### Rebuild the raw data

Use this path only when the Bloomberg universe, SEC extraction, or WRDS data must
be replaced. These steps require licensed data, credentials, API access, and
manual identifier review. They are intentionally separate from the normal
offline rebuild.

Follow [Full Data Build](#full-data-build), then return to:

```bash
python arb_pipeline.py check
python arb_pipeline.py fast
```

## Standard Local Inputs

The public repository does not contain Bloomberg or CRSP source data. Place the
following files at these exact paths before running `fast`.

| Path | Purpose | Produced by |
|---|---|---|
| `BBG Data Pull 2006+ Final.csv` | All-status outcome-model training data | Bloomberg export |
| `ma_edgar_full/llm_field_extractions.csv` | Deal terms, caps, election deadlines, realized election disclosures | `download_ma_edgar_files.py` |
| `ma_market_wrds/wrds_market_daily.csv` | Target/acquirer prices, volume, and daily paths | `download_wrds_market_data.py` plus `fix_acquirer_prices.py` |
| `ma_market_wrds/event_security_map.csv` | Event-to-security identifiers | `download_wrds_market_data.py` |
| `target_close_dates.csv` | CRSP close-date fallback | `build_close_dates.py` |
| `normalized_labels.csv` | Clean cash-election demand labels | `normalize_labels.py` |
| `eda_output/merged_panel.csv` | Event panel used by capacity and diagnostics | `election_arb_eda.py` |

The preflight command checks that every file exists and contains its required
columns:

```bash
python arb_pipeline.py check
```

## End-to-End Pipeline

The table below is the canonical run order. "Normal rerun" means the step is part
of `python arb_pipeline.py fast`; "data refresh" means it is only needed when raw
inputs change.

| Order | Layer | Main command | Why it exists | Primary output | When to run |
|---:|---|---|---|---|---|
| 1 | Universe and identifiers | `prepare_input_identifiers.py` | Repairs tickers/CUSIPs and creates auditable security identifiers | resolved universe and CUSIP audit | data refresh |
| 2 | SEC identity audit | `cik_resolution.py` | Resolves target names to SEC CIKs and maintains verified overrides | `cik_manual_overrides.csv` | data refresh |
| 3 | SEC and LLM extraction | `download_ma_edgar_files.py` | Downloads merger filings and extracts the 17 canonical fields | `ma_edgar_full/llm_field_extractions.csv` | data refresh |
| 4 | Market and ownership data | WRDS download scripts | Retrieves prices, volume, identifiers, and ownership proxies | `ma_market_wrds/`, `ma_ownership_wrds/` | data refresh |
| 5 | Price repair and close dates | `fix_acquirer_prices.py`, `build_close_dates.py` | Corrects identifier collisions and anchors historical close dates | corrected daily prices, `target_close_dates.csv` | data refresh |
| 6 | Label normalization | `normalize_labels.py` | Separates election demand from post-proration allocation | `normalized_labels.csv` | data refresh |
| 7 | Merged analysis panel | `election_arb_eda.py` | Joins extracted terms, market, ownership, and normalized labels | `eda_output/merged_panel.csv` | data refresh |
| 8 | Deadline spread | `deadline_spread.py` | Prices the cash-versus-stock choice at the election deadline | `deadline_spread.csv` | normal rerun |
| 9 | Outcome overlay | `arb_pipeline.py outcome` | Produces completed/terminated/withdrawn probabilities | `deal_outcome_probabilities.csv` | normal rerun |
| 10 | Terms and Monte Carlo | `arb_pipeline.py mc` | Fits demand, applies proration, and validates the edge | `arb_deals.csv`, `arb_output/` | normal rerun |
| 11 | Signal and capacity | `arb_pipeline.py signal` | Applies risk gates, holder structure, liquidity, borrow, and self-impact | `arb_signals.csv`, strategy summaries | normal rerun |
| 12 | Historical strategy | included in signal | Reconstructs daily target/acquirer hedged P&L | daily/event return CSVs | normal rerun |
| 13 | Presentation material | `arb_pipeline.py material` | Exports stable tables, figures, metrics, and charts | `material/` | normal rerun |

## Normal Rebuild, Step by Step

The consolidated commands are preferred because they keep the run order and
default file paths consistent.

### 1. Validate inputs

```bash
python arb_pipeline.py check
```

This checks the seven standard local inputs and fails before expensive work if a
file or required column is missing.

### 2. Build the outcome model

```bash
python arb_pipeline.py outcome
```

Purpose:

- trains the completed/terminated/withdrawn Naive Bayes model on all-status
  Bloomberg labels;
- tunes hyperparameters using expanding-year validation;
- writes event-level probabilities for downstream Monte Carlo;
- exports outcome-model performance material.

Output:

- `deal_outcome_probabilities.csv`
- `material/01_outcome_*`

The 293-event deployment frame is completion-only and is not used as the
three-class test set. Temporal out-of-sample evaluation uses 812 all-status
observations from 2016-2026.

### 3. Build deadline terms and run Monte Carlo

```bash
python deadline_spread.py
python arb_pipeline.py mc
```

Purpose:

- resolves the election deadline or close-date fallback;
- prices cash and stock consideration at that date;
- excludes floating ratios from fixed-ratio spread inference;
- fits the Beta election-demand distribution;
- runs leave-one-out calibration and the realized-edge event study;
- simulates completed, terminated, and withdrawn payoff paths.

Output:

- `deadline_spread.csv`
- `arb_deals.csv`
- `arb_output/summary.json`
- `arb_output/summary.md`
- `arb_output/*.png`
- `material/00_mc_*`

`arb_pipeline.py fast` invokes this deadline-spread rebuild before the outcome,
Monte Carlo, and signal layers. Running it explicitly is useful when debugging
the terms layer in isolation.

### 4. Build signals, capacity, and historical P&L

```bash
python arb_pipeline.py signal \
  --outcome-probs deal_outcome_probabilities.csv
```

Purpose:

- evaluates ENTER and REVERSE payoffs under the three-state outcome distribution;
- applies expected-return, fifth-percentile, and loss-probability gates;
- sends inconsistent terms to REVIEW and weak opportunities to PASS;
- estimates rolling holder parameters `p` and `q`;
- limits ENTER capacity by sellers, ADV, position size, and proration self-impact;
- limits REVERSE capacity by borrow, ADV, position size, and noisy-buyer demand;
- reconstructs capacity-weighted daily target/acquirer hedged returns.

Output:

- `arb_signals.csv`
- `arb_signals_clean.csv`
- `arb_strategy_summary.json`
- `arb_strategy_summary.csv`
- `arb_strategy_daily_returns.csv`
- `arb_strategy_event_daily_returns.csv`
- `material/02_holder_*` through `material/06_strategy_result_*`

### 5. Refresh presentation artifacts

```bash
python arb_pipeline.py material
python build_walkthrough.py
```

Purpose:

- collects each layer's important outputs and performance metrics;
- produces stable CSV/JSON tables and presentation-ready PNG charts;
- rebuilds the standalone HTML walkthrough.

Output:

- `material/material_manifest.md`
- `material/material_index.csv`
- `material/*.csv`, `material/*.json`, `material/*.png`
- `arb_output/walkthrough.html`

The bilingual DOCX/PDF speaker scripts are versioned presentation deliverables
stored in `material/`; `material_builder.py` does not regenerate those two files.

## Full Data Build

The commands in this section are expensive or credentialed. Review every audit
file before moving to the next step.

### 1. Install credentials

```bash
export WRDS_USERNAME="your_wrds_username"
export ANTHROPIC_API_KEY="your_anthropic_api_key"
export SEC_USER_AGENT="Your Name your.email@example.com"
```

Do not store passwords or API keys in the repository.

### 2. Prepare the Bloomberg universe and CRSP identifier cache

Place the licensed Bloomberg export at:

```text
BBG Data Pull 2006+ Final.csv
```

Download the CRSP stock-name table once:

```bash
python prepare_input_identifiers.py backfill-cusips \
  --input "BBG Data Pull 2006+ Final.csv" \
  --dump-stocknames \
  --wrds-username "$WRDS_USERNAME"
```

Run the offline identifier reconciliation:

```bash
python prepare_input_identifiers.py backfill-cusips \
  --input "BBG Data Pull 2006+ Final.csv" \
  --out-dir cusip_backfill \
  --election-only
```

Review:

- `cusip_backfill/cusip_backfill_audit.csv`
- `cusip_backfill/cusip_needs_review.csv`

The research universe used downstream is the reviewed US cash-or-stock election
subset, stored locally as `US_election_deals_for_analysis.csv`. This review gate
is intentional: delisted securities and historical ticker reuse cannot be made
fully reliable from a current ticker alone.

Clean the ticker helper columns:

```bash
python prepare_input_identifiers.py clean-tickers \
  --input US_election_deals_for_analysis.csv
```

### 3. Audit CIK resolution

```bash
python cik_resolution.py audit --user-agent "$SEC_USER_AGENT"
python cik_resolution.py build-overrides --user-agent "$SEC_USER_AGENT"
```

Review candidate CIKs before updating the committed
`cik_manual_overrides.csv`.

### 4. Build the close-date anchor

```bash
python build_close_dates.py \
  --input US_election_deals_for_analysis.csv \
  --stocknames stocknames_cache.csv \
  --out target_close_dates.csv
```

This uses the CRSP security-name validity endpoint as the close-date fallback.
No additional WRDS query is required.

### 5. Download SEC filings and extract fields

```bash
python download_ma_edgar_files.py \
  --input US_election_deals_for_analysis.csv \
  --output-dir ma_edgar_full \
  --user-agent "$SEC_USER_AGENT" \
  --cik-overrides cik_manual_overrides.csv \
  --close-dates target_close_dates.csv \
  --llm-stage batch \
  --max-batch-cost-usd 80
```

This stage writes the SEC manifest, field-locator diagnostics, and
`ma_edgar_full/llm_field_extractions.csv`. The raw filing directory can exceed
40 GB and is gitignored.

### 6. Download WRDS market and ownership data

```bash
python download_wrds_market_data.py \
  --input US_election_deals_for_analysis.csv \
  --output-dir ma_market_wrds \
  --wrds-username "$WRDS_USERNAME" \
  --llm-extractions ma_edgar_full/llm_field_extractions.csv

python download_ownership_etf_data.py \
  --input US_election_deals_for_analysis.csv \
  --output-dir ma_ownership_wrds \
  --provider wrds \
  --wrds-username "$WRDS_USERNAME"
```

If the installed WRDS client requests a password interactively, use the
corresponding `--wrds-password-stdin` option.

Apply the known price corrections after the market pull:

```bash
python fix_acquirer_prices.py
```

`fix_acquirer_prices.py` includes explicit PERMNO overrides for the known
Isle/MTR acquirer collision and Sirius target collision.

### 7. Normalize labels

```bash
python normalize_labels.py \
  --extractions ma_edgar_full/llm_field_extractions.csv \
  --out normalized_labels.csv
```

This is a small second LLM pass. It distinguishes shares that holders
**elected** from shares they ultimately **received after proration**.

### 8. Build the merged panel

```bash
python election_arb_eda.py \
  --extractions ma_edgar_full/llm_field_extractions.csv \
  --ownership ma_ownership_wrds/ownership_mix_by_event.csv \
  --market ma_market_wrds/event_market_features.csv \
  --normalized normalized_labels.csv \
  --output-dir eda_output
```

Then run the normal rebuild:

```bash
python arb_pipeline.py check
python arb_pipeline.py fast
```

## Code Layout

```text
arb_pipeline.py              preferred consolidated CLI
arb_outcome.py               outcome probability model
arb_terms.py                 deterministic deal-term table
arb_mc.py                    election-demand and proration engine
arb_backtest.py              demand calibration and realized-edge tests
arb_capacity.py              holder structure and capacity helpers
arb_signal.py                risk gates, sizing, blotter, daily P&L
structural_election_model.py rolling p/q holder model
material_builder.py          slide-material exporters

download_*.py                external data acquisition
prepare_input_identifiers.py identifier preparation
cik_resolution.py            SEC identity audit
normalize_labels.py          realized-demand normalization
election_arb_eda.py          merged panel and EDA
deadline_spread.py           election-deadline pricing

arb_output/                  tracked high-level model figures and walkthrough
material/                    tracked slide-ready tables, metrics, charts, scripts
reference/                   field definitions and source maps
```

The modular `arb_*.py` files remain the readable implementation and test surface.
`arb_pipeline.py` is the delivery entry point that combines their run order into
one command. Generated root-level CSVs are reproducible working files and are
gitignored; curated presentation outputs live in `material/`.

## Tests

Run the focused regression suite:

```bash
python -m unittest -v test_outcome_tuning.py test_strategy_history.py
```

Run a syntax check over the Python entry points:

```bash
python -m py_compile *.py
```

For a full integration check with local licensed inputs:

```bash
python arb_pipeline.py check
python arb_pipeline.py fast
```

## Output Interpretation

- `6.42%` is aggregate expected P&L divided by aggregate opportunity notional
  across the 20 selected trades. It is not a time-series return.
- `96.67%` is the additive sum of capacity-weighted daily returns across
  sequential opportunities. Capital is reused, so it is not comparable to
  6.42% without accounting for time.
- Active-position Sharpe/Sortino use only days with at least one position.
- Full-calendar Sharpe/Sortino retain inactive business days as zero returns.
- `completion_only` and `all_tradable` are currently identical because all 18
  reconstructed outcomes are completed.
- The three-class outcome model is evaluated with temporal Brier score and
  prior-adjusted balanced metrics, not accuracy on the completion-only
  deployment frame.

## Known Limitations

- **Disclosure ceiling:** only 73 deals disclose usable election demand.
- **Outcome signal:** temporal Brier improves from 0.317 for prior-only to 0.307
  for the tuned model; useful, but modest.
- **Survivorship:** historical daily paths currently contain completed trades
  only. A complete all-outcome backtest needs actual break paths.
- **Execution costs:** reported historical returns exclude transaction costs,
  financing, and borrow fees.
- **Capacity assumptions:** ADV and observed ownership are data inputs, but some
  sale, lending, borrow, and holder-behavior rates remain assumptions.
- **Licensed data:** Bloomberg and CRSP source exports cannot be committed to the
  public repository.

## Supporting Documentation

- [ARB_FRAMEWORK.md](ARB_FRAMEWORK.md): model mechanics and scope
- [Field_File_Timeline_Guide.md](Field_File_Timeline_Guide.md): field timing and
  source hierarchy
- [material/material_manifest.md](material/material_manifest.md): presentation
  output catalog
- [material/stage4_to_stage10_speaker_script_5_7min.md](material/stage4_to_stage10_speaker_script_5_7min.md):
  concise speaker script source
