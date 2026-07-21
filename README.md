# M&A EDGAR Field Locator for Corporate-Action Election Arbitrage

This project builds a field-driven SEC EDGAR retrieval pipeline for corporate-action election/proration merger research. It is not a generic document hoarder. The pipeline asks: for each transaction, where are the required fields, did we retrieve the filings/exhibits that contain them, and did we give Claude the right evidence to extract values?

See `Field_File_Timeline_Guide.md` for the English field-file-timeline specification, and
**`ARB_FRAMEWORK.md` for the finalized Monte Carlo model and trade decision layer** that consumes
these extractions.

> **Pipeline status (2026-07):** the extraction and WRDS stages are complete (317 deals → 73 clean
> election-demand labels, near the disclosure ceiling). The downstream modeling is the finalized
> Monte Carlo prototype in `ARB_FRAMEWORK.md`: a calibrated demand distribution (KS p=0.96) →
> proration mechanics → simulated payoff → a trade blotter (31 deals, 22 ENTER, signal skill
> corr +0.67).

## Core files

- `download_ma_edgar_files.py` — SEC EDGAR retrieval, field locator, Claude payload/package generation, optional Claude API extraction.
- `field_specs.json` — canonical SEC field definitions, preferred SEC forms, document keys, release timing, keywords, and required/conditional flags.
- `Field_File_Timeline_Guide.md` — English guide for project members and Claude prompt design.
- `download_ownership_etf_data.py` — separate ownership/ETF helper; not a replacement for SEC filing extraction.
- `download_wrds_market_data.py` — WRDS CRSP market/trading helper for target/acquirer prices, liquidity, shares, market cap, and deal-spread features.
- `build_election_strategy_model.py` — colleague's local v1 merge/audit/strategy script. **Superseded for modeling by the `arb_*` Monte Carlo framework** (see `ARB_FRAMEWORK.md`); kept for reference.
- `election_arb_eda.py` — exploratory data analysis and statistical tests. Merges the Claude extraction, WRDS ownership, and WRDS market outputs into a deal-level panel (`eda_output/merged_panel.csv`) used by the modeling layer.
- `normalize_labels.py` — second-pass LLM normalization that separates election **demand** (`pct_elected_cash`) from post-proration **allocation**; writes `normalized_labels.csv` (73 clean demand labels).
- `reextract_unresolved.py` — targeted re-extraction of specific deals with the sharpened prompt (batch or `--sync`); used to test and confirm the demand-disclosure ceiling.
- `fix_acquirer_prices.py` — recovers missing acquirer prices caused by identifier drift (renames/delistings) by resolving PERMNO from the acquirer CUSIP; grew MC-ready deals 25 → 32.
- `build_close_dates.py` — builds `target_close_dates.csv`, the authoritative CRSP delisting/close date per target, used to anchor realized-results (label) evidence selection.
- `secapi_io_fulltext_ma_screen.py` — optional sec-api.io helper (**moved to `archive/alt_scripts/`**).
- `audit_cik_resolution.py` — resolution-only audit ($0, EDGAR-only) of the name→CIK matcher across the deal universe; writes a score-sorted `cik_resolution_audit.csv` for review.
- `build_cik_overrides.py` — finds and verifies correct CIKs for the recoverable resolver tail (delisted/renamed targets) by cross-checking each candidate's EDGAR filing history near the close date.
- `cik_manual_overrides.csv` — hand-verified name→CIK overrides, consulted before EDGAR full-text search.
- `backfill_cusips.py`, `add_clean_ticker_cols.py` — one-time identifier prep that produced the clean `Target/Acquirer cusip` and `Ticker Clean` columns in the analysis input; consumed by the WRDS ownership/market stages (SEC retrieval itself is name-based and ignores them).
- `reference/` — canonical field/source map CSV, JSON, and Word document used to define which fields Claude should return and which source family each field belongs to.

### Monte Carlo model and trade layer (finalized prototype — see `ARB_FRAMEWORK.md`)

- `arb_terms.py` — assembles one clean deal-terms table (`arb_deals.csv`) the model reads.
- `arb_mc.py` — the engine: demand distribution (Beta) + proration mechanics + per-deal simulation.
- `arb_backtest.py` — validation: leave-one-out demand calibration + realized-edge event study.
- `arb_run.py` — driver: runs terms → model → backtest and writes `arb_output/` (figures + summary).
- `arb_signal.py` — trade decision layer: entry, election side, hedge, sizing, go/no-go → `arb_signals.csv`.
- `deadline_spread.py` — builds the deadline-date election spread and the fixed/floating split.
- `build_walkthrough.py` — renders the browser walkthrough artifact (`arb_output/walkthrough.html`).

Archived, non-core material lives in `archive/` (run logs, smoke tests, superseded intermediates).

## What the SEC script covers

The SEC script covers legal payoff and post-election label fields, including:

- `consideration_menu`
- `cash_consideration_per_share`
- `stock_consideration_per_share`
- `exchange_ratio`
- `cash_cap`
- `stock_cap`
- `proration_formula`
- `non_election_default_rule`
- `election_deadline`
- `record_date`
- `odd_lot_priority`
- `guaranteed_delivery_window`
- `preliminary_proration_results`
- `final_proration_results`
- `realized_cash_election_demand`
- `realized_stock_election_demand`
- `deal_completion_or_break`

External model fields such as prices, borrow cost, borrow availability, 13F ownership, ETF holdings, N-PORT on-loan balances, and fund lending policies are documented in the guide but are not expected to come from SEC merger filings.

## Recommended field-locator run

```bash
python download_ma_edgar_files.py \
  --input /Users/wxy/Downloads/US_election_deals_for_analysis.csv \
  --output-dir ma_field_locator \
  --user-agent "Xiangyu Wang xiangyuwang@berkeley.edu" \
  --payment-types "Cash or Stock" "Cash and Stock" \
  --deal-status Completed \
  --pre-days 60 \
  --post-days 730 \
  --download-exhibits \
  --field-specs field_specs.json \
  --field-locator-top-k 3 \
  --claude-package-max-docs-per-event 10 \
  --make-claude-packages \
  --llm-stage prepare
```

Important: use `--download-exhibits`. Election forms, letters of transmittal, notices of guaranteed delivery, merger agreements, and EX-99.1 proration releases often sit in exhibits.

The current research CSV (`US_election_deals_for_analysis.csv`) is accepted directly.  The pipeline now adds canonical helper columns to `candidate_events.csv`:

- `normalized_target_symbol`, `normalized_acquirer_symbol`
- `normalized_target_cusip`, `normalized_acquirer_cusip`
- `normalized_target_cusip_status`, `normalized_acquirer_cusip_status`

WRDS ownership and market scripts use these normalized columns by default, so the cleaned ticker/CUSIP work in the source CSV is preserved without passing extra `--target-symbol-col` / `--target-cusip-col` arguments.

## Outputs

- `manifest_sec_filings.csv` — every retrieved SEC candidate document/exhibit.
- `field_locator.csv` — event-field-document evidence rows. This is the main locator output.
- `event_field_coverage.csv` — one row per event-field, including missing required fields. Use `missing_required=True` as the audit gate.
- `selected_upload_docs.csv` — unique documents selected for Claude upload/review.
- `claude_field_payloads.jsonl` — field-level Claude payloads with requested fields and candidate evidence.
- `claude_upload_packages/` — per-event folder with `evidence_index.json`, `claude_prompt.txt`, and selected local files when `--make-claude-packages` is used.

## CIK resolution and close-date anchor (delisted targets)

Merger targets are delisted post-acquisition, so the current-registrant ticker table (`company_tickers.json`) misses ~96% of them. Name→CIK resolution therefore uses **EDGAR full-text search** (`efts.sec.gov`, which indexes filers back to 2001) as the primary resolver, with `company_tickers.json` fuzzy matching as a fallback and a hand-verified override table on top:

- `--cik-overrides cik_manual_overrides.csv` — trusted name→CIK overrides, consulted **before** efts (recovers delisted/renamed targets that fuzzy matching breaks or matches to the wrong entity). Build/verify candidates with `build_cik_overrides.py`.
- `--min-name-score 90` — raise the fuzzy accept threshold to suppress wrong-company matches on generic names (e.g. bank/financial names). Audit the whole universe first with `audit_cik_resolution.py`.

Realized-results (label) evidence is anchored on the deal **close date** so the terse results 8-K is preferred over the loud deal-announcement 8-K:

- `--close-dates target_close_dates.csv` — authoritative CRSP delisting/close date per target (built by `build_close_dates.py`). The field locator drops label candidates filed >30 days before close, rewards documents near close, and adds a results-signal bonus so a genuine election-results press release out-scores a bare completion 8-K.

Claude is also instructed to capture realized election demand when it is disclosed as raw share counts, an aggregate dollar amount, or a proration/oversubscription factor (not only a clean percentage), and to derive the percentage when a share base is present (`basis="derived"`).

## Claude API extraction

To call Claude directly and force actual field values to be returned:

```bash
export ANTHROPIC_API_KEY="your_key"
python download_ma_edgar_files.py \
  --input ma_export_33248147_212700.csv \
  --output-dir ma_field_locator_claude \
  --user-agent "Xiangyu Wang xiangyuwang@berkeley.edu" \
  --payment-types "Cash or Stock" "Cash and Stock" \
  --deal-status Completed \
  --pre-days 60 \
  --post-days 730 \
  --download-exhibits \
  --field-specs field_specs.json \
  --field-locator-top-k 3 \
  --claude-package-max-docs-per-event 10 \
  --close-dates target_close_dates.csv \
  --cik-overrides cik_manual_overrides.csv \
  --min-name-score 90 \
  --llm-model claude-sonnet-5 \
  --llm-max-tokens 12000 \
  --llm-stage send
```

This writes:

- `llm_field_payloads.jsonl`
- `llm_field_results.jsonl`
- `llm_field_extractions.csv`

Claude is instructed to return a `fields` object with every requested canonical field. If a field is not supported by the retrieved evidence, it must return `value=null` and `basis="not_found"` rather than omitting the field.

### Batch mode (recommended for large universes)

`--llm-stage batch` submits every deal as one Anthropic Message Batch (~50% cheaper, asynchronous) instead of one synchronous call per deal — the practical way to run the full ~283-deal universe:

```bash
python download_ma_edgar_files.py \
  --input US_election_deals_for_analysis.csv \
  --output-dir ma_edgar_full \
  --user-agent "Name email@domain.com" \
  --download-exhibits --field-specs field_specs.json --make-claude-packages \
  --close-dates target_close_dates.csv --cik-overrides cik_manual_overrides.csv \
  --min-name-score 90 \
  --llm-model claude-sonnet-5 --llm-stage batch \
  --max-batch-cost-usd 78 --batch-poll-seconds 60
```

- `--max-batch-cost-usd` is a **hard pre-flight cost cap**: the run estimates the batch cost from the built payloads and aborts *before submitting* if it would exceed the ceiling, so a run can never overspend.
- Downloads are cached (`--resume` is on by default), so a `batch` run reuses filings from a prior `prepare` run in the same `--output-dir` and only pays for the Claude calls.
- The batch is polled to completion and its results are written to the same `llm_field_extractions.csv` / `llm_field_results.jsonl` as synchronous `send`.

## ETF / passive ownership from WRDS

Use `download_ownership_etf_data.py` after the SEC run to add target ETF ownership and passive-control features. The WRDS provider first tries to resolve each target into CRSP identifiers (`permno`, CUSIP, ticker), then queries a fund-holdings table and converts ETF-held shares into `etf_ownership_percent` when CRSP shares outstanding are available.

Dry-run request plan:

```bash
python download_ownership_etf_data.py \
  --input smoke_celgene_prepare/candidate_events.csv \
  --output-dir smoke_celgene_ownership_wrds_plan \
  --provider wrds \
  --dry-run
```

Live WRDS run:

```bash
export WRDS_USERNAME="your_wrds_username"
read -s WRDS_PASSWORD
printf '%s\n' "$WRDS_PASSWORD" | \
python download_ownership_etf_data.py \
  --input smoke_celgene_prepare/candidate_events.csv \
  --output-dir smoke_celgene_ownership_wrds \
  --provider wrds \
  --wrds-password-stdin \
  --cik-matches smoke_celgene_prepare/cik_name_matches.csv
```

The default WRDS configuration assumes CRSP-style names: `crsp.stocknames`, `crsp_q_mutualfunds.holdings`, `crsp_q_mutualfunds.fund_hdr`, and `crsp.dsf`. If your WRDS subscription exposes ETF holdings under different libraries/tables, override the `--wrds-...` table and column arguments rather than editing code.

WRDS/FMP ownership outputs:

- `event_symbol_map.csv` — event-level target/acquirer identifiers, including WRDS `permno`/CUSIP when available.
- `wrds_security_map.csv` — WRDS security lookup diagnostics for CRSP identifier matching.
- `etf_holders_of_target.csv` — ETF/index-fund rows holding the target, with shares, market value, fund metadata, and `ownership_percent` when computable.
- `ownership_mix_by_event.csv` — event-level aggregate `etf_shares_or_exposure`, `etf_ownership_percent`, and `passive_control_percent` for model merge.

Historical targets are often delisted, so SEC current ticker matching may be missing or wrong. If WRDS name lookup is ambiguous, fill `symbol_overrides_template.csv` with `event_idx,side,symbol,permno,cusip,note` and rerun using `--symbol-overrides`.

## Market / trading data from WRDS

Use `download_wrds_market_data.py` after the SEC/Claude run to add market variables from WRDS CRSP. It resolves target and acquirer identifiers through CRSP `stocknames`, then pulls `crsp.dsf` daily rows around the announcement date.

```bash
export WRDS_USERNAME="your_wrds_username"
read -s WRDS_PASSWORD
printf '%s\n' "$WRDS_PASSWORD" | \
python download_wrds_market_data.py \
  --input ma_field_locator/candidate_events.csv \
  --output-dir ma_market_wrds \
  --wrds-password-stdin \
  --cik-matches ma_field_locator/cik_name_matches.csv \
  --llm-extractions ma_field_locator_claude/llm_field_extractions.csv \
  --include-short-volume
```

Market outputs:

- `wrds_security_map.csv` — CRSP identifier lookup diagnostics for target and acquirer.
- `wrds_market_daily.csv` — daily CRSP rows for the requested event window.
- `wrds_market_snapshot.csv` — as-of target/acquirer price, volume, dollar volume, ADV20/ADV60, bid-ask spread, shares outstanding, and market cap.
- `event_market_features.csv` — event-level `target_price`, `acquirer_price`, `simple_contract_value`, and `deal_spread` when SEC/Claude terms include cash and exchange ratio.
- `missing_market_data_inventory.csv` — explicit inventory of market fields that remain unavailable from the detected WRDS tables, such as true short interest, borrow cost, borrow availability, N-PORT on-loan balances, and fund lending policies.

`--include-short-volume` uses the configured WRDS short-volume table as a proxy only. It is not a substitute for true short-interest positions or securities-lending/borrow data.

For backtesting, `event_market_features.csv` also includes entry/exit close proxies:

- `entry_rule_date`, `entry_target_price`, `entry_acquirer_price`
- `exit_result_date`, `exit_target_price`, `exit_acquirer_price`

Entry date is inferred from Claude source filing dates for pre-election mechanics. Exit date is inferred from final/realized proration result source dates. If those dates are missing, the script falls back and marks the fallback in `market_feature_notes`.

## Model panel and v1 strategy signal

After SEC/Claude, ownership, and market runs finish, build the local model panel and coverage audit:

```bash
python build_election_strategy_model.py \
  --events ma_field_locator/candidate_events.csv \
  --llm-extractions ma_field_locator_claude/llm_field_extractions.csv \
  --ownership-mix ma_ownership_wrds/ownership_mix_by_event.csv \
  --market-features ma_market_wrds/event_market_features.csv \
  --output-dir ma_model_v1
```

This writes:

- `model_input_panel.csv` — one row per event with legal terms, ownership, and market features merged.
- `variable_coverage_report.csv` — available/missing/not-applicable status for each modeling field.
- `election_model_predictions.csv` — rolling fitted structural election-demand prediction and cap-aware trade/backtest rows.
- `election_model_predictions_heuristic.csv` — archived v1 heuristic output for comparison.
- `rolling_parameter_estimates.csv` — event-by-event fitted `p` and `q`.
- `election_backtest_trades.csv` — one row per event with trade/no-trade decision, missed-arbitrage flag, loss flag, and P&L.
- `backtest_summary.json` — trade count, missed arbitrage count, loss count, average P&L, and total P&L.
- `model_run_summary.json` — compact run summary.

The default structural model assumes zero borrow cost, uses a 1,000,000 dollar target long notional, and sizes the acquirer short with `--own-hedge-policy dollar_neutral` unless overridden.  The fitted parameters are:

- `p`: irrational investors' cash-election probability.
- `q`: EV-sensitive rational investors' original target ownership share.

The model rolls by prior labeled events, not by calendar time.  Realized labels come from Claude's `realized_cash_election_demand` / `realized_stock_election_demand` fields when available; missing labels are excluded from fitting.

## Week-3 EDA and p_active(spread) fitting

`election_arb_eda.py` consumes the three pipeline outputs and produces the empirical analysis behind the structural model — the distribution of realized cash-election demand, its response to the deadline-date spread, and a fitted `p_active(spread)` function for the Monte Carlo.

```bash
python election_arb_eda.py \
  --extractions ma_field_locator_claude/llm_field_extractions.csv \
  --ownership ma_ownership_wrds/ownership_mix_by_event.csv \
  --market ma_market_wrds/event_market_features.csv \
  --output-dir eda_output
```

This writes:

- `eda_output/merged_panel.csv` — deal-level panel joining legal terms, ownership, and market features, with derived columns (realized cash share, deadline spread, backed-out active-investor cash-election rate).
- `eda_output/plots/*.png` — realized-cash-share histogram, spread-vs-active-election scatter (the key chart), demand-vs-cap, election-by-default-rule, passive-ownership interaction, and time-series plots.
- `eda_output/tables/*.csv` — OLS coefficients, a K-S test that the default rule matters, and the fitted `active_election_function.csv` (`p_active(spread)`).
- `eda_output/eda_summary.md` — narrative summary with the V1 modeling assumptions and caveats.

The active-investor cash-election rate is backed out by treating lent/passive shares as taking the default rule, then attributing residual realized demand to the active population. It is fit as a simple linear `p_active(spread)` for V1; a logistic or piecewise form can replace it later. Post-election fields (`final_proration_results`, realized demand) are used only as calibration labels, never as trade-entry features.

## Coverage gate

After each run, inspect:

```bash
python - <<'PY'
import pandas as pd
cov = pd.read_csv('ma_field_locator/event_field_coverage.csv')
print(cov[cov['missing_required'] == True][[
    'event_id', 'field_name', 'expected_document_keys', 'preferred_forms'
]].head(50))
PY
```

Missing required fields mean one of three things:

1. The transaction is not actually an election/proration deal.
2. The right filing/exhibit was not retrieved or was not text-readable.
3. The field is buried in a document that needs a broader search or manual review.

Do not use post-election fields such as `final_proration_results` or `deal_completion_or_break` as trade-entry features. They are labels for calibration and backtest settlement.
