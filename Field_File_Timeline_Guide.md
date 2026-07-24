# Corporate-Action Election Arbitrage: Field-File-Timeline Guide

## Purpose

This guide is the canonical extraction map for the corporate-action election/proration dataset. The goal is not to collect filings for their own sake. The goal is to retrieve the exact SEC filings or exhibits that contain the economic state variables needed to model election demand, realized fill rate, expected value, hedging, and backtest labels.

The project materials define deal terms as the legal payoff function: S-4s, proxy statements, merger agreements, election forms, and 8-Ks state the cash amount, exchange ratio, stock/cash caps, default treatment for non-electing holders, deadlines, odd-lot priority, and proration formula. Realized proration and deal outcomes are post-deadline labels and must not be used as trade-entry features.

## Transaction Timeline as File Keys

| Timeline key | Transaction phase | File/source key | SEC form / source | Typical release timing | Canonical fields extractable | Trading use |
|---|---|---|---|---|---|---|
| T0_announcement_8k | Deal announcement | Announcement 8-K and press release | 8-K, EX-99.1, Form 425 | Announcement date or shortly after signing | `target_name`, `acquirer_name`, `announce_date`, `payment_type`, `headline_consideration`, `expected_closing_timing`, `basic_conditions` | Event universe and initial screening; usually not sufficient for final proration mechanics |
| T0_merger_agreement | Signed merger agreement | Merger agreement exhibit | Usually EX-2.1 to 8-K or S-4 | Announcement date or shortly after signing | `cash_consideration_per_share`, `stock_consideration_per_share`, `exchange_ratio`, `basic_conditions`, sometimes `cash_cap`, `stock_cap`, `proration_formula` | Early legal backup; use final proxy/prospectus if later conflict exists |
| T1_registration_statement | Registration/proxy drafting | S-4 / S-4/A or F-4 / F-4/A | After announcement and before shareholder vote/election deadline | `consideration_menu`, `cash_consideration_per_share`, `stock_consideration_per_share`, `exchange_ratio`, `cash_cap`, `stock_cap`, `proration_formula`, `non_election_default_rule`, sometimes `record_date` and `election_deadline` | Core pre-election trade-entry mechanics; latest amendment preferred |
| T1_final_prospectus | Final prospectus/proxy-prospectus | 424B3 / 424B4 / 424B5 | After registration is effective, before vote/election deadline | Same as S-4, often more final: `consideration_menu`, `exchange_ratio`, `cash_cap`, `stock_cap`, `proration_formula`, `non_election_default_rule` | Preferred over original S-4 when available |
| T1_definitive_proxy | Definitive merger proxy | DEFM14A | Before shareholder meeting/vote | Clean summary of `record_date`, `shareholder_meeting_date`, `consideration_menu`, `cash_cap`, `stock_cap`, `proration_formula`, `non_election_default_rule` | Highest-priority summary for caps, proration, and default rule |
| T1_proxy_amendments | Proxy/prospectus amendments and communications | PREM14A, DEFA14A, Form 425 | Between announcement and vote/election deadline | Updates, clarifications, investor FAQs; may restate `consideration_menu`, `exchange_ratio`, `election_deadline`, or mechanics | Secondary evidence; useful for locating updates/conflicts |
| T2_election_materials | Election mechanics | Election form, form of election, letter of transmittal | After election materials are mailed; before election deadline | `election_deadline`, `non_election_default_rule`, `valid_election_requirements`, `cash_election_procedure`, `stock_election_procedure`, `odd_lot_priority`, `guaranteed_delivery_window` | Operational source for how holders actually make elections |
| T2_tender_exchange_offer | Tender or exchange offer mechanics | Schedule TO-T, offer to purchase, SC 14D-9, letter of transmittal | Offer launch through expiration | `expiration_date`, `election_deadline`, `withdrawal_rights`, `proration_formula`, `odd_lot_priority`, `guaranteed_delivery_window`, `target_recommendation` | Primary source for tender/exchange-offer deals |
| T3_market_snapshot | Trading-window market data | Price/borrow/volume feeds | Bloomberg, WRDS/CRSP, Polygon, yfinance, borrow vendors | Daily/intraday during announcement-to-election window | `target_price`, `acquirer_price`, `deal_spread`, `trading_cost_proxy`, `volume_liquidity`, `borrow_cost`, `borrow_availability` | Converts legal payoff into net alpha and hedge feasibility; not sourced from SEC merger filings |
| T3_ownership_snapshot | Investor composition | 13F, ETF holdings, N-PORT, fund SAI, securities lending data | SEC/vender/issuer files | Lagged or periodic; must use filing availability date | `passive_ownership`, `etf_ownership`, `hedge_fund_ownership`, `nport_on_loan`, `fund_lending_policy`, `lending_control_rule` | Forecasts election demand and effective passive imbalance; not in merger filings |
| T4_proration_label | Post-deadline election results | Post-deadline 8-K, EX-99.1 press release, final proration announcement | After election deadline, usually near/after closing | `preliminary_proration_results`, `final_proration_results`, `realized_cash_election_demand`, `realized_stock_election_demand`, `actual_cash_fill_rate`, `actual_stock_fill_rate` | Backtest/calibration label only; never a trade-entry feature |
| T4_completion_break | Deal outcome | Closing or termination 8-K / press release | Closing date or termination date | `deal_completion_or_break`, `closing_date`, `termination_date`, `tail_risk_outcome` | Backtest settlement and tail-risk label |

## Canonical SEC Field Dictionary

| Canonical field | Required? | Best file keys | Release timing | Claude extraction instruction |
|---|---:|---|---|---|
| `consideration_menu` | Yes | S-4/F-4, 424B, DEFM14A, election form | T1/T2 pre-election | Extract the available choices: cash election, stock election, mixed election, non-election/default path. |
| `cash_consideration_per_share` | Yes | Merger agreement, S-4/F-4, 424B, DEFM14A | T0/T1 pre-election | Extract the exact cash amount per target share or cash election amount. |
| `stock_consideration_per_share` | Yes | Merger agreement, S-4/F-4, 424B, DEFM14A | T0/T1 pre-election | Extract the stock consideration description and share amount if stated separately from exchange ratio. |
| `exchange_ratio` | Yes | Merger agreement, S-4/F-4, 424B, DEFM14A | T0/T1 pre-election | Extract fixed/floating exchange ratio and any collar language. |
| `cash_cap` | Yes | DEFM14A, 424B, S-4/F-4, election materials | T1/T2 pre-election | Extract maximum aggregate cash amount, cash election number, cash fraction, or allocation cap. |
| `stock_cap` | Yes | DEFM14A, 424B, S-4/F-4, election materials | T1/T2 pre-election | Extract maximum aggregate stock amount, stock election number, stock fraction, or allocation cap. |
| `proration_formula` | Yes | DEFM14A, 424B, S-4/F-4, letter of transmittal, offer to purchase | T1/T2 pre-election | Extract allocation mechanics for oversubscribed/undersubscribed cash or stock elections. |
| `non_election_default_rule` | Yes | DEFM14A, 424B, S-4/F-4, election form, letter of transmittal | T1/T2 pre-election | Extract what happens if a holder makes no valid election. This can flip the trade sign. |
| `election_deadline` | Yes | Election form, letter of transmittal, Schedule TO-T, DEFM14A, 424B | T2 pre-election | Extract exact deadline/expiration date and time; mark estimated if only expected. |
| `record_date` | Yes | DEFM14A, S-4/F-4, 424B | T1 pre-election | Extract record date for voting or election eligibility. |
| `odd_lot_priority` | Conditional | Letter of transmittal, offer to purchase, Schedule TO-T | T2 pre-election | Extract only if present; otherwise mark not_found. |
| `guaranteed_delivery_window` | Conditional | Notice of guaranteed delivery, letter of transmittal, Schedule TO-T | T2 pre-election | Extract delivery window and mechanics only if present. |
| `preliminary_proration_results` | Yes for labels | 8-K, EX-99.1, Form 425 | T4 post-election | Extract preliminary election/proration result; label only. |
| `final_proration_results` | Yes for labels | 8-K, EX-99.1, final proration announcement | T4 post-election | Extract final proration factor/results; label only. |
| `realized_cash_election_demand` | Conditional | 8-K, EX-99.1 | T4 post-election | Extract realized cash-election demand if disclosed. |
| `realized_stock_election_demand` | Conditional | 8-K, EX-99.1 | T4 post-election | Extract realized stock-election demand if disclosed. |
| `deal_completion_or_break` | Yes for labels | Closing/termination 8-K, press release | T4 post-election | Extract completion/termination status and date. |

## External Non-SEC Model Fields

The following fields are required for the full model but are not expected to be extracted from merger filings. They should be merged from external datasets after the SEC field extraction step.

| Canonical field | Source family | Timing discipline | Model role |
|---|---|---|---|
| `target_price` | Bloomberg, WRDS/CRSP, Polygon, yfinance | Use only prices available at the decision timestamp | Computes `EV_predicted - P_target` |
| `acquirer_price` | Bloomberg, WRDS/CRSP, Polygon, yfinance | Use only prices available at the decision timestamp | Converts exchange ratio into stock value |
| `deal_spread` | Computed from terms and prices | Daily/intraday pre-election | Main market-implied value input |
| `volume_liquidity` | Market data vendor | Pre-trade only | Transaction-cost/capacity filter |
| `borrow_cost` | Bloomberg/S3/Markit/DataLend or prime broker | Pre-trade only | Net-alpha hurdle |
| `borrow_availability` | Borrow vendor or prime broker | Pre-trade only | Hedge feasibility |
| `short_interest` | FINRA/vendor | Use publication date, not settlement date | Crowdedness/borrow-pressure proxy |
| `passive_ownership` | 13F, ETF holdings, Bloomberg/WRDS | Use filing/publication availability date | Forecasts default/non-optimizing election demand |
| `etf_ownership` | Bloomberg/WRDS/ETF issuer daily holdings | Use available date | Fresh passive ownership estimate |
| `hedge_fund_ownership` | 13F/Bloomberg/WRDS | Lagged; use filing date | Active/risk-arb election propensity proxy |
| `nport_on_loan` | N-PORT/vendor | Lagged; use public availability date | Adjusts passive ownership into effective election control |
| `fund_lending_policy` | Fund SAI/prospectus | Static/periodic | Recall/election-control assumption |
| `lending_control_rule` | MSLA/GMSLA templates, legal review | Static | Determines who controls elections on lent shares |

## Claude Field-Level JSON Contract

Claude must return exactly one JSON object. It must include every field in `requested_fields`, even when not found.

```json
{
  "event_id": "string",
  "target_name": "string",
  "acquirer_name": "string",
  "fields": {
    "consideration_menu": {
      "value": "directly extracted value/summary, or null",
      "basis": "direct/inferred/not_found",
      "timing_bucket": "pre_election_trade_entry/post_election_label/external_model_input",
      "source_doc_ids": ["doc_id"],
      "source_form_types": ["DEFM14A"],
      "source_filing_dates": ["YYYY-MM-DD"],
      "evidence_quotes": ["short exact quote"],
      "confidence": "high/medium/low",
      "notes": "ambiguities, conflicts, or missing information"
    }
  },
  "recommended_follow_up_documents": [
    {"doc_id": "doc_id", "reason": "why it should be manually reviewed or uploaded"}
  ],
  "timing_notes": "State which fields are pre-election features and which are post-election labels."
}
```

## API/Code Requirements

1. The SEC script must request all form families that can contain the required SEC fields: `8-K`, `8-K/A`, `S-4`, `S-4/A`, `F-4`, `F-4/A`, `424B3`, `424B4`, `424B5`, `DEFM14A`, `PREM14A`, `DEFA14A`, `425`, `SC TO-T`, `SC TO-T/A`, `SC TO-I`, `SC TO-I/A`, `SC 14D9`, `SC 14D9/A`, and `SC 13E3`.
2. The script should be run with `--download-exhibits`, because election forms, letters of transmittal, notices of guaranteed delivery, merger agreements, and press releases often sit in exhibits rather than the primary filing.
3. The output `event_field_coverage.csv` is the mandatory audit gate. A required field with `missing_required=True` means the retrieved SEC evidence is not sufficient and the event should be re-searched or manually reviewed.
4. The output `claude_field_payloads.jsonl` and `claude_upload_packages/` must tell Claude exactly which canonical fields are requested and which document/snippet is the candidate evidence for each field.
5. If `--llm-stage send` is used with an Anthropic API key, the script sends field-level payloads and writes `llm_field_results.jsonl` and `llm_field_extractions.csv`.
6. The ownership helper should be run separately for `passive_ownership` and `etf_ownership`. For WRDS, resolve historical target identifiers through CRSP `stocknames` or an override file, pull the latest point-in-time ETF/fund holdings snapshot available before the decision date, and divide ETF-held shares by contemporaneous shares outstanding when available. Store the result in `ownership_mix_by_event.csv` as `etf_ownership_percent` / `passive_control_percent`; do not mix these external fields into the SEC legal-term extraction contract.
7. The market helper should be run separately for `target_price`, `acquirer_price`, `deal_spread`, and liquidity fields. It should resolve target/acquirer identifiers through CRSP `stocknames`, pull point-in-time CRSP daily data from WRDS, and merge the resulting prices with Claude-extracted legal terms. Store the result in `event_market_features.csv`; unavailable borrow/short/lending fields must be tracked in `missing_market_data_inventory.csv` rather than silently treated as zero.
8. Company→CIK resolution must handle **delisted targets**. Resolve names via EDGAR full-text search (`efts.sec.gov`, covers filers back to 2001) as the primary matcher, fall back to `company_tickers.json` fuzzy matching, and consult a hand-verified override table first (`--cik-overrides cik_manual_overrides.csv`). Raise `--min-name-score` (≈90) to suppress wrong-company matches on generic bank/financial names; audit the universe with `cik_resolution.py audit` (build overrides for the tail with `cik_resolution.py build-overrides`).
9. Realized-results (T4 label) selection must be anchored on the deal **close date** (`--close-dates target_close_dates.csv`, the CRSP delisting date from `build_close_dates.py`): drop label candidates filed >30 days before close, reward filings near close, and add a results-signal bonus so a genuine election-results press release out-scores a bare completion 8-K. Claude must capture realized demand disclosed as raw share counts, an aggregate dollar amount, or a proration/oversubscription factor — not only a clean percentage — and derive the percentage when a share base is present (`basis="derived"`).
10. For large universes, use `--llm-stage batch` to submit all deals as one Anthropic Message Batch (~50% cheaper, asynchronous). `--max-batch-cost-usd` is a hard pre-flight cost cap that aborts before submitting if the estimate exceeds the ceiling; results are written to the same `llm_field_extractions.csv` as synchronous `send`.

## Recommended Command

```bash
python download_ma_edgar_files.py \
  --input ma_export_33248147_212700.csv \
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

To make Claude return actual values through the API instead of preparing packages only:

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
  --llm-model claude-sonnet-4-6 \
  --llm-max-tokens 12000 \
  --llm-stage send
```

To add ETF passive-control inputs through WRDS:

```bash
export WRDS_USERNAME="your_wrds_username"
read -s WRDS_PASSWORD
printf '%s\n' "$WRDS_PASSWORD" | \
python download_ownership_etf_data.py \
  --input ma_field_locator/candidate_events.csv \
  --output-dir ma_ownership_wrds \
  --provider wrds \
  --wrds-password-stdin \
  --cik-matches ma_field_locator/cik_name_matches.csv
```

Use `--dry-run` first to write `event_symbol_map.csv` and `ownership_request_plan.csv` without opening a WRDS connection. If a historical target cannot be resolved cleanly, edit the generated `symbol_overrides_template.csv` with `symbol`, `permno`, or `cusip`, then rerun with `--symbol-overrides`.

To add market/trading inputs through WRDS:

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

Use `--dry-run` first to write `event_security_map.csv` and `wrds_market_request_plan.csv`. The default market source is `crsp.dsf`; override the `--wrds-market-...` arguments if the WRDS subscription exposes the needed fields under a different library or table. `--include-short-volume` is a short-volume proxy only; true short-interest positions, borrow cost, borrow availability, N-PORT on-loan balances, and fund lending policy require additional vendor subscriptions or future EDGAR fund-filing parsers.

To run the downstream model: the original local v1 script has been **superseded by the `arb_*` Monte Carlo framework** (see `ARB_FRAMEWORK.md`). Modeling now runs as `arb_terms.py` → `arb_outcome.py` → `arb_run.py` → `arb_signal.py`.

As before, no tradable election edge is inferred when the legal terms show no shareholder election, or when caps/proration/default mechanics are missing; those deals are blocked with a preserved missing-variable explanation.

Finally, to run the Week-3 exploratory analysis and fit the empirical active-investor election function:

```bash
python election_arb_eda.py \
  --extractions ma_field_locator_claude/llm_field_extractions.csv \
  --ownership ma_ownership_wrds/ownership_mix_by_event.csv \
  --market ma_market_wrds/event_market_features.csv \
  --output-dir eda_output
```

This writes a deal-level `merged_panel.csv`, diagnostic plots, OLS/K-S tables, and `active_election_function.csv` — the fitted `p_active(spread)` used to calibrate active-investor election propensity in the structural model. Post-election fields (`final_proration_results`, realized demand) are used only as calibration labels, never as trade-entry features.
