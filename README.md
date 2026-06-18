# M&A EDGAR Field Locator for Corporate-Action Election Arbitrage

This project builds a field-driven SEC EDGAR retrieval pipeline for corporate-action election/proration merger research. It is not a generic document hoarder. The pipeline asks: for each transaction, where are the required fields, did we retrieve the filings/exhibits that contain them, and did we give Claude the right evidence to extract values?

See `Field_File_Timeline_Guide.md` for the English field-file-timeline specification.

## Core files

- `download_ma_edgar_files.py` — SEC EDGAR retrieval, field locator, Claude payload/package generation, optional Claude API extraction.
- `field_specs.json` — canonical SEC field definitions, preferred SEC forms, document keys, release timing, keywords, and required/conditional flags.
- `Field_File_Timeline_Guide.md` — English guide for project members and Claude prompt design.
- `download_ownership_etf_data.py` — separate ownership/ETF helper; not a replacement for SEC filing extraction.
- `secapi_io_fulltext_ma_screen.py` — optional sec-api.io full-text helper.
- `reference/` — canonical field/source map CSV, JSON, and Word document used to define which fields Claude should return and which source family each field belongs to.
- `SMOKE_TEST_STATUS.md` — status of the Celgene/Bristol-Myers smoke test and the current Claude API send blocker.

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

Important: use `--download-exhibits`. Election forms, letters of transmittal, notices of guaranteed delivery, merger agreements, and EX-99.1 proration releases often sit in exhibits.

## Outputs

- `manifest_sec_filings.csv` — every retrieved SEC candidate document/exhibit.
- `field_locator.csv` — event-field-document evidence rows. This is the main locator output.
- `event_field_coverage.csv` — one row per event-field, including missing required fields. Use `missing_required=True` as the audit gate.
- `selected_upload_docs.csv` — unique documents selected for Claude upload/review.
- `claude_field_payloads.jsonl` — field-level Claude payloads with requested fields and candidate evidence.
- `claude_upload_packages/` — per-event folder with `evidence_index.json`, `claude_prompt.txt`, and selected local files when `--make-claude-packages` is used.

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
  --llm-stage send
```

This writes:

- `llm_field_payloads.jsonl`
- `llm_field_results.jsonl`
- `llm_field_extractions.csv`

Claude is instructed to return a `fields` object with every requested canonical field. If a field is not supported by the retrieved evidence, it must return `value=null` and `basis="not_found"` rather than omitting the field.

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
