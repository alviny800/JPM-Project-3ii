# Smoke Test Status

Run date: 2026-06-17 America/Los_Angeles

## What ran

Small end-to-end prepare test:

```bash
python3 download_ma_edgar_files.py \
  --input ma_export_33248147_212700.csv \
  --output-dir smoke_celgene_prepare \
  --user-agent "Xiangyu Wang xiangyuwang@berkeley.edu" \
  --payment-types "Cash and Stock" \
  --deal-status Completed \
  --start-date 2019-01-03 \
  --end-date 2019-01-03 \
  --max-events 1 \
  --pre-days 60 \
  --post-days 730 \
  --download-exhibits \
  --field-specs field_specs.json \
  --field-locator-top-k 3 \
  --claude-package-max-docs-per-event 6 \
  --llm-max-docs-per-event 6 \
  --llm-max-doc-chars 25000 \
  --make-claude-packages \
  --llm-stage prepare \
  --sleep-seconds 0.11
```

## Result

- Event: `Celgene Corp / Bristol-Myers Squibb Co`
- Manifest rows: 202
- Field locator rows: 36
- Claude requested fields: 17 canonical SEC fields
- Generated: `smoke_celgene_prepare/claude_field_payloads.jsonl`
- Generated: `smoke_celgene_prepare/llm_field_payloads.jsonl`
- Generated: `smoke_celgene_prepare/event_field_coverage.csv`

## Field Coverage

Covered in the smoke test:

- `consideration_menu`
- `cash_consideration_per_share`
- `stock_consideration_per_share`
- `exchange_ratio`
- `cash_cap`
- `proration_formula`
- `non_election_default_rule`
- `election_deadline`
- `record_date`
- `odd_lot_priority`
- `guaranteed_delivery_window`
- `deal_completion_or_break`

Missing required evidence in this smoke test:

- `stock_cap`
- `preliminary_proration_results`
- `final_proration_results`

## Claude API Send Status

`--llm-stage send` was attempted after prepare, but the Anthropic API returned:

```text
401 Client Error: Unauthorized for url: https://api.anthropic.com/v1/messages
```

So the direct Claude API extraction did not produce `llm_field_results.jsonl` or `llm_field_extractions.csv`. The payload file is ready for a rerun once a valid Anthropic API key is available.

## Note

The large downloaded SEC document tree, SEC cache, and selected uploaded document copies are intentionally ignored by git. The committed smoke-test files are the lightweight locator/payload outputs needed to inspect the pipeline shape.
