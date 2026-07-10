# Smoke Test Status

> **Status note (2026-07-10):** This is a historical snapshot of the June-26 single-deal smoke test. The pipeline has since scaled to the **full US election universe** (317 deals, ~283 SEC-reachable), switched extraction to `claude-sonnet-5` with **batch processing** (`--llm-stage batch`, ~50% cheaper, hard cost cap), added **EDGAR full-text CIK resolution** + hand-verified overrides and a **CRSP close-date anchor** for realized-results selection, and now **captures the realized election-demand label** (previously "not found") by deriving it from share counts, aggregate cash, or the proration factor. See `README.md` for the current run commands.

Run date: 2026-06-26 America/Los_Angeles

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

`--llm-stage send` now runs with Claude using `claude-sonnet-4-6` and `--llm-max-tokens 12000`. The Celgene/Bristol-Myers smoke extraction wrote `llm_field_extractions.csv` and returned the key fixed consideration terms:

- `$50.00` cash per Celgene share.
- `1.0` Bristol-Myers Squibb share per Celgene share.
- One tradeable CVR per Celgene share.
- Record date: March 1, 2019.

Fields that are not applicable to this fixed cash-and-stock deal, such as cash/stock election caps and proration formula, correctly return as not found instead of being fabricated.

## WRDS Ownership And Market Status

The WRDS ownership helper resolves historical Celgene through CRSP and writes ETF/passive ownership outputs. In the live smoke run, `ownership_mix_by_event.csv` returned `etf_ownership_percent=0.153286`.

The WRDS market helper resolves both Celgene and Bristol-Myers Squibb through CRSP and writes market outputs. In the live smoke run, `event_market_features.csv` returned target price `80.43`, acquirer price `45.12`, simple contract value `95.12`, and deal spread `14.69` on 2019-01-03. The simple contract value excludes the CVR because no market CVR value is supplied by the SEC terms.

## Note

The large downloaded SEC document tree, SEC cache, and selected uploaded document copies are intentionally ignored by git. The committed smoke-test files are the lightweight locator/payload outputs needed to inspect the pipeline shape.
