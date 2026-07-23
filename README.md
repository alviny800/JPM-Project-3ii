# Election Arbitrage — Cash-or-Stock Merger Arbitrage Research

Models **cash-or-stock election merger arbitrage**: in many mergers, target holders *elect* cash
or stock but a **proration cap** limits each — so electing the richer, under-crowded side captures
more than the average holder. The payoff hinges on one unknown: **how much of the float elects
cash (election demand)**. This repo builds the full pipeline — SEC/Bloomberg data → Claude
extraction → a calibrated demand model → proration Monte Carlo → a risk-aware trade book.

> **Status (2026-07):** extraction + WRDS complete → **73 clean election-demand labels** (a proven
> disclosure ceiling). Demand model calibrated (leave-one-out **KS p=0.96**). Risk-aware trade book:
> 88 signals (5 ENTER / 15 REVERSE / 15 REVIEW), ~$766m profit-optimal notional, ~$50m expected P&L.
> See **`ARB_FRAMEWORK.md`** for the model, this file for the pipeline.

---

## Repository layout

```
top level          pipeline scripts (24) + input data + docs + the deck
ma_edgar_full/     raw SEC downloads (~43 GB, gitignored) + llm_field_extractions.csv
ma_market_wrds/    CRSP daily prices (target + acquirer), event↔security map
ma_ownership_wrds/ ETF/passive ownership
eda_output/        merged_panel.csv + EDA plots
arb_output/        model figures + summary.json + walkthrough.html
reference/         canonical field/source map
archive/           superseded/uncommitted material (logs, backups) — local only
```
Licensed data (Bloomberg, CRSP) and the 43 GB of raw filings are **gitignored** — kept local.

---

## The pipeline, stage by stage

### Stage 0 — Universe & input prep
| Script | What it does | Output |
|---|---|---|
| `prepare_input_identifiers.py` | `backfill-cusips`: re-resolve Excel-corrupted CUSIPs from CRSP · `clean-tickers`: strip `" US"` suffix | identifier columns on the analysis file |
| `event_csv_adapter.py` | Normalize event CSVs, add canonical helper columns | — |

*Inputs: `BBG Data Pull 2006+ Final.csv` (2,068 raw deals) → `US_election_deals_for_analysis.csv` (317 filtered).*

### Stage 1 — Identifier resolution
| Script | What it does | Output |
|---|---|---|
| `cik_resolution.py` | `audit`: run name→SEC-CIK matcher across the universe · `build-overrides`: verify CIKs for the delisted tail | `cik_manual_overrides.csv` |
| `fix_acquirer_prices.py` | Recover acquirer prices lost to identifier drift (BB&T→Truist) via CUSIP→PERMNO | appends to `wrds_market_daily.csv` |

### Stage 2 — EDGAR + Claude extraction
| Script | What it does | Output |
|---|---|---|
| `download_ma_edgar_files.py` | EDGAR retrieval, field locator, Claude extraction of 17 fields | `ma_edgar_full/llm_field_extractions.csv` |
| `field_specs.json` | The 17 canonical field definitions driving the locator + prompt | — |
| `reextract_unresolved.py` | Sharp-prompt re-extraction of specific deals (proved the disclosure ceiling) | `reextracted_labels.csv` |
| `batch_tools.py` | `submit`: batch a payload slice · `merge`: combine batch results | `llm_field_extractions.csv` |

### Stage 3–4 — WRDS market & ownership
| Script | What it does | Output |
|---|---|---|
| `download_wrds_market_data.py` | CRSP daily prices (target + acquirer), liquidity, spread | `ma_market_wrds/` |
| `download_ownership_etf_data.py` | ETF/passive ownership (came back ~0% for this small-cap universe → unused) | `ma_ownership_wrds/` |
| `build_close_dates.py` | Authoritative CRSP close date per target | `target_close_dates.csv` |

### Stage 5 — Clean & EDA
| Script | What it does | Output |
|---|---|---|
| `normalize_labels.py` | Separate election **demand** from post-proration **allocation** | `normalized_labels.csv` (73 labels) |
| `election_arb_eda.py` | Merge everything into a deal-level panel + diagnostics | `eda_output/merged_panel.csv` |

### Stage 6 — Spread
| Script | What it does | Output |
|---|---|---|
| `deadline_spread.py` | Deadline-date election spread + fixed/floating split | `deadline_spread.csv` |

### Stage 7 — Value engine (the Monte Carlo)
| Script | What it does | Output |
|---|---|---|
| `arb_terms.py` | Assemble the clean deal-terms table | `arb_deals.csv` |
| `arb_mc.py` | Beta demand model + proration mechanics + simulation | — |
| `arb_backtest.py` | Leave-one-out demand calibration + realized-edge event study | — |
| `arb_run.py` | Driver: terms → model → backtest → figures | `arb_output/` |

### Stage 8–13 — Risk-aware trade layer
| Script | What it does | Output |
|---|---|---|
| `arb_outcome.py` | Naive-Bayes outcome model (completed/terminated/withdrawn) on BBG `Deal Status` | `deal_outcome_probabilities.csv` |
| `deal_outcome_model.py` | Feature-based outcome model (used by the structural model) | — |
| `structural_election_model.py` | Rolling structural holder model (p/q); feeds capacity | — |
| `arb_capacity.py` | Capacity + self-impact sizing (ADV, float, borrow, holder mix) | — |
| `arb_signal.py` | Trade decisions: ENTER / REVERSE / REVIEW / PASS, hedge, risk gates, sizing | `arb_signals.csv`, `arb_strategy_summary.json` |

### Docs & presentation
| File | What it is |
|---|---|
| `README.md` · `ARB_FRAMEWORK.md` · `Field_File_Timeline_Guide.md` | Docs (pipeline · model · field map) |
| `build_walkthrough.py` | Renders the browser artifact → `arb_output/walkthrough.html` |
| `build_final_deck.py` (+ `haas_theme.pptx`) | Builds the JPM deck → `JPM_Election_Arb_Final.pptx` |

---

## How to run

### Fast local rebuild (regenerates the model + blotter + figures, ~$0, seconds)
Once extractions and WRDS data exist:
```bash
python deadline_spread.py                 # spread + fixed/floating
python arb_outcome.py                     # deal_outcome_probabilities.csv  (needed by arb_run)
python arb_run.py                         # demand model → MC → backtest → figures
python arb_signal.py                      # the trade blotter
python build_walkthrough.py               # refresh the browser artifact
```

### Full pipeline from a fresh Bloomberg pull (heavy stages — run once)
```bash
python prepare_input_identifiers.py backfill-cusips   # then: clean-tickers
python cik_resolution.py audit                        # then: build-overrides
python download_ma_edgar_files.py --input US_election_deals_for_analysis.csv \
    --cik-overrides cik_manual_overrides.csv --llm-stage batch --max-batch-cost-usd 80
python download_ownership_etf_data.py  --input US_election_deals_for_analysis.csv --target-cusip-col "Target cusip"
python download_wrds_market_data.py    --input US_election_deals_for_analysis.csv --target-cusip-col "Target cusip"
python normalize_labels.py --sync
python election_arb_eda.py --normalized normalized_labels.csv
# then the fast rebuild above
```

---

## Honest limitations (kept front-and-center)
- **Disclosure ceiling** — 73 labels is a *proven* limit: a sharp-prompt re-extraction of 26 candidates recovered 1. Election demand simply isn't disclosed on most deals.
- **Survivorship** — the backtest is on deals that *closed*; a full P&L needs the terminated deals.
- **Weak outcome features** — the naive-Bayes outcome model defaults to base rates except for outliers (e.g. mega-deals).
- **Capacity assumptions** — holder-mix / borrow figures are labeled, not measured.

## Notes
- **Disk:** `ma_edgar_full/` holds ~43 GB of raw filings (gitignored). The model only needs `llm_field_extractions.csv` + cached payloads; the raw `documents/` can be deleted to reclaim space.
- **Licensing:** no CRSP/Bloomberg data is committed (price-bearing intermediates are gitignored).
- **CIK resolution & close-date anchor:** delisted targets miss the current-registrant table, so name→CIK uses EDGAR full-text search + hand-verified overrides; realized-results evidence is anchored on the CRSP close date. Full detail in `Field_File_Timeline_Guide.md`.
