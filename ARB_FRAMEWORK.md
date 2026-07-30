# Election-Arb Monte Carlo & Backtest Framework

Models **cash-or-stock election merger arbitrage**: predict aggregate election demand →
proration mechanics → realized consideration → strategy P&L distribution → a trade blotter.
Built on the 73 deals with clean, disclosed election demand (near the disclosure ceiling — see
`memory/election-arb-disclosure-ceiling.md`).

## The one idea
The only *random* quantity is **f_cash** = the fraction of shares that elect cash at the
deadline. Everything else is deterministic given deal terms. In a fully-prorated deal the
**blended (average) consideration is fixed** by the cash pool `pi_cash`; the arb edge comes from
**optimal election** — elect the richer side, and when that side is *under-subscribed* you
capture more than the average. How much you capture depends on the demand draw → that's the MC.

## Modules (run in this order)
| File | Role | Output |
|---|---|---|
| `arb_terms.py` | DATA LAYER — assemble clean deal terms (C, R, P_acq, pi_cash, realized f_cash) | `arb_deals.csv` |
| `arb_mc.py` | ENGINE — demand model (Beta + spread-conditioning) + proration mechanics + per-deal MC | (importable) |
| `arb_capacity.py` | CAPACITY OVERLAY — rolling structural holder mix, finite float/ADV, noisy vs positive-holder flow, and self-impact on election demand | (importable) |
| `arb_outcome.py` | OUTCOME ADAPTER — temporally tune/consume completed/terminated/withdrawn Naive Bayes probabilities; otherwise mark scenario defaults | `deal_outcome_probabilities.csv` when run |
| `arb_backtest.py` | VALIDATION — (A) leave-one-out calibration, (B) realized-edge event study | `arb_realized_edge.csv` |
| `arb_run.py` | MC DRIVER — terms, calibration, per-deal MC, and portfolio MC | `arb_output/` |
| `arb_signal.py` | TRADE LAYER — long election trade, passive-settlement reverse trade, hedge, max/optimal capacity, go/no-go rule | `arb_signals.csv`, `arb_strategy_summary.json` |
| `fix_acquirer_prices.py` | recover acquirer prices lost to identifier drift (CUSIP→PERMNO) | appends to `wrds_market_daily.csv` |

After the credentialed Stage 0-1 inputs have been built, run the canonical
offline Stage 2-10 rebuild:
`.venv/bin/python3 arb_pipeline.py check && .venv/bin/python3 arb_pipeline.py fast`.

`arb_pipeline.py` is the offline analytics orchestration entry point; it does
not download Bloomberg, SEC, WRDS, or LLM data. The stage modules remain the
implementation and test surface; generated `arb_output/` and `material/`
directories are disposable and are not version-controlled.

Optional outcome probabilities: `arb_signal.py --outcome-probs path/to/outcome_probs.csv`.
That file must be event-level and include `event_id` plus either
`p_completed/p_terminated/p_withdrawn` (aliases `deal_*_probability` also work) or an aggregate
`p_break`. If no such file/columns are present, `arb_signals.csv` sets
`outcome_probability_source=default_scenario_no_event_probabilities`; it does not pretend a
terminated/withdrawn model has been trained. `REVERSE` signals are short-target trades with no
election right: completion-state liability is the passive blended consideration, not the
optimal elected payoff.

Capacity is not a cosmetic multiplier. `arb_signal.py` writes capacity columns that model who
supplies/takes the target shares:

- Holder mix is rolling-estimated from prior disclosed election events using the older structural
  model. `p_hat` is the noisy/irrational holder cash-election probability; `q_hat` is the
  rational EV-sensitive holder share of total target ownership. Passive ownership remains the
  point-in-time WRDS/ETF estimate. The old fixed `positive_holder_share_of_active` is now only a
  fallback when the rolling structural estimate is unavailable.
- `ENTER`: buy target shares from noisy, positive, then passive/inactive holders. If long EV is
  strongly positive, rolling-estimated positive holders are assumed not to sell, so capacity is
  mostly noisy flow. The model then recomputes aggregate `f_cash` after our own election and
  reruns proration.
- `REVERSE`: short target by borrowing from inactive holders and selling to noisy buyers. This
  may shift market election demand, but it does **not** give us an election right; payoff stays
  passive blended settlement.

Capacity has three explicit levels:

- `capacity_raw_max_*`: flow/ADV/position-limit capacity before the trade payoff is re-gated.
- `capacity_max_*`: largest feasible size that still passes the risk gates after our self-impact.
- `capacity_optimal_*`: feasible size that maximizes expected dollar P&L. The legacy
  `capacity_notional` aliases point here.

The realized backtest reports two settlement methods. `baseline` accepts the historical election
result even though sizing considered our impact. `self_impact` shifts completed-state `f_cash` by
our size and counterparty mix before recomputing proration for the long election trade. Reverse
baseline and self-impact are intentionally identical under fixed-pool passive blended settlement,
because the short side has no election right.

## Results (current, n=73 demand / 32 MC-ready)
These are the committed/current run artifacts. Rerun `arb_run.py` and `arb_signal.py` after the
ignored local input CSVs are present to refresh them under the three-state outcome adapter.

- **Demand model:** Beta(0.59, 0.793), mean 43% elect cash; U-shaped (election is near a corner decision).
- **Calibration backtest ✅:** leave-one-out PIT mean 0.498, ~80% in the 80% band, **KS p=0.959** → the sampled distribution is honest out-of-sample.
- **Realized edge:** 100% positive; median ~1.4% of blended (positive is partly guaranteed — the magnitude is the informative part).
- **Trade blotter:** 88 priced signals: **10 ENTER**, **10 REVERSE**, 20 REVIEW, and 48 PASS-family (1 PASS + 28 loss-probability + 19 p05). The profit-optimal book has 6.42% weighted expected return. Realized settlement coverage is 16 trades with 100% positive hit rate and +0.78 expected-vs-realized correlation.
- **Capacity / P&L:** all 20 trades have positive shares/ADV capacity estimates. Holder mix is rolling-structural-fit for all 20; median `p_hat` is 0.30 and median `q_hat` is 0.10. Median optimal capacity is ~$17.9m notional / ~0.74% of target shares. Total optimal notional is ~$621.4m, expected P&L is ~$39.9m, and self-impact realized P&L is ~$17.0m on 16 settlement-covered trades. (Figures depend on how `deal_outcome_probabilities.csv` is generated; regenerate the outcome layer before comparing.)
- **Historical strategy reconstruction:** 18/20 executed trades have sufficient target/acquirer CRSP history. The capacity-weighted annualized Sharpe is 1.08 on active-position days and 0.45 when all intervening business days are retained as zero-return cash days; the corresponding Sortino ratios are 3.07 and 1.27. Four paths use an explicit market-to-deadline fallback and 14 reconcile to observed settlement returns. `all_tradable` currently equals `completion_only` because the executable sample contains 18 completed and zero terminated/withdrawn events.
- **Self-impact proof:** long election self-impact can theoretically eliminate 4 opportunities by shifting aggregate demand toward the elected side, but none reaches break-even at its selected optimal size; the median break-even among those four is ~44.25% of target shares. Reverse self-impact cannot eliminate payoff under the fixed-pool passive blended settlement assumption.
- **Portfolio MC:** positive proration edge; a completed/terminated/withdrawn scenario opens a downside tail (the tail is deal-outcome, not election, risk).
- **Outcome model:** nested expanding-year OOS on 812 labeled deals gives multiclass Brier 0.307 versus 0.317 for the prior-only benchmark (+3.0% Brier skill). A separately tuned prior-adjusted decision rule gives 46.0% balanced accuracy and 41.2% macro-F1.
- **Spread conditioning:** logit slope ≈ 0 on our data — supported by the framework, MC'd over its uncertainty rather than asserted.

## Two structural ceilings (both now near their limits)
- **Demand disclosure (73):** structural, proven — election *demand* is genuinely not disclosed on most deals (a sharp-prompt re-extraction of 26 candidates recovered 1). Not fixable.
- **Acquirer price / MC-ready (32 of 37):** *was* fixable — CRSP PERMNO resolution failed on identifier drift (renames/delistings). `fix_acquirer_prices.py` recovered it via CUSIP; 25 → 32. Remaining ~5 are foreign-listed or window gaps.

## Honest scope / what's NOT done yet
- **Demand distribution is solid and near the ceiling** — no more recoverable from EDGAR (98% of no-label deals already had the 8-K pulled; the number simply isn't disclosed).
- **Full signal layer is wired locally** — event-level completed/terminated/withdrawn probabilities are generated by the temporally tuned Bloomberg `Deal Status` Naive Bayes model when the licensed file is present, and `arb_signal.py` consumes that output explicitly. Tuning is nested inside each temporal fold; the 293-event completed-only frame is not used as a performance test.
- **Reverse trades are deliberately conservative** — they can short the target when the passive blended settlement is overpriced, but they do not receive an election right or the optimal-election payoff.
- **Capacity assumptions are partly estimated, partly structural.** Passive ownership and ADV are observed inputs; `p_hat/q_hat` come from rolling prior-event structural fits; sell/buy fractions, passive sale/lending fractions, and borrow availability remain behavioral assumptions. The capacity overlay labels holder source, fit depth, noisy/positive/passive shares, raw maximum, risk-gated maximum, profit-optimal size, and both realized settlement methods in the output columns.
- `pi_cash` defaults to 0.50 on ~28% of deals where the cap wasn't parseable — flagged in
  `arb_deals.csv` via `pi_cash_source`.

## Future refinements (deferred, not required)
- **Censored-demand Beta fit ($0, no new data).** The demand model fits only the ~73 deals with
  directly disclosed demand. ~116 deals disclose proration *outcomes* that carry partial demand
  info: unbound (allocation < cap) give exact demand; bound give a floor ("demand ≥ cap") or
  ceiling. Swap `fit_beta` in `arb_mc.py` for a censored maximum-likelihood Beta to use all of it.
  Modest payoff (~+6 exact, tighter U-tails); the model is already calibrated (KS p=0.96).
- Do NOT pursue more EDGAR extraction / document-scoring to grow n — empirically a dead end
  (sharp-prompt re-extraction on 26 candidates recovered 1). Demand is genuinely not disclosed.
