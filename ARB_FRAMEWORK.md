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
| `arb_backtest.py` | VALIDATION — (A) leave-one-out calibration, (B) realized-edge event study | `arb_realized_edge.csv` |
| `arb_run.py` | DRIVER — runs all, writes figures + summary | `arb_output/` |
| `arb_signal.py` | TRADE LAYER — entry, election side, hedge, sizing, go/no-go rule | `arb_signals.csv` |
| `fix_acquirer_prices.py` | recover acquirer prices lost to identifier drift (CUSIP→PERMNO) | appends to `wrds_market_daily.csv` |

Run everything: `.venv/bin/python3 arb_run.py && .venv/bin/python3 arb_signal.py`

## Results (current, n=73 demand / 32 MC-ready)
- **Demand model:** Beta(0.59, 0.78), mean 43% elect cash; U-shaped (election is near a corner decision).
- **Calibration backtest ✅:** leave-one-out PIT mean 0.50, ~81% in the 80% band, **KS p=0.96** → the sampled distribution is honest out-of-sample.
- **Realized edge:** 100% positive; median ~1.7% of blended (positive is partly guaranteed — the magnitude is the informative part).
- **Trade blotter:** 31 deals, **22 ENTER**; **signal skill corr(E[return], realized) = +0.67** (predicted return tracks realized).
- **Portfolio MC:** positive proration edge; a 12% deal-break scenario opens a downside tail (the tail is deal-break, not election, risk).
- **Spread conditioning:** logit slope ≈ 0 on our data — supported by the framework, MC'd over its uncertainty rather than asserted.

## Two structural ceilings (both now near their limits)
- **Demand disclosure (73):** structural, proven — election *demand* is genuinely not disclosed on most deals (a sharp-prompt re-extraction of 26 candidates recovered 1). Not fixable.
- **Acquirer price / MC-ready (32 of 37):** *was* fixable — CRSP PERMNO resolution failed on identifier drift (renames/delistings). `fix_acquirer_prices.py` recovered it via CUSIP; 25 → 32. Remaining ~5 are foreign-listed or window gaps.

## Honest scope / what's NOT done yet
- **Demand distribution is solid and near the ceiling** — no more recoverable from EDGAR (98% of no-label deals already had the 8-K pulled; the number simply isn't disclosed).
- **Full trade P&L vs entry price** (survivorship-aware) needs one scoped WRDS pull: extend daily
  prices to each deal's close, and add the terminated deals for deal-break risk. Figures 1–3 and
  the calibration result need nothing further.
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
