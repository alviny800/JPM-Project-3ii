# Election-Arb Monte Carlo — results summary

## 1. Demand model (the MC's stochastic core)
- Calibrated on **73 realized election outcomes**.
- Fit: **Beta(0.59, 0.793)**, mean **43% elect cash**.
- Shape is U-ish: deals cluster toward "almost all cash" / "almost all stock" — election is close to a corner decision, as economics predicts.

## 2. Calibration backtest — IS THE MODEL HONEST?  ✅
- Leave-one-out PIT: mean **0.498** (ideal 0.50), **80%** inside the 80% band (ideal 80%), **KS p=0.959** → not distinguishable from perfectly calibrated.
- Interpretation: the distribution the Monte Carlo samples is trustworthy on out-of-sample deals.

## 3. Realized-edge event study — DOES THE ALPHA EXIST?
- 32 MC-ready deals. Proration-capture edge **100% positive**.
- Median **1.4% of blended**, mean **2.49%** (right-skewed: usually small, occasionally large).

## 4. Portfolio Monte Carlo
- Equal-weight 32 deals, 20,000 paths.
- Proration edge (no break): mean **7.7%**.
- With event-level completed/terminated/withdrawn probabilities averaging **20.3% terminated/withdrawn** (terminated loss 25%, withdrawn loss 35%): mean **0.56%**, 5th-pct **-3.76%** — the tail is deal-outcome risk, not election risk.

## Spread conditioning
- logit(demand) slope on deadline spread = **+0.0423** (se 0.0414) → ~flat on our data. Framework supports conditioning; the data says the tilt is weak, so we MC over that slope's uncertainty rather than assert it.

## Honest scope / current coverage
- **Demand distribution: solid (n=73).** Near the disclosure ceiling — no more recoverable from EDGAR (verified: 98% of no-label deals already had the 8-K pulled, the number just isn't disclosed).
- **No new Claude run is needed for this pipeline.** Election/proration demand uses the existing Claude extraction + normalized labels; deal-outcome risk uses BBG `Deal Status` across Completed/Terminated/Withdrawn.
- The realized election-edge backtest remains the completed-election universe because terminated/withdrawn deals do not have final proration/election outcomes. Their effect enters the trade and portfolio layers through the event-level outcome probabilities.
