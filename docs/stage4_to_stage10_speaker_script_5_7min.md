# Stage 4 to Stage 10 Speaker Script

Target length: approximately 6 to 7 minutes.

## Slide 9 - Stage 4: Outcome Overlay (about 75 seconds)

Up to this point, Monte Carlo assumes that the deal completes. Stage 4 removes
that assumption by assigning every deal three probabilities: completed,
terminated, and withdrawn.

The three boxes show average predicted probabilities across our 293-deal
deployment universe: 85.0 percent completed, 11.3 percent terminated, and 3.8
percent withdrawn. They are probability weights, not accuracy. Completed paths
use the election and proration Monte Carlo. Terminated paths use a target break
value with at least a 25 percent loss floor, while withdrawn paths use a more
severe 35 percent floor.

The prior does not come from the 293-deal completion-only deployment frame. It
comes from 1,997 all-status Bloomberg labels: 1,648 completed, 258 terminated,
and 91 withdrawn. We regularize those frequencies with 25 pseudo-observations
at the 88, 7, and 5 percent scenario defaults, producing a prior of about 82.6,
12.8, and 4.6 percent. The tuned model then blends 75 percent of that prior with
25 percent of the feature-based Naive Bayes posterior.

The inputs are three numeric features - log announced value, TV-to-EBITDA, and
announcement year - plus six categorical features: deal type, payment type,
target and acquirer market suffixes, and indicators for whether each side has a
CUSIP. Deal Status is the training label, not an input, and no future election
result is used.

We tune the Naive Bayes model inside expanding-year validation. The chart shows
that its temporal out-of-sample Brier score is 0.307, versus 0.317 for a
prior-only forecast. That is a modest 3 percent skill gain. Balanced accuracy
is 46 percent, so this is not a perfect classifier. Its practical value is that
completion is no longer treated as certain.

## Slide 10 - Stage 5: Risk-Adjusted Monte Carlo (about 45 seconds)

Stage 5 inserts those three outcome states into the payoff simulation.

The blue distribution is the completed-state proration edge. Across the 32
MC-ready deals, its mean is 7.70 percent. The purple distribution replaces some
paths with terminated and withdrawn losses using the Stage 4 probabilities.
After that overlay, the mean falls to 1.94 percent and the fifth percentile is
negative 2.62 percent.

The election mechanism creates a completed-deal edge, but close risk consumes a
large part of it and creates a left tail. The next stages therefore test whether
that risk-adjusted distribution is still tradable.

## Slide 11 - Stage 6: Direction and Risk Gates (about 45 seconds)

Stage 6 turns each priced payoff distribution into an action.

ENTER means buying the target and exercising the richer election right.
REVERSE means shorting the target and accepting the passive blended settlement
liability. Both directions must clear the same three checks: expected return
above the hurdle, fifth-percentile return above the downside floor, and loss
probability below the ceiling.

Out of 88 priced events, 10 become ENTER, 10 become REVERSE, 20 are REVIEW, and
48 are in the PASS family. The large REVIEW and PASS groups are intentional.
REVIEW isolates inconsistent deal terms. PASS means the deal is understandable,
but its return or downside is not good enough. We preserve uncertainty rather
than force every event into a trade.

## Slide 12 - Stage 7: Holder Structure (about 40 seconds)

Stage 7 asks who is on the other side of our trade.

The holder model separates passive ownership from active ownership, then splits
active holders into noisy and EV-sensitive behavior. The parameter p is the
noisy holder's cash-election probability, while q is the EV-sensitive ownership
share.

Of 293 estimates, 261 use rolling prior-event fits and 32 use defaults. Median p
is 0.30 and median q is 0.10. The 32.1-point demand MAE shows why this is not a
standalone demand forecast. It is a capacity prior for share supply, borrow
supply, buyer demand, and self-impact.

## Slide 13 - Stage 8: Capacity and Self-Impact (about 60 seconds)

Stage 8 converts the 20 risk-approved directions into feasible position sizes.

All 20 candidates receive positive capacity. For ENTER, capacity is limited by
available sellers, ten days of trading at no more than 20 percent of ADV per
day, a 5 percent position limit, and the possibility that our own election
changes proration. For REVERSE, capacity is the minimum of borrowable shares,
ADV, the position limit, and noisy-buyer demand. In this run, noisy-buyer demand
is the binding constraint for all 10 REVERSE trades.

The 621.4 million dollars is the sum of optimal target-leg notionals across the
20 opportunities. It is not simultaneous capital and excludes the acquirer
hedge. Expected P&L is 39.9 million dollars, equivalent to a 6.42 percent
notional-weighted expected return.

The median position is 0.74 percent of target shares. The 44.25 percent
self-impact threshold is the median among four ENTER deals with a finite
threshold. It marks where an extremely large position would fail a risk gate;
actual positions are far below it.

## Slide 14 - Stage 9: Combined Trade Book (about 50 seconds)

Stage 9 aggregates the sized ENTER and REVERSE trades into one strategy view.

The left bar is the 39.9 million dollars of ex-ante expected P&L. Sixteen trades
have enough final information for direct realized-payoff validation, producing
about 17.0 million dollars of self-impact-adjusted realized P&L. The expected
versus realized return correlation is positive 0.78, which suggests that the
signal ranks these completed outcomes in a useful direction.

This remains a small, completion-heavy sample. The 6.42 percent is total
expected P&L divided by total notional across 20 trades, not a cumulative
historical return. The next slide changes the unit from a trade to a daily
portfolio.

## Slide 15 - Stage 10: Historical Daily Strategy Results (about 75 seconds)

Stage 10 reconstructs daily target and acquirer hedge P&L for 18 of the 20
selected trades. Fourteen paths reconcile to an observed settlement return,
four retain a market-to-deadline fallback, and two trades lack enough common
target and acquirer price history.

Each day, concurrent positions are weighted by their optimal notionals. On the
771 days with an active position, mean daily return is 0.13 percent, annualized
volatility is 29.33 percent, Sharpe is 1.08, and Sortino is 3.07. When we retain
all 4,469 business days and record inactive days as zero return, mean return
falls to 0.02 percent, volatility to 12.20 percent, Sharpe to 0.45, and Sortino
to 1.27. Maximum additive drawdown is negative 15.21 percent in both versions.

The two charts are different views of the same daily portfolio series. The left
attributes return to events in close-date order; the right shows it through
calendar time. Both end at 96.67 percent, the additive sum of daily
capacity-weighted returns. This exceeds Stage 8's 6.42 percent average expected
return because capital is reused across sequential opportunities.

All 18 reconstructed outcomes are completed. The history supports the election
and sizing mechanism, but it is not a survivorship-free live-performance
estimate. A full backtest still needs realized break paths and execution costs.
