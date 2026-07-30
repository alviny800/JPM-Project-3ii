import math
import unittest

import numpy as np
import pandas as pd

from arb_signal import _annualized_daily_stats, _risk_adjusted_block


class StrategyHistoryTests(unittest.TestCase):
    def test_annualized_sharpe_uses_daily_time_series(self):
        returns = pd.Series([0.01, -0.005, 0.002, 0.0])
        stats = _annualized_daily_stats(returns)
        expected = returns.mean() / returns.std(ddof=1) * math.sqrt(252.0)
        self.assertAlmostEqual(stats["annualized_sharpe_0rf"], expected)
        self.assertEqual(stats["day_count"], 4)

    def test_sortino_uses_zero_percent_minimum_acceptable_return(self):
        returns = np.array([0.01, -0.005, 0.002, 0.0])
        stats = _annualized_daily_stats(returns)
        downside_deviation = np.sqrt(np.mean(np.minimum(returns, 0.0) ** 2))
        expected = returns.mean() / downside_deviation * math.sqrt(252.0)
        self.assertAlmostEqual(stats["annualized_sortino_0pct_mar"], expected)

    def test_completion_only_cross_sectional_sortino_is_undefined_without_losses(self):
        trades = pd.DataFrame({
            "signal": ["ENTER", "REVERSE", "ENTER"],
            "realized_return_%": [3.0, 1.0, 2.0],
        })
        stats = _risk_adjusted_block(trades)["all_trades"]
        self.assertEqual(stats["negative_trade_count"], 0)
        self.assertIsNone(stats["sortino_0pct_mar"])
        self.assertAlmostEqual(stats["cross_sectional_mean_to_std"], 2.0)


if __name__ == "__main__":
    unittest.main()
