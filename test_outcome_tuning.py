import math
import os
from pathlib import Path
import unittest

import pandas as pd

from arb_outcome import (
    BbgOutcomeNaiveBayes,
    OUTCOME_STATES,
    OutcomeDefaults,
    load_bbg_with_keys,
    tune_bbg_outcome_naive_bayes,
)


ROOT = Path(__file__).resolve().parent
PROJECT_ROOT = Path(os.environ.get("JPM_PROJECT_ROOT", ROOT))
BBG_PATH = PROJECT_ROOT / "BBG Data Pull 2006+ Final.csv"


class OutcomeTuningTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.rows = load_bbg_with_keys(str(BBG_PATH))
        cls.params, cls.tuning = tune_bbg_outcome_naive_bayes(
            cls.rows,
            OutcomeDefaults(),
        )
        cls.model = BbgOutcomeNaiveBayes(
            cls.rows,
            OutcomeDefaults(),
            params=cls.params,
        )

    def test_expected_training_labels_are_available(self) -> None:
        self.assertEqual(self.model.fit_n, 1997)
        self.assertEqual(
            self.model.label_counts,
            {"completed": 1648, "terminated": 258, "withdrawn": 91},
        )

    def test_tuning_uses_temporal_validation_rows(self) -> None:
        best = self.tuning.iloc[0]
        self.assertGreater(int(best["validation_rows"]), 0)
        self.assertTrue(math.isfinite(float(best["multiclass_brier_score"])))
        years = [int(year) for year in str(best["validation_years"]).split(",")]
        self.assertEqual(years, sorted(years))

    def test_probabilities_are_finite_and_normalized(self) -> None:
        row = self.rows.iloc[-1].copy()
        row["Deal Type"] = "previously unseen deal type"
        row["Target Ticker"] = "NEW ZZ"
        probabilities = self.model.predict_proba(row)
        self.assertEqual(set(probabilities), set(OUTCOME_STATES))
        self.assertTrue(all(math.isfinite(value) for value in probabilities.values()))
        self.assertTrue(all(0.0 <= value <= 1.0 for value in probabilities.values()))
        self.assertAlmostEqual(sum(probabilities.values()), 1.0, places=12)

    def test_deal_status_is_not_a_prediction_feature(self) -> None:
        row = self.rows.iloc[-1].copy()
        changed = row.copy()
        changed["Deal Status"] = "terminated"
        original = self.model.predict_proba(row)
        counterfactual = self.model.predict_proba(changed)
        for state in OUTCOME_STATES:
            self.assertAlmostEqual(original[state], counterfactual[state], places=12)


if __name__ == "__main__":
    unittest.main()
