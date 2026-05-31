"""Unit tests for jurisdiction-aware location scoring (TextPreprocessor.calculate_location_score)."""

import os
import sys
import unittest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))

from finetuner.utils.text_preprocessor import TextPreprocessor


class TestLocationScoreJurisdiction(unittest.TestCase):
    def test_state_only_query_matches_target_state_bermuda(self):
        """Jurisdiction-only query should not be capped at 0.4 when state matches."""
        s = TextPreprocessor.calculate_location_score(
            None, "Bermuda", "", "Bermuda"
        )
        self.assertEqual(s, 1.0)

    def test_state_only_query_jurisdiction_in_target_city(self):
        """Catalog row with country in city column still matches query state."""
        s = TextPreprocessor.calculate_location_score(
            None, "Bermuda", "Bermuda", ""
        )
        self.assertEqual(s, 1.0)

    def test_same_us_city_and_state_full_score(self):
        """Regression: typical US city + state alignment remains perfect."""
        s = TextPreprocessor.calculate_location_score(
            "Springfield", "IL", "Springfield", "IL"
        )
        self.assertEqual(s, 1.0)

    def test_different_cities_same_state_below_perfect(self):
        """Regression: query includes city; must not lift to 1.0 on state match alone."""
        s = TextPreprocessor.calculate_location_score(
            "Houston", "TX", "Dallas", "TX"
        )
        self.assertLess(s, 1.0)
        self.assertGreater(s, 0.35)

    def test_cross_field_query_city_vs_target_state(self):
        """Swapped ingestion: query city holds label comparable to target state."""
        s = TextPreprocessor.calculate_location_score(
            "Bermuda", "", "", "Bermuda"
        )
        self.assertEqual(s, 1.0)


if __name__ == "__main__":
    unittest.main()
