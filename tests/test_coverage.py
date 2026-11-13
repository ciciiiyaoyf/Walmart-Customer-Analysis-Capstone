"""Coverage contracts, decision-time boundaries, and the observed reference window."""
import unittest

import numpy as np
import pandas as pd
from pandas.testing import assert_frame_equal

from src.coverage import build_transaction_coverage, observed_organic_reference_window


def orders(*rows):
    """Rows are (customer, timestamp, source, completed, revenue)."""
    result = pd.DataFrame(rows, columns=["customer_id", "order_ts", "order_source", "is_completed", "net_revenue"])
    result.insert(0, "order_id", np.arange(len(result)))
    result["order_ts"] = pd.to_datetime(result.order_ts, format="mixed")
    return result


def interval(start="2025-03-22 10:00", end="2025-06-20 10:00", **kwargs):
    return dict(scope="all_sources", start=start, end=end, available_as_of="2025-06-20 10:00", **kwargs)


class CoverageTests(unittest.TestCase):
    def setUp(self):
        self.spine = pd.DataFrame({"assignment_id": [10, 11], "customer_id": [1, 2],
                                   "send_ts": pd.to_datetime(["2025-06-20 10:00"] * 2)})
        self.empty = orders()

    def coverage(self, events=None, **kwargs):
        return build_transaction_coverage(self.spine, self.empty if events is None else events, **kwargs)

    def test_unknown_zero_is_not_certified_inactivity(self):
        result = self.coverage()
        self.assertEqual(len(result), 2)
        self.assertTrue(result.observed_orders_90d.eq(0).all())
        self.assertTrue(result.complete_orders_90d.isna().all())
        self.assertTrue(result.orders_90d_coverage_status.eq("unverified").all())

    def test_explicit_complete_zero_and_customer_scope(self):
        result = self.coverage(manifest={"coverage_intervals": [interval(customer_ids=[1])]})
        self.assertEqual(result.loc[0, "complete_orders_90d"], 0)
        self.assertEqual(result.loc[0, "orders_90d_coverage_status"], "verified_complete")
        self.assertTrue(pd.isna(result.loc[1, "complete_orders_90d"]))

    def test_completed_observed_counts_and_source_boundaries(self):
        events = orders((1, "2025-03-22 09:59:59", "organic_preperiod", True, 1),
                        (1, "2025-03-22 10:00", "organic_preperiod", True, 2),
                        (1, "2025-05-21 10:00", "campaign_attributed", True, 3),
                        (1, "2025-06-01", "organic_preperiod", False, 4),
                        (1, "2025-06-20 10:00", "organic_preperiod", True, 5),
                        (1, "2025-06-20 10:00:01", "organic_preperiod", True, 6))
        historical = self.coverage(events)
        scoring = self.coverage(events, inclusive=True)
        self.assertEqual(historical.loc[0, "observed_orders_90d"], 2)
        self.assertEqual(historical.loc[0, "observed_orders_30d"], 1)
        self.assertEqual(historical.loc[0, "observed_organic_orders_90d"], 1)
        self.assertEqual(historical.loc[0, "observed_campaign_orders_30d"], 1)
        self.assertEqual(scoring.loc[0, "observed_orders_90d"], 3)
        self.assertEqual(scoring.loc[0, "observed_orders_30d"], 2)
        self.assertEqual(scoring.loc[1, "observed_orders_90d"], 0)
        self.assertEqual(scoring.loc[0, "latest_observed_organic_order_ts"], pd.Timestamp("2025-06-20 10:00"))

    def test_inclusive_cutoff_requires_covered_endpoint(self):
        manifest = {"coverage_intervals": [interval()]}
        self.assertTrue(self.coverage(manifest=manifest).complete_orders_90d.eq(0).all())
        self.assertTrue(self.coverage(manifest=manifest, inclusive=True).complete_orders_90d.isna().all())
        manifest["coverage_intervals"][0]["end_inclusive"] = True
        self.assertTrue(self.coverage(manifest=manifest, inclusive=True).complete_orders_90d.eq(0).all())

    def test_certification_unavailable_at_decision_cannot_fill_count(self):
        certification = interval()
        certification["available_as_of"] = "2025-06-20 10:00:01"
        result = self.coverage(manifest={"coverage_intervals": [certification]})
        self.assertTrue(result.complete_orders_30d.isna().all())

    def test_partial_coverage_can_certify_30_days_but_not_90(self):
        result = self.coverage(manifest={"coverage_intervals": [interval(start="2025-05-01")]})
        self.assertTrue(result.complete_orders_30d.eq(0).all())
        self.assertTrue(result.complete_orders_90d.isna().all())

    def test_adjacent_intervals_cover_window_but_gap_does_not(self):
        first = interval(end="2025-05-01")
        second = interval(start="2025-05-01")
        manifest = {"coverage_intervals": [second, first]}
        self.assertTrue(self.coverage(manifest=manifest).complete_orders_90d.eq(0).all())
        second["start"] = "2025-05-01 00:00:01"
        self.assertTrue(self.coverage(manifest=manifest).complete_orders_90d.isna().all())

    def test_latest_event_is_not_a_coverage_watermark(self):
        events = orders((1, "2025-06-20 09:59:59", "organic_preperiod", True, 5))
        result = self.coverage(events)
        self.assertEqual(result.loc[0, "observed_orders_90d"], 1)
        self.assertTrue(result.complete_orders_90d.isna().all())
        self.assertEqual(result.loc[1, "source_latest_observed_organic_order_ts"], pd.Timestamp("2025-06-20 09:59:59"))
        self.assertTrue(pd.isna(result.loc[1, "latest_observed_organic_order_ts"]))

    def test_future_orders_cannot_change_past_diagnostics(self):
        events = orders((1, "2025-05-25", "organic_preperiod", True, 5))
        future = orders((1, "2025-05-25", "organic_preperiod", True, 5),
                        (1, "2025-06-21", "organic_preperiod", True, 9),
                        (2, "2025-06-22", "campaign_attributed", True, 10))
        assert_frame_equal(self.coverage(events), self.coverage(future))

    def test_scoring_customer_key_is_supported(self):
        spine = self.spine.rename(columns={"send_ts": "policy_as_of_ts"}).drop(columns="assignment_id")
        result = build_transaction_coverage(spine, self.empty, key="customer_id", cut="policy_as_of_ts", inclusive=True)
        self.assertEqual(result.customer_id.tolist(), [1, 2])
        self.assertTrue(result.window_end_inclusive.all())

    def test_source_only_certification_is_rejected(self):
        entry = interval()
        entry["scope"] = "organic_preperiod"
        with self.assertRaisesRegex(ValueError, "all_sources"):
            self.coverage(manifest={"coverage_intervals": [entry]})

    def test_invalid_contract_and_duplicate_decision_fail(self):
        with self.assertRaisesRegex(ValueError, "later"):
            self.coverage(manifest={"coverage_intervals": [interval(start="2025-06-21")]})
        with self.assertRaisesRegex(ValueError, "unique"):
            build_transaction_coverage(pd.concat([self.spine, self.spine]), self.empty)


class ReferenceWindowTests(unittest.TestCase):
    def setUp(self):
        self.spine = pd.DataFrame({"assignment_id": [10, 11, 12, 13], "customer_id": [1, 1, 1, 2],
                                   "send_ts": pd.to_datetime(["2025-01-09", "2025-01-10", "2025-06-20", "2025-06-20"])})
        self.events = orders((1, "2024-10-11 23:59:59", "organic_preperiod", True, 2),
                             (1, "2024-10-12", "organic_preperiod", True, 3),
                             (1, "2025-01-09 23:59:59", "organic_preperiod", True, 5),
                             (1, "2024-11-01", "organic_preperiod", False, 7),
                             (1, "2024-12-01", "campaign_attributed", True, 11),
                             (1, "2025-01-10", "organic_preperiod", True, 13),
                             (1, "2025-07-01", "organic_preperiod", True, 17))

    def test_fixed_reference_is_shared_and_excludes_wrong_source_status_and_bounds(self):
        result = observed_organic_reference_window(self.spine, self.events)
        self.assertTrue(result.reference_window_start.eq(pd.Timestamp("2024-10-12")).all())
        self.assertEqual(result.loc[1, "observed_organic_reference_orders"], 2)
        self.assertEqual(result.loc[2, "observed_organic_reference_orders"], 2)
        self.assertEqual(result.loc[2, "observed_organic_reference_net_revenue"], 8)
        self.assertEqual(result.loc[3, "observed_organic_reference_orders"], 0)
        self.assertEqual(result.loc[2, "reference_status"], "observed_only")
        self.assertGreater(result.loc[2, "reference_age_days"], 0)

    def test_before_anchor_is_unknown_even_when_some_reference_orders_exist(self):
        result = observed_organic_reference_window(self.spine, self.events)
        self.assertEqual(result.loc[0, "reference_status"], "not_yet_available")
        self.assertFalse(result.loc[0, "reference_available"])
        self.assertTrue(pd.isna(result.loc[0, "observed_organic_reference_orders"]))
        self.assertTrue(pd.isna(result.loc[0, "observed_organic_reference_net_revenue"]))
        self.assertTrue(pd.isna(result.loc[0, "latest_observed_organic_order_ts_in_reference"]))
        self.assertTrue(pd.isna(result.loc[0, "reference_age_days"]))

    def test_future_events_cannot_change_reference(self):
        result = observed_organic_reference_window(self.spine, self.events)
        result_without_future = observed_organic_reference_window(self.spine, self.events.iloc[:-1])
        assert_frame_equal(result, result_without_future)

    def test_signup_context_does_not_turn_observed_reference_into_complete_history(self):
        customers = pd.DataFrame({"customer_id": [1, 2], "signup_date": ["2024-01-01", "2024-12-20"]})
        result = observed_organic_reference_window(self.spine, self.events, customers=customers)
        self.assertEqual(result.loc[2, "reference_days_since_signup_within_window"], 90)
        self.assertEqual(result.loc[3, "reference_days_since_signup_within_window"], 21)
        self.assertEqual(result.loc[3, "reference_status"], "observed_only")


if __name__ == "__main__":
    unittest.main()
