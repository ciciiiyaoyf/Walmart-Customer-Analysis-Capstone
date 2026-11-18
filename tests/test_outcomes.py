import numpy as np
import pandas as pd
import pytest

from src.outcomes import applicable_refunds, refund_reconciliation, inferred_waived_shipping_fee
from src.features import build_features


def test_assumed_75_dollar_shipping_cutoff_is_inclusive():
    # Include unseen boundary baskets: the observed-data minimum is 75.08,
    # but the approved assumption gives ordinary free shipping from 75.00.
    fee = inferred_waived_shipping_fee([74.99, 75.00, 75.01, 75.08, 100.00],
                                      free_shipping_cutoff=75., standard_fee=5.99)
    np.testing.assert_array_equal(fee, [5.99, 0., 0., 0., 0.])


def refund_fixture():
    orders = pd.DataFrame({"assignment_id": ["a1"], "order_id": ["o1"],
                           "outcome_end_ts": pd.to_datetime(["2025-01-15"])})
    returns = pd.DataFrame({"order_id": ["o1"] * 3,
                            "return_ts": pd.to_datetime(["2025-01-05", "2025-01-15", "2025-01-20"]),
                            "refund_amount": [10., 7., 13.]})
    return orders, returns


@pytest.mark.parametrize("rule,expected", [("in_window", 10.), ("all_linked", 30.)])
def test_alternative_return_rules_reconcile_without_double_counting(rule, expected):
    orders, returns = refund_fixture()
    result = applicable_refunds(orders, returns, rule)
    assert result.loc["a1", "refund_amount_applicable"] == expected
    assert result.loc["a1", "in_window_return_count"] == 1
    assert result.loc["a1", "post_window_return_count"] == 2
    assert all(refund_reconciliation(result, 30., rule).values())


def test_as_of_cutoff_excludes_unavailable_refunds_including_in_sensitivity():
    orders, returns = refund_fixture()
    result = applicable_refunds(orders, returns, "all_linked", as_of_ts="2025-01-15")
    assert result.loc["a1", "refund_amount_applicable"] == 17.
    assert all(refund_reconciliation(result, 17., "all_linked").values())


def test_invalid_return_rule_is_not_silently_treated_as_all_linked():
    orders, returns = refund_fixture()
    with pytest.raises(ValueError, match="Unknown return rule"):
        applicable_refunds(orders, returns, "in_windw")


def test_no_linked_refunds_has_stable_schema():
    orders, returns = refund_fixture()
    result = applicable_refunds(orders, returns.iloc[:0])
    assert result.empty
    assert all(refund_reconciliation(result, 0.).values())


def test_multiple_known_refunds_do_not_duplicate_order_features():
    # Realistic enough to exercise the shared feature builder and item join.
    customers = pd.DataFrame({"customer_id": ["c1"], "signup_date": pd.to_datetime(["2024-01-01"]),
                              "is_member": [0], "marketing_consent": [1], "region": ["west"],
                              "acquisition_channel": ["web"], "customer_tier": ["Standard"],
                              "preferred_category": ["shoes"], "shipping_zone": ["z1"]})
    orders = pd.DataFrame({"order_id": ["o1"], "customer_id": ["c1"], "campaign_id": ["CP01"],
                           "order_ts": pd.to_datetime(["2025-01-01"]), "is_completed": [True],
                           "net_revenue": [100.], "cogs": [40.], "shipping_cost": [5.],
                           "merchandise_gross_revenue": [100.], "discount_amount": [0.],
                           "shipping_revenue": [0.], "offer_redeemed": [0]})
    assignments = pd.DataFrame({"assignment_id": ["past"], "customer_id": ["c1"], "campaign_id": ["CP01"],
                                "send_ts": pd.to_datetime(["2024-12-31"]), "offer_type": ["NO_OFFER"],
                                "email_delivered": [1]})
    sessions = pd.DataFrame({"customer_id": ["c1"], "session_ts": pd.to_datetime(["2025-01-01"]),
                             "pages_viewed": [1], "added_to_cart": [0], "category_viewed": ["shoes"]})
    emails = pd.DataFrame({"assignment_id": ["past"], "campaign_id": ["CP01"], "customer_id": ["c1"],
                           "event_ts": pd.to_datetime(["2025-01-01"]), "event_type": ["open"]})
    returns = pd.DataFrame({"order_id": ["o1"] * 3,
                            "return_ts": pd.to_datetime(["2025-01-05", "2025-01-10", "2025-02-10"]),
                            "refund_amount": [10., 20., 99.]})
    tables = {"customers": customers, "orders": orders, "returns": returns,
              "historical_assignments": assignments, "web_sessions": sessions, "email_events": emails,
              "order_items": pd.DataFrame({"order_id": ["o1"], "product_id": ["p1"], "quantity": [2]}),
              "products": pd.DataFrame({"product_id": ["p1"], "category": ["shoes"]})}
    spine = pd.DataFrame({"assignment_id": ["now"], "customer_id": ["c1"],
                          "send_ts": pd.to_datetime(["2025-02-01"])})
    features, _ = build_features(spine, tables)
    row = features.iloc[0]
    assert row.observed_prior_order_count == 1
    assert row.observed_orders_90d == 1
    assert row.observed_net_revenue_90d == 100.
    assert row.observed_historical_ncm_90d == 25.
    assert row.observed_refund_amount_90d == 30.
    assert row.observed_units_purchased_90d == 2
