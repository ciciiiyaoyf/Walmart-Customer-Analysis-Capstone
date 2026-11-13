"""Observed transaction features retain decision cutoffs without claiming coverage."""
import numpy as np
import pandas as pd
import pytest

from src.features import (
    CATEGORICAL,
    FEATURES,
    LEGACY_CATEGORICAL,
    LEGACY_FEATURES,
    OBSERVED_FEATURE_RENAMES,
    build_features,
    feature_series,
    observed_feature_name,
)


@pytest.fixture
def history():
    cutoff = pd.Timestamp("2025-06-20 10:00:00")
    spine = pd.DataFrame({"assignment_id": ["now"], "customer_id": ["c1"], "send_ts": [cutoff]})
    tables = {
        "customers": pd.DataFrame({
            "customer_id": ["c1"], "signup_date": pd.to_datetime(["2024-01-01"]),
            "is_member": [0], "marketing_consent": [1], "region": ["west"],
            "acquisition_channel": ["web"], "customer_tier": ["Standard"],
            "preferred_category": ["shoes"], "shipping_zone": ["z1"],
        }),
        "orders": pd.DataFrame({
            "order_id": ["recent", "old"], "customer_id": ["c1", "c1"],
            "campaign_id": ["CP01", "CP01"],
            "order_ts": [cutoff - pd.Timedelta(days=30), cutoff - pd.Timedelta(days=100)],
            "is_completed": [True, True], "net_revenue": [100., 80.], "cogs": [40., 30.],
            "shipping_cost": [5., 5.], "merchandise_gross_revenue": [110., 80.],
            "discount_amount": [10., 0.], "shipping_revenue": [0., 0.], "offer_redeemed": [0, 0],
        }),
        "returns": pd.DataFrame({
            "order_id": ["recent"], "return_ts": [cutoff - pd.Timedelta(days=20)],
            "refund_amount": [10.],
        }),
        "historical_assignments": pd.DataFrame({
            "assignment_id": ["past"], "customer_id": ["c1"], "campaign_id": ["CP01"],
            "send_ts": [cutoff - pd.Timedelta(days=60)], "offer_type": ["NO_OFFER"],
            "email_delivered": [1],
        }),
        "web_sessions": pd.DataFrame({
            "customer_id": ["c1"], "session_ts": [cutoff - pd.Timedelta(days=1)],
            "pages_viewed": [2], "added_to_cart": [1], "category_viewed": ["shoes"],
        }),
        "email_events": pd.DataFrame({
            "assignment_id": ["past"], "campaign_id": ["CP01"], "customer_id": ["c1"],
            "event_ts": [cutoff - pd.Timedelta(days=2)], "event_type": ["open"],
        }),
        "order_items": pd.DataFrame({
            "order_id": ["recent", "old"], "product_id": ["p1", "p1"], "quantity": [2, 1],
        }),
        "products": pd.DataFrame({"product_id": ["p1"], "category": ["shoes"]}),
    }
    return spine, tables


def test_observed_names_preserve_predictor_order_and_transaction_values(history):
    spine, tables = history
    result, _ = build_features(spine, tables)
    assert result.columns.tolist() == ["assignment_id", "customer_id", "send_ts"] + FEATURES
    assert len(FEATURES) == len(set(FEATURES)) == len(LEGACY_FEATURES) == 43
    assert FEATURES == [observed_feature_name(feature) for feature in LEGACY_FEATURES]
    assert CATEGORICAL == [observed_feature_name(feature) for feature in LEGACY_CATEGORICAL]
    assert not set(OBSERVED_FEATURE_RENAMES).intersection(result.columns)
    assert "prior_offer_redemption_rate" in result.columns
    row = result.iloc[0]
    expected = {
        "observed_prior_order_count": 2,
        "observed_orders_30d": 1,
        "observed_orders_90d": 1,
        "observed_days_since_last_order": 30,
        "observed_gross_revenue_90d": 110.,
        "observed_net_revenue_90d": 100.,
        "observed_avg_order_value_90d": 100.,
        "observed_discount_amount_90d": 10.,
        "observed_cogs_90d": 40.,
        "observed_shipping_revenue_90d": 0.,
        "observed_shipping_cost_90d": 5.,
        "observed_historical_ncm_90d": 45.,
        "observed_returns_90d": 1,
        "observed_refund_amount_90d": 10.,
        "observed_historical_return_rate": 0.5,
        "observed_units_purchased_90d": 2,
        "observed_distinct_products_90d": 1,
        "observed_distinct_categories_90d": 1,
        "observed_top_purchase_category_90d": "shoes",
    }
    assert row[list(expected)].to_dict() == expected


@pytest.mark.parametrize("inclusive", [False, True])
def test_events_after_decision_cannot_change_any_predictor(history, inclusive):
    spine, tables = history
    expected, _ = build_features(spine, tables, inclusive=inclusive)
    future = spine.send_ts.iloc[0] + pd.Timedelta(seconds=1)
    changed = {name: table.copy(deep=True) for name, table in tables.items()}
    time_columns = {
        "orders": "order_ts", "returns": "return_ts", "historical_assignments": "send_ts",
        "web_sessions": "session_ts", "email_events": "event_ts",
    }
    for name, timestamp in time_columns.items():
        extra = changed[name].iloc[[0]].copy()
        extra[timestamp] = future
        if name == "orders":
            extra["order_id"] = "future_order"
        if name == "historical_assignments":
            extra["assignment_id"] = "future_assignment"
        changed[name] = pd.concat([changed[name], extra], ignore_index=True)
    extra_items = changed["order_items"].iloc[[0]].assign(order_id="future_order", quantity=999)
    changed["order_items"] = pd.concat([changed["order_items"], extra_items], ignore_index=True)
    actual, _ = build_features(spine, changed, inclusive=inclusive)
    pd.testing.assert_frame_equal(actual, expected)


def test_decision_relative_window_and_cutoff_boundaries_are_preserved(history):
    spine, tables = history
    cutoff = spine.send_ts.iloc[0]
    # A completed order exactly 90 days ago belongs to the original 90-day window.
    tables["orders"].loc[1, "order_ts"] = cutoff - pd.Timedelta(days=90)
    # The decision timestamp is allowed for scoring, but excluded for training.
    exact = tables["orders"].iloc[[0]].assign(order_id="at_cutoff", order_ts=cutoff)
    tables["orders"] = pd.concat([tables["orders"], exact], ignore_index=True)
    historical, _ = build_features(spine, tables)
    scoring, _ = build_features(spine, tables, inclusive=True)
    assert historical.loc[0, "observed_orders_90d"] == 2
    assert scoring.loc[0, "observed_orders_90d"] == 3
    assert historical.loc[0, "observed_orders_30d"] == 1
    assert scoring.loc[0, "observed_orders_30d"] == 2


def test_stale_history_stays_observed_zero_instead_of_shifting_the_window(history):
    spine, tables = history
    tables["orders"]["order_ts"] = pd.Timestamp("2025-01-09")
    tables["returns"]["return_ts"] = pd.Timestamp("2025-01-10")
    result, _ = build_features(spine, tables, inclusive=True)
    assert result.loc[0, "send_ts"] == pd.Timestamp("2025-06-20 10:00:00")
    assert result.loc[0, "observed_prior_order_count"] == 2
    assert result.loc[0, "observed_orders_30d"] == result.loc[0, "observed_orders_90d"] == 0
    assert result.loc[0, "observed_days_since_last_order"] > 160


def test_series_prefers_observed_column_over_unknown_complete_counts():
    data = pd.DataFrame({"observed_orders_90d": [0., np.nan], "orders_90d": [np.nan, 7.]})
    pd.testing.assert_series_equal(feature_series(data, "orders_90d"), data.observed_orders_90d)
    pd.testing.assert_series_equal(feature_series(data, "observed_orders_90d"), data.observed_orders_90d)
    legacy_only = data.drop(columns="observed_orders_90d")
    pd.testing.assert_series_equal(feature_series(legacy_only, "orders_90d"), legacy_only.orders_90d)
    with pytest.raises(KeyError):
        feature_series(pd.DataFrame(), "orders_90d")
