import numpy as np
import pandas as pd
import pytest

from src.diagnostics import build_accrued_target, cold_history_mask, support_report, verified_inactive_mask


def accrual_fixture():
    history = pd.DataFrame({"assignment_id": ["a1", "a2"], "customer_id": ["c1", "c2"],
                            "campaign_id": ["CP01", "CP02"], "send_ts": pd.to_datetime(["2025-01-01", "2025-02-01"]),
                            "net_revenue_14d": [100., 200.], "cogs_14d": [40., 50.], "shipping_cost_14d": [5., 5.],
                            "ncm_14d": [45., 145.]})
    orders = pd.DataFrame({"order_id": ["o1", "o2", "cancelled", "outside"], "customer_id": ["c1", "c2", "c1", "c1"],
                           "campaign_id": ["CP01", "CP02", "CP01", "CP01"],
                           "order_ts": pd.to_datetime(["2025-01-02", "2025-02-02", "2025-01-03", "2025-01-20"]),
                           "is_completed": [True, True, False, True]})
    returns = pd.DataFrame({"order_id": ["o1", "o1", "o1", "o1", "cancelled", "outside"],
                            "return_ts": pd.to_datetime(["2025-01-10", "2025-02-10", "2025-02-15", "2025-03-01", "2025-01-10", "2025-01-21"]),
                            "refund_amount": [10., 20., 7., 99., 1000., 1000.]})
    return history, orders, returns


def test_accrued_target_uses_later_returns_once_and_excludes_immature_rows():
    history, orders, returns = accrual_fixture()
    accrued = build_accrued_target(history, orders, returns, as_of_ts="2025-02-15")
    assert accrued.assignment_id.tolist() == ["a1"]
    assert accrued.iloc[0].refund_amount_accrued_45d == 30.
    assert accrued.iloc[0].ncm_accrued_45d == 25.
    assert accrued.iloc[0].ncm_14d == 45.  # primary target remains unchanged
    assert accrued.accrual_end_ts.le(accrued.label_as_of_ts).all()


def test_future_refunds_do_not_change_available_target():
    history, orders, returns = accrual_fixture()
    original = build_accrued_target(history, orders, returns, as_of_ts="2025-02-15")
    extra = pd.DataFrame({"order_id": ["o1"], "return_ts": pd.to_datetime(["2025-02-16"]), "refund_amount": [10000.]})
    changed = build_accrued_target(history, orders, pd.concat([returns, extra]), as_of_ts="2025-02-15")
    pd.testing.assert_frame_equal(original, changed)


def test_refund_at_or_beyond_horizon_stays_excluded_even_when_observed():
    history, orders, returns = accrual_fixture()
    accrued = build_accrued_target(history, orders, returns, as_of_ts="2025-04-01")
    assert accrued.set_index('assignment_id').loc['a1', 'refund_amount_accrued_45d'] == 30.
    assert set(accrued.assignment_id) == {'a1', 'a2'}


def test_no_mature_history_returns_empty_typed_target():
    history, orders, returns = accrual_fixture()
    accrued = build_accrued_target(history, orders, returns, as_of_ts="2025-01-14")
    assert accrued.empty
    assert "ncm_accrued_45d" in accrued


def test_support_report_distinguishes_missingness_range_and_new_categories():
    train = pd.DataFrame({"x": [0., 1., 2., np.nan], "cat": ["A", "B", "A", None]})
    scoring = pd.DataFrame({"x": [0., 3., np.nan, np.nan], "cat": ["A", "C", None, None]})
    report = support_report(train, scoring, ["x"], ["cat"]).set_index("feature")
    assert report.loc['x', 'scoring_missing_rate'] == .5
    assert report.loc['x', 'train_missing_rate'] == .25
    assert report.loc['x', 'scoring_out_of_range_rate'] == .25
    assert report.loc['x', 'scoring_zero_rate'] == .25
    assert report.loc['cat', 'scoring_unseen_category_rate'] == .25


def test_cold_history_mask_does_not_treat_unknown_as_inactive():
    data = pd.DataFrame({"prior_emails_delivered": [0., 0., 1., np.nan], "orders_90d": [0., 1., 0., 0.]})
    assert cold_history_mask(data).tolist() == [True, False, False, False]


def test_support_report_flags_training_feature_with_no_observations():
    report = support_report(pd.DataFrame({'x': [np.nan, np.nan]}), pd.DataFrame({'x': [0., 1.]}), ['x'], [])
    assert report.iloc[0].scoring_no_training_support_rate == 1.
    assert pd.isna(report.iloc[0].scoring_out_of_range_rate)


def test_first_exposure_mask_uses_observed_counts_without_filling_unknowns():
    data = pd.DataFrame({'prior_emails_delivered': [0., 0., 0., 1.],
                         'observed_orders_90d': pd.Series([0, 1, pd.NA, 0], dtype='Float64'),
                         'orders_90d': [99., 0., 0., 0.]})
    assert cold_history_mask(data).tolist() == [True, False, False, False]


def test_verified_inactivity_requires_complete_coverage_and_known_zero():
    data = pd.DataFrame({
        'orders_90d_coverage_status': ['verified_complete', 'verified_complete',
                                       'unverified', 'unverified', 'verified_complete', pd.NA],
        'complete_orders_90d': pd.Series([0., 2., 0., pd.NA, pd.NA, 0.], dtype='Float64'),
        'observed_orders_90d': [0., 2., 0., 0., 0., 0.],
    })
    assert verified_inactive_mask(data).tolist() == [True, False, False, False, False, False]
    assert not verified_inactive_mask(data[['observed_orders_90d']]).any()
