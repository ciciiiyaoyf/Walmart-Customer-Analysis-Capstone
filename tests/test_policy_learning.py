import numpy as np
import pandas as pd
import pytest

from src.modeling import ACTIONS, predict_actions
from src.policy_learning import SegmentEffectRegressor, fit_segment_models, stochastic_free_ship_scores, segment_keys


def example():
    rows = []
    for campaign, level in [("CP01", 0.), ("CP02", 10.)]:
        for member in (0, 1):
            for action in ACTIONS:
                for _ in range(15):
                    effect = (8. if member == 0 else -4.) if action == "FREE_SHIP" else 0.
                    rows.append(dict(campaign_id=campaign, is_member=member, customer_tier="Standard",
                                     offer_type=action, ncm_14d=level + 20*member + effect,
                                     assignment_source="exploration"))
    return pd.DataFrame(rows)


def test_partial_pooling_retains_heterogeneity_and_prediction_needs_no_labels():
    data = example()
    x = pd.DataFrame({"is_member": [0, 1]})
    low = predict_actions(fit_segment_models(data, shrinkage=1.), x)
    high = predict_actions(fit_segment_models(data, shrinkage=10000.), x)
    low_effect, high_effect = low[:, 3]-low[:, 0], high[:, 3]-high[:, 0]
    assert low_effect[0] > 0 > low_effect[1]
    assert np.ptp(high_effect) < np.ptp(low_effect)
    np.testing.assert_allclose(high_effect.mean(), 2., atol=1e-6)


def test_unseen_segment_uses_population_average_contrast():
    model = SegmentEffectRegressor(shrinkage=20.).fit(example())
    x = pd.DataFrame({"is_member": [-999]})
    assert np.isclose(model.predict_action(x, "FREE_SHIP")[0]-model.predict_action(x, "NO_OFFER")[0], 2.)


def test_rejects_nonrandomized_training_or_missing_arm():
    data = example()
    data.loc[0, "assignment_source"] = "business_rule"
    with pytest.raises(ValueError, match="randomized"):
        fit_segment_models(data)
    data = example().query("offer_type != 'PCT_20'")
    with pytest.raises(ValueError, match="Every action"):
        fit_segment_models(data)


def test_campaign_baseline_drift_is_absorbed():
    data = example()
    drift = data.copy()
    drift.loc[drift.campaign_id.eq("CP02"), "ncm_14d"] += 1000.
    x = pd.DataFrame({"is_member": [0, 1]})
    a = predict_actions(fit_segment_models(data), x)
    b = predict_actions(fit_segment_models(drift), x)
    np.testing.assert_allclose(a-a[:, :1], b-b[:, :1], atol=1e-6)


def test_stochastic_score_matches_average_of_all_fixed_recipient_subsets():
    from itertools import combinations
    from src.evaluation import dr_components
    data = pd.DataFrame(dict(assignment_id=list("abcd"), customer_id=list("abcd"),
                             campaign_id=["CP01"]*4, offer_type=ACTIONS, ncm_14d=[10., 30., 4., 18.]))
    mu = np.arange(16).reshape(4, 4).astype(float)
    scores = stochastic_free_ship_scores(data, mu, np.ones((4, 4)), ["a"], budget=2., max_offer_share=.5)
    expected = []
    for chosen in combinations("bcd", 2):
        policy = np.where(data.customer_id.isin(chosen), "FREE_SHIP", "NO_OFFER")
        phi, _ = dr_components(data.ncm_14d, data.offer_type, policy, mu, np.full(4, .25))
        baseline, _ = dr_components(data.ncm_14d, data.offer_type, ["NO_OFFER"]*4, mu, np.full(4, .25))
        expected.append(phi-baseline)
    np.testing.assert_allclose(scores.direct_component+4*scores.residual_component, np.mean(expected, axis=0))
    assert scores.offer_probability.iloc[0] == 0
    with pytest.raises(ValueError, match="nonbinding budget"):
        stochastic_free_ship_scores(data, mu, np.ones((4, 4)), ["a"], budget=1., max_offer_share=.5)


def test_history_segmentation_separates_observed_zero_positive_and_unknown():
    data = pd.DataFrame({'is_member': [0, 0, 0],
                         'observed_orders_90d': pd.Series([0, 2, pd.NA], dtype='Float64'),
                         'orders_90d': [9, 0, 0]})
    assert segment_keys(data, 'membership_history').tolist() == [
        '0|no_observed_recent_order', '0|observed_recent_order', '0|unknown_observed_history']
    legacy = data.drop(columns='observed_orders_90d')
    assert segment_keys(legacy, 'membership_history').tolist() == [
        '0|observed_recent_order', '0|no_observed_recent_order', '0|no_observed_recent_order']


def test_history_feature_rename_preserves_segment_model_predictions():
    legacy = example().assign(orders_90d=lambda frame: frame.is_member * 2)
    current = legacy.rename(columns={'orders_90d': 'observed_orders_90d'})
    old = SegmentEffectRegressor(segment='membership_history').fit(legacy)
    new = SegmentEffectRegressor(segment='membership_history').fit(current)
    for action in ACTIONS:
        np.testing.assert_array_equal(old.predict_action(legacy, action), new.predict_action(current, action))
