import numpy as np
import pandas as pd
import pytest

from src.metric_diagnostics import (regression_diagnostic, profit_bias_table, calibration_table,
                                    hurdle_components, hurdle_purchase_table, purchaser_margin_table)
from src.profit_models import HurdleActionModel
from src.modeling import ACTIONS


class FixedProbability:
    def predict_proba(self, x):
        return np.tile([.75, .25], (len(x), 1))


class FixedMargin:
    def predict(self, x):
        return np.full(len(x), -20.)


def example_rows():
    return pd.DataFrame({"assignment_id": ["a", "b", "c", "d"], "customer_id": [1, 2, 3, 4],
        "campaign_id": "CP07", "offer_type": ["NO_OFFER", "NO_OFFER", "PCT_10", "PCT_10"],
        "ncm_14d": [-12., 0., 24., 0.], "purchase_14d": [1, 0, 1, 0], "x": [0, 1, 2, 3]})


def prediction_rows():
    return example_rows().assign(model="test", factual_prediction=[-3., -3., 4., 4.])


def test_profit_bias_has_signed_definition_and_preserves_loss_targets():
    result = regression_diagnostic([-10., 0., 20.], [-8., 2., 22.], [1, 2, 3], 100)
    assert result["mean_prediction_bias"] == 2.
    assert result["mae"] == 2. and result["rmse"] == 2.
    assert result["bias_ci_low"] == 2. and result["bias_ci_high"] == 2.


def test_bias_overall_reconciles_offer_groups_and_empty_groups_stay_unknown():
    result = profit_bias_table(prediction_rows(), 50).set_index("scope")
    assert result.loc["pooled", "mean_prediction_bias"] == pytest.approx(-2.5)
    assert result.loc["PCT_20", "n"] == 0
    assert np.isnan(result.loc["PCT_20", "rmse"])
    arms = result.loc[ACTIONS]
    assert (arms.mean_prediction_bias.fillna(0)*arms.n).sum()/arms.n.sum() == result.loc["pooled", "mean_prediction_bias"]


def test_regression_rejects_invalid_values_and_constant_r2_is_unknown():
    with pytest.raises(ValueError, match="Finite"):
        regression_diagnostic([1., np.nan], [1., 2.], [1, 2], 50)
    with pytest.raises(ValueError, match="Matching"):
        regression_diagnostic([1.], [1., 2.], [1, 2], 50)
    assert np.isnan(regression_diagnostic([2., 2.], [1., 3.], [1, 2], 50)["r2"])


def test_calibration_preserves_ties_and_is_invariant_to_row_order():
    data = prediction_rows()
    data["factual_prediction"] = [1., 1., 1., 2.]
    first = calibration_table(data, bins=5, bootstrap_draws=50)
    second = calibration_table(data.iloc[::-1], bins=5, bootstrap_draws=50)
    pd.testing.assert_frame_equal(first, second)
    pooled = first.loc[first.scope.eq("pooled")]
    assert pooled.n.sum() == len(data)
    assert len(pooled.loc[pooled.prediction_min.le(1) & pooled.prediction_max.ge(1)]) == 1
    assert pooled.iloc[0].n == 3


def test_constant_predictions_create_one_calibration_bin():
    data = prediction_rows().assign(factual_prediction=2.)
    result = calibration_table(data, bootstrap_draws=50)
    assert len(result.loc[result.scope.eq("pooled")]) == 1
    assert result.loc[result.scope.eq("pooled"), "n"].item() == 4


def test_hurdle_components_preserve_negative_margin_and_probability_product():
    data = example_rows()
    data.index = [5, 5, 2, 2]
    models = {action: HurdleActionModel(FixedProbability(), FixedMargin(), action, -4., .5) for action in ACTIONS}
    original = data.copy(deep=True)
    result = hurdle_components(models, data, ["x"])
    np.testing.assert_allclose(result.purchase_probability, .25)
    np.testing.assert_allclose(result.conditional_margin_prediction, -12.)
    np.testing.assert_allclose(result.factual_prediction, -3.)
    pd.testing.assert_frame_equal(original, data)
    assert result.assignment_id.tolist() == data.assignment_id.tolist()


def test_margin_metrics_only_use_buyers_and_conditional_margin():
    data = prediction_rows().assign(purchase_probability=.25, conditional_margin_prediction=[-12., 999., 24., -999.])
    result = purchaser_margin_table(data, 50).set_index("scope")
    assert result.loc["pooled", "n"] == 2
    assert result.loc["pooled", "rmse"] == 0.
    assert result.loc["pooled", "mae"] == 0.
    assert result.loc["NO_OFFER", "pred_mean"] == -12.


def test_purchase_recall_is_threshold_based_and_undefined_groups_are_missing():
    data = prediction_rows().assign(purchase_probability=[.8, .6, .4, .2], conditional_margin_prediction=10.)
    result = hurdle_purchase_table(data, .5).set_index("scope")
    assert result.loc["pooled", "recall"] == .5
    assert result.loc["pooled", "precision"] == .5
    assert result.loc["pooled", "predicted_positive_share"] == .5
    assert result.loc["pooled", "tp"] == 1 and result.loc["pooled", "fn"] == 1
    assert np.isnan(result.loc["PCT_20", "recall"])
    assert np.isnan(result.loc["PCT_10", "precision"])
    data["purchase_14d"] = 0
    zero = hurdle_purchase_table(data, .5).query("scope == 'pooled'").iloc[0]
    assert np.isnan(zero.average_precision) and np.isnan(zero.recall)


def test_fractional_purchase_labels_are_not_silently_rounded():
    data = prediction_rows().assign(purchase_probability=.5, conditional_margin_prediction=10.)
    data["purchase_14d"] = data.purchase_14d.astype(float)
    data.loc[0, "purchase_14d"] = .5
    with pytest.raises(ValueError, match="Binary"):
        hurdle_purchase_table(data, .5)
    with pytest.raises(ValueError, match="Binary"):
        purchaser_margin_table(data, 50)


def test_duplicate_predictions_and_wrong_action_model_rejected():
    data = prediction_rows()
    with pytest.raises(ValueError, match="One prediction"):
        profit_bias_table(pd.concat([data, data]), 50)
    models = {action: HurdleActionModel(FixedProbability(), None, "FREE_SHIP", -4., 0.) for action in ACTIONS}
    with pytest.raises(ValueError, match="dictionary key"):
        hurdle_components(models, example_rows(), ["x"])


def test_saved_component_check_aligns_decision_ids_and_rejects_wrong_population():
    from scripts.build_metric_diagnostics import align_saved_predictions
    expected = prediction_rows()
    aligned = align_saved_predictions(expected.iloc[::-1], expected)
    np.testing.assert_array_equal(aligned.index, expected.assignment_id)
    with pytest.raises(ValueError, match="assignment sets"):
        align_saved_predictions(expected.iloc[:2], expected)
    modified = expected.copy()
    modified.loc[0, "offer_type"] = "PCT_20"
    with pytest.raises(ValueError, match="offer_type"):
        align_saved_predictions(modified, expected)


def test_threshold_development_never_fits_or_reads_cp07_outcomes(monkeypatch):
    import scripts.build_metric_diagnostics as diagnostics
    seen_training = []
    def fake_fit(train, num, cat, config):
        seen_training.append(set(train.campaign_id))
        assert train.assignment_source.eq("exploration").all()
        return {action: HurdleActionModel(FixedProbability(), None, action, 10., 0.) for action in ACTIONS}
    monkeypatch.setattr(diagnostics, "fit_profit_models", fake_fit)
    history = pd.concat([example_rows().assign(campaign_id=f"CP{i:02}", assignment_id=lambda x: str(i)+x.assignment_id,
                send_ts=pd.Timestamp("2025-01-01")+pd.Timedelta(days=21*i), assignment_source="exploration")
                for i in range(1, 8)], ignore_index=True)
    first = diagnostics.rolling_hurdle_components(history, ["x"], [], {"family": "hurdle"})
    history.loc[history.campaign_id.eq("CP07"), ["purchase_14d", "ncm_14d"]] = [999, np.nan]
    second = diagnostics.rolling_hurdle_components(history, ["x"], [], {"family": "hurdle"})
    pd.testing.assert_frame_equal(first, second)
    assert all("CP07" not in campaigns for campaigns in seen_training)
    assert set(first.campaign_id) == {"CP04", "CP05", "CP06"}
    with pytest.raises(ValueError, match="CP04-CP06"):
        diagnostics.rolling_hurdle_components(history, ["x"], [], {"family": "hurdle"}, campaigns=("CP07",))
