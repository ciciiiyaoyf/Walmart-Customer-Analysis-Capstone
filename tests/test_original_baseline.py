"""The active baseline must preserve the original purchase-ranking behavior."""
import joblib
import numpy as np
import pandas as pd

from src.common import ROOT
from src.modeling import feature_lists, fit_logistic_baseline


def test_baseline_reproduces_archived_original_cp07_predictions():
    """Catch accidental replacement by an experimental feature representation."""
    history = pd.read_parquet(ROOT / "data/processed/historical_feature_outcome.parquet")
    train = history.loc[history.campaign_id.lt("CP07")]
    test = history.loc[history.campaign_id.eq("CP07")]
    num, cat = feature_lists()
    model = fit_logistic_baseline(train, num, cat)
    actual = pd.Series(model.predict_proba(test[num + cat])[:, 1],
                       index=test.assignment_id, name="prediction")
    archived = pd.read_csv(ROOT / "outputs/experiments/propensity/cp07_predictions.csv")
    expected = archived.loc[archived.model.eq("original")].set_index("assignment_id")
    assert actual.index.is_unique and expected.index.is_unique
    assert set(actual.index) == set(expected.index)
    np.testing.assert_allclose(actual.loc[expected.index].to_numpy(),
                               expected.prediction.to_numpy(), rtol=0, atol=1e-10)


def test_missing_category_representation_does_not_change_saved_model_predictions(tmp_path):
    """Live joins and Parquet nulls must activate the same learned category."""
    training = pd.DataFrame({"signal": [0., 1., 2., 3., 4., 5., 6., 7.],
                             "segment": [None, "a", "b", None, "a", "b", None, "a"],
                             "purchase_14d": [0, 0, 1, 1, 0, 1, 1, 0]})
    model = fit_logistic_baseline(training, ["signal"], ["segment"])
    scoring = pd.DataFrame({"signal": [1.5, 3.5, 5.5], "segment": [None, "a", None]})
    expected = model.predict_proba(scoring)
    artifact = tmp_path / "baseline.joblib"
    joblib.dump(model, artifact)
    reloaded = joblib.load(artifact)
    for null in [None, np.nan, pd.NA]:
        equivalent = scoring.copy()
        equivalent.loc[[0, 2], "segment"] = null
        np.testing.assert_array_equal(model.predict_proba(equivalent), expected)
        np.testing.assert_array_equal(reloaded.predict_proba(equivalent), expected)
