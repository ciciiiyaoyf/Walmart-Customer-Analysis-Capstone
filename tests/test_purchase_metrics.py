"""Purchase metrics use explicit operating points without held-out tuning."""
import numpy as np
import pandas as pd
import pytest

from src.propensity import purchase_metrics, choose_purchase_threshold
import scripts.experiment_propensity as experiment


def test_average_precision_alias_and_recall_at_explicit_threshold():
    y = [1, 0, 1, 0]
    p = [.8, .7, .4, .1]
    ranking = purchase_metrics(y, p)
    assert ranking['average_precision'] == pytest.approx(5 / 6)
    assert ranking['pr_auc'] == ranking['average_precision']
    assert 'recall' not in ranking and 'threshold' not in ranking
    classified = purchase_metrics(y, p, threshold=.6)
    assert classified['recall'] == classified['precision'] == .5
    assert classified['predicted_positive_share'] == .5
    assert [classified[k] for k in ['tn', 'fp', 'fn', 'tp']] == [1, 1, 1, 1]
    # Recall increases when everyone is flagged, but precision reveals the trade-off.
    all_flagged = purchase_metrics(y, p, threshold=0.)
    assert all_flagged['recall'] == 1 and all_flagged['precision'] == .5


def test_development_threshold_maximizes_f1_and_breaks_ties_deterministically():
    assert choose_purchase_threshold([1, 0, 1, 0], [.8, .7, .4, .1]) == .4
    # The first-only and all-positive classifications have equal F1 = 2/3.
    assert choose_purchase_threshold([1, 0, 0, 1], [.9, .8, .7, .6]) == .6


def test_undefined_recall_is_not_reported_as_zero_performance():
    metrics = purchase_metrics([0, 0], [.1, .8], threshold=.5)
    assert np.isnan(metrics['recall']) and np.isnan(metrics['average_precision'])
    assert np.isnan(metrics['roc_auc'])
    assert metrics['fp'] == 1 and metrics['tn'] == 1
    with pytest.raises(ValueError, match='development purchasers'):
        choose_purchase_threshold([0, 0], [.1, .8])


@pytest.mark.parametrize('y,p,threshold', [([0, 1], [.1, np.nan], .5),
    ([0, 2], [.1, .5], .5), ([0, 1], [-.1, .5], .5), ([0], [.1, .5], .5),
    ([0, 1], [.1, .5], np.nan), ([0, 1], [.1, .5], 1.1)])
def test_invalid_probabilities_labels_or_thresholds_are_rejected(y, p, threshold):
    with pytest.raises(ValueError):
        purchase_metrics(y, p, threshold=threshold)


def test_threshold_development_ignores_cp07_and_respects_temporal_training(monkeypatch):
    data = pd.DataFrame({'campaign_id': np.repeat([f'CP0{i}' for i in range(1, 8)], 4),
                         'purchase_14d': np.tile([1, 0, 1, 0], 7),
                         'signal': np.tile([.8, .7, .4, .1], 7)})
    data['assignment_id'] = np.arange(len(data))
    data['customer_id'] = np.arange(len(data))
    fits, scores = [], []
    class FakeModel:
        def fit(self, X, y):
            self.last_train = data.loc[X.index, 'campaign_id'].max()
            fits.append(self.last_train)
            return self
        def predict_proba(self, X):
            current = data.loc[X.index, 'campaign_id'].min()
            assert self.last_train < current
            scores.append(current)
            return np.column_stack([1-X.signal, X.signal])
    monkeypatch.setattr(experiment, 'make_propensity_pipe', lambda *args, **kwargs: FakeModel())
    monkeypatch.setattr(experiment, 'fit_purchase_model',
        lambda frame, num, cat, config: FakeModel().fit(frame[num+cat], frame.purchase_14d))
    config = {'mode': 'raw', 'C': .01, 'calibration': 'none'}
    original, predictions = experiment.development_purchase_thresholds(data, ['signal'], [], config)
    contaminated = data.copy()
    contaminated.loc[contaminated.campaign_id.eq('CP07'), ['purchase_14d', 'signal']] = 999
    revised, again = experiment.development_purchase_thresholds(contaminated, ['signal'], [], config)
    assert original == revised and original['original'] == original['selected'] == .4
    pd.testing.assert_frame_equal(predictions, again)
    assert set(scores) == {'CP04', 'CP05', 'CP06'} and max(fits) == 'CP05'


def test_bootstrap_uses_frozen_thresholds_and_reports_recall_difference(monkeypatch):
    base = pd.DataFrame({'assignment_id': range(8), 'customer_id': range(8),
                         'purchase_14d': [1, 0, 1, 0, 1, 0, 1, 0]})
    predictions = pd.concat([
        base.assign(model='original', prediction=[.8, .7, .4, .1]*2),
        base.assign(model='selected', prediction=[.8, .7, .6, .1]*2)], ignore_index=True)
    monkeypatch.setattr(experiment, 'choose_purchase_threshold',
        lambda *args: pytest.fail('Held-out/bootstrap labels must not select a threshold'))
    result = experiment.paired_purchase_intervals(predictions, n_bootstrap=50,
        thresholds={'original': .5, 'selected': .5}).set_index('metric')
    assert result.loc['recall', 'original'] == .5
    assert result.loc['recall', 'selected'] == 1.
    assert result.loc['recall', 'difference_selected_minus_original'] == .5
    assert result.loc['recall', 'original_threshold'] == .5
    assert result.loc['recall', 'selected_threshold'] == .5
    assert result.loc['recall', 'thresholds_fixed_in_bootstrap']
    assert 'average_precision' in result.index and 'pr_auc' in result.index
    without = experiment.paired_purchase_intervals(predictions, n_bootstrap=50)
    assert 'recall' not in set(without.metric)


def test_bootstrap_rejects_incomplete_assignment_pairs():
    predictions = pd.DataFrame({'assignment_id': [1, 2, 1], 'customer_id': [1, 2, 1],
                                'purchase_14d': [1, 0, 1], 'model': ['original', 'original', 'selected'],
                                'prediction': [.9, .1, .9]})
    with pytest.raises(ValueError, match='probabilities'):
        experiment.paired_purchase_intervals(predictions, n_bootstrap=10)
