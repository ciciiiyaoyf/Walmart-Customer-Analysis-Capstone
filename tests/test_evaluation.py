"""Statistical identities and leakage guards, without fitting expensive models."""
import numpy as np
import pandas as pd
import pytest

from src.evaluation import cluster_dr_interval, dr_components, evaluate_policy, select_candidate
from src.modeling import ACTIONS, CANDIDATES


def test_dr_exact_outcome_nuisance_recovers_policy_reward_for_every_row():
    potential = np.array([[0., 3., -2., 1.], [7., 9., 1., 11.], [-1., 0., 6., 3.]])
    observed = np.array(["PCT_10", "NO_OFFER", "PCT_20"])
    policy = np.array(["FREE_SHIP", "PCT_10", "NO_OFFER"])
    y = potential[np.arange(3), [ACTIONS.index(a) for a in observed]]
    score, _ = dr_components(y, observed, policy, potential, [.2, .4, .1])
    np.testing.assert_allclose(score, [1., 9., -1.])


def test_dr_is_unbiased_under_known_nonuniform_randomization_with_wrong_reward_model():
    # Enumerate the logging randomization exactly, avoiding a stochastic/flaky test.
    probability = np.array([.1, .2, .3, .4])
    potential = np.array([[2., 6., -1., 4.], [0., -3., 9., 2.]])
    target = ["PCT_20", "FREE_SHIP"]
    for rewards, policy in zip(potential, target):
        wrong_mu = np.tile([8., 1., 3., -5.], (4, 1))
        score, _ = dr_components(rewards, ACTIONS, [policy] * 4, wrong_mu, probability)
        assert probability @ score == pytest.approx(rewards[ACTIONS.index(policy)])


def test_paired_no_offer_lift_and_interval_are_exactly_zero():
    data = pd.DataFrame({"customer_id": [f"c{i}" for i in range(8)],
                         "assignment_id": [f"a{i}" for i in range(8)],
                         "campaign_id": "CP07", "offer_type": ACTIONS * 2,
                         "ncm_14d": [4, 2, 1, 5, 0, 3, -2, 1],
                         "promo_cost_14d": [0, 2, 3, 1, 0, 1, 4, 1],
                         "prior_emails_delivered": 0, "orders_90d": 0})
    allocation = pd.DataFrame({"customer_id": data.customer_id.iloc[::-1],
                               "recommended_action": "NO_OFFER", "estimated_promo_cost": 0.})
    mu = np.tile([1., 2., 3., 4.], (len(data), 1))
    costs = np.tile([0., 1., 2., .2], (len(data), 1))
    summary, scores = evaluate_policy(data, allocation, mu, costs, "ALL_NO_OFFER", "CP07", draws=100)
    assert summary["dr_incremental_ncm_per_customer"] == 0
    assert summary["ci_low"] == summary["ci_high"] == 0
    assert summary["hajek_incremental_ncm_per_customer"] == 0
    assert summary["dr_promo_spend"] == 0
    assert scores.dr_incremental_score.eq(0).all()


@pytest.mark.parametrize("p", [[0., .25], [-.1, .25], [1.1, .25], [np.nan, .25]])
def test_invalid_logging_probabilities_are_rejected(p):
    with pytest.raises(ValueError):
        dr_components([1., 2.], ["NO_OFFER", "PCT_10"], ["PCT_10", "NO_OFFER"], np.ones((2, 4)), p)


@pytest.mark.parametrize("invalid", ["outcome_matrix", "short_observed", "short_policy", "action_matrix"])
def test_outcome_and_action_vectors_cannot_silently_broadcast(invalid):
    y = np.array([1., 2.])
    observed = np.array(["NO_OFFER", "PCT_10"])
    policy = np.array(["PCT_10", "NO_OFFER"])
    if invalid == "outcome_matrix":
        y = y[:, None]
    elif invalid == "short_observed":
        observed = observed[:1]
    elif invalid == "short_policy":
        policy = policy[:1]
    else:
        observed = observed[:, None]
    with pytest.raises(ValueError):
        dr_components(y, observed, policy, np.ones((2, 4)), [.25, .25])


def development_scores(positive=False):
    rows = []
    changes = {"ORIGINAL_TLEARNER": [0., 0., 0., 0.],
               "SHRUNK_TLEARNER": [2., 2., 2., 2.] if positive else [4., -4., 0., 0.],
               "ACTION_MEAN": [-1., -1., -1., -1.],
               "POOLED_XGBOOST": [0., 0., 0., 0.]}
    for campaign in ["CP04", "CP05", "CP06"]:
        for customer in range(4):
            for method in CANDIDATES:
                rows.append({"assignment_id": f"{campaign}-{customer}", "customer_id": f"c{customer}",
                             "campaign_id": campaign, "policy": method,
                             "dr_incremental_score": float(customer + changes[method][customer])})
    return pd.DataFrame(rows)


def test_model_selection_refuses_any_cp07_rows():
    scores = development_scores()
    scores.loc[0, "campaign_id"] = "CP07"
    with pytest.raises(ValueError, match="CP04"):
        select_candidate(scores, draws=100)


def test_model_selection_retains_original_without_positive_adjusted_lower_bound():
    selected, comparisons = select_candidate(development_scores(), draws=500)
    assert selected == "ORIGINAL_TLEARNER"
    assert not comparisons.eligible_to_replace.any()


def test_model_selection_can_choose_clear_development_gain():
    selected, comparisons = select_candidate(development_scores(positive=True), draws=500)
    assert selected == "SHRUNK_TLEARNER"
    assert comparisons.set_index("candidate").loc["SHRUNK_TLEARNER", "adjusted_ci_low"] == pytest.approx(2.)


def test_cluster_bootstrap_reestimates_p_using_full_population_even_for_subgroup():
    rows = []
    for customer in range(60):
        for campaign in range(2):
            rows.append({"assignment_id": f"{customer}-{campaign}", "customer_id": customer,
                         "campaign_id": f"CP0{campaign + 4}", "observed_action": ACTIONS[(customer + campaign) % 4],
                         "direct_component": customer / 10 + campaign,
                         "residual_component": 1. + (customer % 7) / 10,
                         # This subset has NO_OFFER/PCT_10 only; estimating logging
                         # frequencies on the subset would give the wrong answer.
                         "included": customer % 4 == 0})
    frame = pd.DataFrame(rows)
    draws, seed = 200, 19
    actual = cluster_dr_interval(frame, included=frame.included.to_numpy(), draws=draws, seed=seed)

    def direct_reference(sample):
        frequencies = sample.groupby(["campaign_id", "observed_action"]).size()
        totals = sample.groupby("campaign_id").size()
        p = np.array([frequencies.loc[(c, a)] / totals.loc[c]
                      for c, a in zip(sample.campaign_id, sample.observed_action)])
        scores = sample.direct_component.to_numpy() + sample.residual_component.to_numpy() / p
        return scores[sample.included.to_numpy()].mean()

    expected_point = direct_reference(frame)
    rng = np.random.default_rng(seed)
    boot = []
    for ids in rng.integers(0, 60, size=(draws, 60)):
        row_indices = (ids[:, None] * 2 + np.array([0, 1])).ravel()
        sampled = frame.iloc[row_indices]
        assert sampled.groupby("campaign_id").observed_action.nunique().eq(4).all()
        assert sampled.included.any()
        boot.append(direct_reference(sampled))
    expected_bounds = np.quantile(boot, [.025, .975])
    np.testing.assert_allclose(actual, [expected_point, *expected_bounds], rtol=1e-12, atol=1e-12)
    # Proves this differs from recalculating propensities in the subgroup alone.
    assert abs(expected_point - direct_reference(frame[frame.included])) > 1


def test_production_decomposition_path_selects_clear_paired_gain():
    scores = development_scores(positive=True)
    scores["observed_action"] = scores.customer_id.map({f"c{i}": a for i, a in enumerate(ACTIONS)})
    scores["direct_component"] = scores.dr_incremental_score
    scores["residual_component"] = 0.
    selected, comparisons = select_candidate(scores, draws=100)
    assert selected == "SHRUNK_TLEARNER"
    assert comparisons.set_index("candidate").loc["SHRUNK_TLEARNER", "adjusted_ci_low"] == pytest.approx(2.)


def test_evaluation_rejects_extra_allocation_customers_instead_of_dropping_them():
    data = pd.DataFrame({"customer_id": list("abcd")})
    allocation = pd.DataFrame({"customer_id": list("abcde"), "recommended_action": "NO_OFFER"})
    with pytest.raises(ValueError, match="exactly"):
        evaluate_policy(data, allocation, np.zeros((4, 4)), np.zeros((4, 4)), "ALL_NO_OFFER", "CP07")


def _cost_selection_data(cp06_cost, cp07_cost):
    return pd.DataFrame([{"campaign_id": f"CP0{campaign}", "campaign_number": campaign,
                          "offer_type": action,
                          "promo_cost_14d": cp06_cost if campaign == 6 else cp07_cost if campaign == 7 else 1.}
                         for action in ACTIONS[1:] for campaign in range(1, 8)])


def _fake_cost_factory(monkeypatch, predicted_cost):
    import src.evaluation as evaluation
    fitted, predicted = [], []

    class FakeRegressor:
        def fit(self, X, y):
            self.fit_campaigns = tuple(X.campaign_number.tolist())
            self.fit_targets = tuple(y.tolist())
            fitted.append(self)
            return self

        def predict(self, X):
            predicted.append(tuple(X.campaign_number.tolist()))
            return np.full(len(X), predicted_cost, dtype=float)

    monkeypatch.setattr(evaluation, "make_xgb_pipe", lambda *args, **kwargs: FakeRegressor())
    return FakeRegressor, fitted, predicted


def test_cost_family_selection_frozen_at_cp06_ignores_cp07_but_refits_all_rows(monkeypatch):
    from src.evaluation import fit_cost_models
    fake_type, fitted, predicted = _fake_cost_factory(monkeypatch, predicted_cost=10.)
    # XGB wins on CP06 (10 versus prior mean 1). CP07=1 would instead
    # favor the mean if it were mistakenly used for family selection.
    data = _cost_selection_data(cp06_cost=10., cp07_cost=1.)
    result = fit_cost_models(data, ["campaign_number"], [], {}, selection_campaign="CP06")
    assert all(isinstance(model, fake_type) for model in result.values())
    assert predicted == [(6,)] * 3
    assert len(fitted) == 6
    assert [model.fit_campaigns for model in fitted[::2]] == [(1, 2, 3, 4, 5)] * 3
    assert all(model.fit_campaigns == (1, 2, 3, 4, 5, 6, 7) for model in result.values())


def test_selected_cost_mean_refits_on_all_passed_rows_without_using_cp07_for_selection(monkeypatch):
    from src.evaluation import fit_cost_models
    from src.modeling import ConstantPredictor
    _, fitted, predicted = _fake_cost_factory(monkeypatch, predicted_cost=100.)
    # Mean wins CP06 exactly. CP07=71 would favor XGB if used to select,
    # but is legitimately included in the final July mean after selection.
    data = _cost_selection_data(cp06_cost=1., cp07_cost=71.)
    result = fit_cost_models(data, ["campaign_number"], [], {}, selection_campaign="CP06")
    assert all(isinstance(model, ConstantPredictor) for model in result.values())
    assert predicted == [(6,)] * 3
    assert len(fitted) == 3
    assert all(model.fit_campaigns == (1, 2, 3, 4, 5) for model in fitted)
    for model in result.values():
        np.testing.assert_allclose(model.predict(pd.DataFrame({"campaign_number": [8]})), [(6 + 71) / 7])


def test_cost_selection_defaults_to_last_passed_training_campaign(monkeypatch):
    from src.evaluation import fit_cost_models
    _, fitted, predicted = _fake_cost_factory(monkeypatch, predicted_cost=10.)
    data = _cost_selection_data(cp06_cost=10., cp07_cost=1.)
    data = data[data.campaign_id.lt("CP06")].copy()
    data.loc[data.campaign_id.eq("CP05"), "promo_cost_14d"] = 10.
    result = fit_cost_models(data, ["campaign_number"], [], {})
    assert predicted == [(5,)] * 3
    assert [model.fit_campaigns for model in fitted[::2]] == [(1, 2, 3, 4)] * 3
    assert all(model.fit_campaigns == (1, 2, 3, 4, 5) for model in result.values())
