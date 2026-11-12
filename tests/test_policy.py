import itertools

import numpy as np
import pandas as pd
import pytest

from src.policy import (ACTIONS, assert_feasible, build_baseline_policies,
                        get_holdout, measurement_assignment, optimize_policy)


def values(gains, costs=None):
    rows = []
    for i, gain in enumerate(gains):
        for j, action in enumerate(ACTIONS):
            rows.append({"customer_id": f"c{i:02}", "action": action,
                         "predicted_ncm": 20 + gain[j], "estimated_incremental_ncm": gain[j],
                         "estimated_promo_cost": (0 if j == 0 else 1) if costs is None else costs[i][j],
                         "predicted_purchase_probability": (i + 1) / (len(gains) + 1)})
    return pd.DataFrame(rows)


def brute_force_optimum(v, budget, offer_share, pct20_share, holdout=()):
    ids = sorted(v.customer_id.unique())
    indexed = v.set_index(["customer_id", "action"])
    best = -np.inf
    for actions in itertools.product(ACTIONS, repeat=len(ids)):
        if any(cid in holdout and a != "NO_OFFER" for cid, a in zip(ids, actions)):
            continue
        offered = sum(a != "NO_OFFER" for a in actions)
        if offered > np.floor(offer_share * len(ids)) or actions.count("PCT_20") > pct20_share * offered:
            continue
        picked = indexed.loc[list(zip(ids, actions))]
        if picked.estimated_promo_cost.sum() <= budget:
            best = max(best, picked.estimated_incremental_ncm.sum())
    return best


def test_milp_matches_exhaustive_optimum_with_binding_budget_and_holdout():
    v = values([[0, 8, 12, 3], [0, 6, 11, 5], [0, 7, 8, 2], [0, 9, 15, 4]],
               [[0, 4, 8, 1], [0, 3, 7, 1], [0, 4, 5, 1], [0, 4, 9, 1]])
    hold = ["c03"]
    p = optimize_policy(v, budget=7, max_offer_share=.75, max_pct20_share=.5, holdout_ids=hold)
    assert p.estimated_incremental_ncm.sum() == pytest.approx(brute_force_optimum(v, 7, .75, .5, hold))
    assert p.set_index("customer_id").loc["c03", "recommended_action"] == "NO_OFFER"


def test_pct20_denominator_is_offered_customers():
    v = values([[0, 1, 100, 1]] * 10)
    p = optimize_policy(v, budget=100, max_offer_share=.6, max_pct20_share=.15)
    # At most six offers => even one PCT_20 would exceed 15%, not 1/10.
    assert not p.recommended_action.eq("PCT_20").any()
    assert p.recommended_action.ne("NO_OFFER").sum() == 6


def test_tied_policy_and_holdout_are_invariant_to_input_order():
    v = values([[0, 1, 1, 1]] * 12)
    hold = get_holdout(v.customer_id.unique(), share=.25, seed=42)
    assert np.array_equal(hold, get_holdout(v.customer_id.unique()[::-1], share=.25, seed=42))
    first = optimize_policy(v, budget=10, holdout_ids=hold)
    second = optimize_policy(v.sample(frac=1, random_state=9), budget=10, holdout_ids=hold)
    pd.testing.assert_frame_equal(first, second)


def test_zero_value_never_requires_unnecessary_offers():
    p = optimize_policy(values([[0, 0, -1, 0]] * 5), budget=10)
    assert p.recommended_action.eq("NO_OFFER").all()


def test_baselines_remain_feasible_under_tight_budget_and_extreme_historical_mix():
    v = values([[0, 1, 2, 3]] * 20, [[0, 2, 5, .5]] * 20)
    hold = get_holdout(v.customer_id.unique())
    baseline = build_baseline_policies(v, budget=5, historical_mix={"PCT_20": 1}, holdout_ids=hold)
    assert set(baseline) == {"ALL_NO_OFFER", "PROPENSITY_PCT10", "HISTORICAL_MIX", "RANDOM_FREE_SHIP"}
    for p in baseline.values():
        assert assert_feasible(p, budget=5, holdout_ids=hold, expected_customers=v.customer_id.unique())


def test_measurement_arm_includes_policy_no_offer_customers():
    p = optimize_policy(values([[0, -1, -1, -1]] * 5), holdout_ids=["c00"])
    assignment = measurement_assignment(p)
    assert assignment.recommended_action.eq("NO_OFFER").all()
    assert assignment.randomized_arm.value_counts().to_dict() == {"POLICY": 4, "CONTROL": 1}


@pytest.mark.parametrize("problem", ["missing_action", "duplicate", "negative_cost", "missing_holdout"])
def test_bad_inputs_fail_loudly(problem):
    v = values([[0, 1, 2, 3]] * 2)
    kwargs = {}
    if problem == "missing_action":
        v = v.iloc[:-1]
    elif problem == "duplicate":
        v = pd.concat([v, v.iloc[:1]], ignore_index=True)
    elif problem == "negative_cost":
        v.loc[1, "estimated_promo_cost"] = -1
    else:
        kwargs["holdout_ids"] = ["unknown"]
    with pytest.raises(ValueError):
        optimize_policy(v, **kwargs)
