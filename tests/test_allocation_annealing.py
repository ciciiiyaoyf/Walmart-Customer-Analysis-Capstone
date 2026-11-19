"""Behavioral checks for the optional simulated-annealing challenger."""
import numpy as np
import pandas as pd
import pytest

from src.allocation_annealing import solve_annealing_policy
from src.policy import ACTIONS, assert_feasible, optimize_policy


def values(gains, costs=None):
    return pd.DataFrame([
        {"customer_id": f"c{i:02}", "action": action,
         "predicted_ncm": 20 + gain[j], "estimated_incremental_ncm": gain[j],
         "estimated_promo_cost": (int(j != 0) if costs is None else costs[i][j])}
        for i, gain in enumerate(gains) for j, action in enumerate(ACTIONS)
    ])


def test_escapes_known_greedy_budget_trap():
    v = values([[0, -1, -1, 10], [0, -1, -1, 6], [0, -1, -1, 6]],
               [[0, 1, 1, 10], [0, 1, 1, 5], [0, 1, 1, 5]])
    limits = dict(budget=10, max_offer_share=1)
    assert optimize_policy(v, **limits).estimated_incremental_ncm.sum() == 10
    result = solve_annealing_policy(v, **limits, iterations=1500, restarts=4)
    assert result.estimated_incremental_ncm.sum() == 12
    assert result.attrs["improvement_over_greedy"] == 2
    assert result.attrs["search_statistics"]["accepted_worsening"] > 0
    assert result.attrs["global_optimality_guaranteed"] is False


def test_negative_supporting_offer_can_enable_profitable_pct20():
    # Greedy excludes the -1 support offer and returns zero. Together the
    # negative support offer and one PCT_20 earn 9 and satisfy the 50% share.
    v = values([[0, -100, 10, -100], [0, -1, -100, -1]])
    limits = dict(budget=2, max_offer_share=1, max_pct20_share=.5)
    assert optimize_policy(v, **limits).estimated_incremental_ncm.sum() == 0
    result = solve_annealing_policy(v, **limits, iterations=1500, restarts=4)
    assert result.estimated_incremental_ncm.sum() == 9
    assert assert_feasible(result, **limits)


def test_all_constraints_hold_and_input_order_does_not_matter():
    rng = np.random.default_rng(9)
    gains = np.column_stack([np.zeros(12), rng.uniform(-3, 12, (12, 3))])
    costs = np.column_stack([np.zeros(12), rng.uniform(1, 5, (12, 3))])
    v = values(gains, costs)
    limits = dict(budget=12, max_offer_share=.5, max_pct20_share=.15,
                  holdout_ids=["c00", "c11"])
    first = solve_annealing_policy(v, **limits, iterations=500, restarts=4)
    second = solve_annealing_policy(v.sample(frac=1, random_state=8), **limits,
                                    iterations=500, restarts=4)
    pd.testing.assert_frame_equal(first, second)
    assert first.attrs == second.attrs
    assert assert_feasible(first, **limits, expected_customers=v.customer_id.unique())
    assert not first.recommended_action.eq("PCT_20").any()  # One of six exceeds 15%.
    assert first.estimated_incremental_ncm.sum() >= optimize_policy(v, **limits).estimated_incremental_ncm.sum()


def test_no_eligible_customers_are_handled_without_search():
    v = values([[0, 3, 9, 4]] * 3)
    result = solve_annealing_policy(v, holdout_ids=v.customer_id.unique(), iterations=10)
    assert result.recommended_action.eq("NO_OFFER").all()
    assert result.attrs["search_statistics"]["proposals"] == 0


@pytest.mark.parametrize("keyword,value", [("iterations", 0), ("iterations", 2.5),
                                           ("restarts", 0), ("restarts", True)])
def test_invalid_search_parameters_rejected(keyword, value):
    with pytest.raises(ValueError, match=keyword):
        solve_annealing_policy(values([[0, 1, 2, 3]]), **{keyword: value})
