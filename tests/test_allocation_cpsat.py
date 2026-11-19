"""Independent small-instance correctness and rounding checks for CP-SAT."""
import itertools

import pytest

pytest.importorskip("ortools.sat.python.cp_model")

import numpy as np
import pandas as pd

from src.allocation_cpsat import solve_cpsat_policy
from src.policy import ACTIONS, assert_feasible, optimize_policy


def values(gains, costs=None):
    return pd.DataFrame([
        dict(customer_id=f"c{i}", action=action, predicted_ncm=20 + gain,
             estimated_incremental_ncm=gain,
             estimated_promo_cost=(0 if a == 0 else 1) if costs is None else costs[i][a])
        for i, row in enumerate(gains) for a, (action, gain) in enumerate(zip(ACTIONS, row))
    ])


def exhaustive(v, budget, offer_share, twenty_share, hold=()):
    n = v.customer_id.nunique()
    gains = v.estimated_incremental_ncm.to_numpy().reshape(n, 4)
    costs = v.estimated_promo_cost.to_numpy().reshape(n, 4)
    best = -np.inf
    for assignment in itertools.product(range(4), repeat=n):
        if any(assignment[i] for i in range(n) if f"c{i}" in hold):
            continue
        offered = sum(a != 0 for a in assignment)
        if offered > np.floor(offer_share * n) or assignment.count(2) > twenty_share * offered:
            continue
        if sum(costs[i, a] for i, a in enumerate(assignment)) <= budget:
            best = max(best, sum(gains[i, a] for i, a in enumerate(assignment)))
    return best


def test_cpsat_beats_known_greedy_counterexample():
    v = values([[0, -1, -1, 10], [0, -1, -1, 6], [0, -1, -1, 6]],
               [[0, 1, 1, 10], [0, 1, 1, 5], [0, 1, 1, 5]])
    p = solve_cpsat_policy(v, budget=10, max_offer_share=1)
    assert optimize_policy(v, budget=10, max_offer_share=1).estimated_incremental_ncm.sum() == 10
    assert p.estimated_incremental_ncm.sum() == exhaustive(v, 10, 1, .15) == 12
    assert p.attrs["scaled_problem_optimal"]


def test_negative_gain_supporting_offer_can_enable_profitable_twenty():
    # One PCT_20 requires a second commercial recipient at a 50% share cap.
    # The support offer loses $1 but unlocks $10, which greedy misses.
    v = values([[0, -20, 10, -20], [0, -1, -20, -2]])
    p = solve_cpsat_policy(v, budget=2, max_offer_share=1, max_pct20_share=.5)
    assert p.estimated_incremental_ncm.sum() == exhaustive(v, 2, 1, .5) == 9
    assert p.recommended_action.tolist() == ["PCT_20", "PCT_10"]


def test_twenty_denominator_uses_actual_recipients():
    v = values([[0, 1, 100, 1]] * 6)
    p = solve_cpsat_policy(v, budget=10, max_offer_share=1)
    assert p.recommended_action.eq("PCT_20").sum() == 0
    assert p.estimated_incremental_ncm.sum() == 6


def test_controls_and_shuffled_inputs():
    v = values([[0, 5, 8, 2], [0, 1, 4, 8], [0, 9, 1, 3], [0, 3, 7, 4]])
    kwargs = dict(budget=3, max_offer_share=.75, max_pct20_share=.5, holdout_ids=["c2"])
    p = solve_cpsat_policy(v, **kwargs)
    q = solve_cpsat_policy(v.sample(frac=1, random_state=17), **kwargs)
    pd.testing.assert_frame_equal(p, q)
    assert p.estimated_incremental_ncm.sum() == exhaustive(v, 3, .75, .5, ["c2"])
    assert assert_feasible(p, **kwargs)
    assert p.set_index("customer_id").loc["c2", "recommended_action"] == "NO_OFFER"


def test_conservative_cost_scaling_never_creates_budget_violation():
    v = values([[0, -1, -1, 3], [0, -1, -1, 3]],
               [[0, 1, 1, .0049], [0, 1, 1, .0049]])
    p = solve_cpsat_policy(v, budget=.0098, max_offer_share=1, scale=100)
    # The true optimum is 6; conservative cent scaling excludes both offers.
    assert exhaustive(v, .0098, 1, .15) == 6
    assert p.estimated_incremental_ncm.sum() == 0
    assert p.attrs["scaled_problem_optimal"]
    assert not p.attrs["original_problem_bound_met"]
    assert p.attrs["original_problem_upper_bound"] == 6
    assert p.attrs["conservative_subset_objective_upper_bound"] < 6
    assert assert_feasible(p, budget=.0098, max_offer_share=1)


def test_decimal_exact_boundary_is_not_rounded_up_again():
    p = solve_cpsat_policy(values([[0, -1, -1, 2]], [[0, 1, 1, .07]]),
                           budget=.07, max_offer_share=1, scale=100)
    assert p.recommended_action.tolist() == ["FREE_SHIP"]


def test_original_precision_and_valid_independent_bound_are_reported():
    v = values([[0, -1, -1, 1.23456789], [0, -1, -1, .45678912]])
    p = solve_cpsat_policy(v, budget=2, max_offer_share=.5, scale=100)
    assert p.estimated_incremental_ncm.sum() == pytest.approx(1.23456789, abs=1e-12)
    assert p.attrs["original_problem_bound_met"]
    assert p.attrs["original_problem_gap_dollars"] == pytest.approx(0)
    assert p.attrs["objective_rounding_error_dollars"] == .01


@pytest.mark.parametrize("kwargs", [dict(scale=0), dict(scale=1.5), dict(scale=True),
                                    dict(time_limit=0), dict(time_limit=float("nan")),
                                    dict(seed=-1), dict(scale=10**30)])
def test_invalid_solver_parameters_fail_before_solving(kwargs):
    with pytest.raises(ValueError):
        solve_cpsat_policy(values([[0, 1, 2, 3]]), **kwargs)


@pytest.mark.parametrize("seed", range(6))
def test_small_random_instances_match_exhaustive_search(seed):
    rng = np.random.default_rng(seed)
    gains = np.column_stack([np.zeros(4), rng.integers(-3, 10, (4, 3))])
    costs = np.column_stack([np.zeros(4), rng.integers(0, 5, (4, 3))])
    v = values(gains, costs)
    p = solve_cpsat_policy(v, budget=5, max_offer_share=.75, max_pct20_share=.5)
    assert p.estimated_incremental_ncm.sum() == exhaustive(v, 5, .75, .5)
