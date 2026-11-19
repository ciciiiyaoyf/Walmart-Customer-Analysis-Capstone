"""Optional CP-SAT allocation experiment; does not change deployment selection.

All forecasts and reported dollar totals retain their original precision. CP-SAT
uses integer coefficients internally. Cost rounding is conservative, so its
optimality certificate applies to that restricted feasible set only. An
independent cardinality-relaxation bound covers the original allocation problem.
"""
from __future__ import annotations

import math
from decimal import Decimal, ROUND_CEILING, ROUND_FLOOR, ROUND_HALF_EVEN, localcontext
from fractions import Fraction
from numbers import Integral

# Load native OR-Tools libraries before NumPy/pandas where possible. Some
# Windows Conda installations otherwise pick up an incompatible DLL already
# loaded by a numerical package. The experiment launcher uses the same order.
_ORTOOLS_IMPORT_ERROR = None
try:
    from ortools.sat.python import cp_model
except (ImportError, OSError) as exc:
    cp_model = None
    _ORTOOLS_IMPORT_ERROR = exc

import numpy as np
import pandas as pd

from src.policy import (ACTIONS, _check_limits, _prepare, _selected_rows,
                        assert_feasible, optimize_policy)


ALLOCATOR_NAME = "cp_sat_integer_scaled"
_SAFE_INTEGER_TOTAL = 2**62 - 1


def _integer(value, scale, rounding):
    # Decimal(str(...)) respects the stored decimal dollar amount, avoiding
    # binary-float artifacts such as ceil(0.07 * 100) incorrectly becoming 8.
    with localcontext() as ctx:
        ctx.prec = 400
        return int((Decimal(str(value)) * scale).to_integral_value(rounding=rounding))


def _check_total(coefficients, name):
    if sum(abs(int(x)) for x in coefficients) > _SAFE_INTEGER_TOTAL:
        raise ValueError(f"{name} exceeds safe CP-SAT integer range; reduce scale")


def solve_cpsat_policy(values, *, budget=62000, max_offer_share=0.6,
                       max_pct20_share=0.15, holdout_ids=(), time_limit=30,
                       scale=1_000_000, seed=42):
    """Maximize scaled incremental NCM with every original allocation rule.

    ``scale`` is integer units per dollar. Costs round up and the budget rounds
    down, so an integer-feasible assignment remains feasible in original
    dollars. Some originally feasible assignments can be excluded by rounding.
    Gains round to the nearest unit: any assignment's objective-rounding error
    is at most ``number_of_customers / (2 * scale)`` dollars.

    ``solver_status == 'OPTIMAL'`` certifies only the scaled problem. The separate
    ``original_problem_upper_bound`` is a valid upper bound obtained by dropping
    the budget and PCT_20-share constraints, retaining the offer cap and controls.
    If a feasible allocation reaches that bound, it also solves the original
    problem (up to the reported floating-point gap).

    Negative and zero individual-gain offers are retained: they may support a
    profitable PCT_20 assignment under its share-of-recipients requirement.
    """
    _check_limits(budget, max_offer_share, max_pct20_share)
    if isinstance(scale, bool) or not isinstance(scale, Integral) or scale <= 0:
        raise ValueError("scale must be a positive integer")
    scale = int(scale)
    if not np.isfinite(time_limit) or time_limit <= 0:
        raise ValueError("time_limit must be finite and positive")
    if isinstance(seed, bool) or not isinstance(seed, Integral) or not 0 <= seed < 2**31:
        raise ValueError("seed must be a nonnegative 32-bit integer")
    v, hold = _prepare(values, holdout_ids)
    ids = v.customer_id.drop_duplicates().to_numpy()
    n = len(ids)
    gains = v.estimated_incremental_ncm.to_numpy(float).reshape(n, len(ACTIONS))
    costs = v.estimated_promo_cost.to_numpy(float).reshape(n, len(ACTIONS))
    gain_units = [_integer(x, scale, ROUND_HALF_EVEN) for x in gains.flat]
    cost_units = [_integer(x, scale, ROUND_CEILING) for x in costs.flat]
    budget_units = _integer(budget, scale, ROUND_FLOOR)
    share = Fraction(str(max_pct20_share))
    _check_total(gain_units, "objective coefficients")
    _check_total(cost_units + [budget_units], "budget coefficients")
    _check_total([n * (share.denominator + 3 * share.numerator)], "share coefficients")
    cap = min(math.floor(max_offer_share * n), n - len(hold))

    if cp_model is None:
        raise ImportError(
            "CP-SAT experiments require a working optional ortools installation. "
            "On Windows Conda, import ortools.sat.python.cp_model before numpy/pandas "
            "to avoid native DLL conflicts."
        ) from _ORTOOLS_IMPORT_ERROR

    model = cp_model.CpModel()
    x = [[model.new_bool_var(f"x_{i}_{a}") for a in range(len(ACTIONS))]
         for i in range(n)]
    flat = [variable for row in x for variable in row]
    for i, cid in enumerate(ids):
        model.add_exactly_one(x[i])
        if cid in hold:
            model.add(x[i][0] == 1)
    offered = sum(x[i][a] for i in range(n) for a in (1, 2, 3))
    twenty = sum(x[i][2] for i in range(n))
    model.add(offered <= cap)
    model.add(share.denominator * twenty <= share.numerator * offered)
    model.add(sum(c * variable for c, variable in zip(cost_units, flat)) <= budget_units)
    model.maximize(sum(g * variable for g, variable in zip(gain_units, flat)))

    # Hints affect search only. Never constrain CP-SAT to greedy's action set.
    hint = optimize_policy(v, budget=budget, max_offer_share=max_offer_share,
                           max_pct20_share=max_pct20_share, holdout_ids=hold)
    hint_actions = hint.recommended_action.map({a: j for j, a in enumerate(ACTIONS)}).to_numpy()
    if sum(cost_units[4 * i + a] for i, a in enumerate(hint_actions)) <= budget_units:
        for i, action in enumerate(hint_actions):
            for a in range(len(ACTIONS)):
                model.add_hint(x[i][a], int(a == action))

    solver = cp_model.CpSolver()
    solver.parameters.max_time_in_seconds = float(time_limit)
    solver.parameters.num_search_workers = 1
    solver.parameters.random_seed = int(seed)
    status = solver.solve(model)
    status_name = solver.status_name(status)
    if status not in (cp_model.OPTIMAL, cp_model.FEASIBLE):
        raise RuntimeError(f"CP-SAT returned {status_name}; no verified feasible allocation")
    chosen = [next(a for a in range(4) if solver.value(x[i][a])) for i in range(n)]
    selected = _selected_rows(v, pd.Series(np.asarray(ACTIONS)[chosen], index=ids), hold)
    assert_feasible(selected, budget=budget, max_offer_share=max_offer_share,
                    max_pct20_share=max_pct20_share, holdout_ids=hold,
                    expected_customers=ids)

    eligible = np.array([cid not in hold for cid in ids])
    best_extra = np.maximum(0, gains[eligible, 1:].max(axis=1) - gains[eligible, 0])
    original_bound = float(math.fsum(gains[:, 0])
                           + math.fsum(sorted(best_extra, reverse=True)[:cap]))
    objective = float(math.fsum(selected.estimated_incremental_ncm))
    rounding_error = n / (2 * scale)
    scaled_bound = float(solver.best_objective_bound)
    # best_objective_bound is a double; account conservatively for one ULP
    # when translating it back to dollars.
    subset_bound = float(np.nextafter(scaled_bound, np.inf) / scale + rounding_error)
    selected.attrs.update(
        allocator=ALLOCATOR_NAME, solver="OR-Tools CP-SAT", solver_status=status_name,
        objective=objective, scale=scale, seed=int(seed), time_limit_seconds=float(time_limit),
        solver_wall_time_seconds=float(solver.wall_time),
        scaled_objective=float(solver.objective_value), scaled_best_bound=scaled_bound,
        scaled_problem_optimal=status == cp_model.OPTIMAL,
        objective_rounding_error_dollars=rounding_error,
        conservative_subset_objective_upper_bound=subset_bound,
        conservative_subset_gap_dollars=max(0.0, subset_bound - objective),
        original_problem_upper_bound=original_bound,
        original_problem_gap_dollars=max(0.0, original_bound - objective),
        original_problem_bound_met=math.isclose(original_bound, objective, rel_tol=0, abs_tol=1e-8),
        global_optimality_guaranteed=False,
        optimality_scope="CP-SAT certificate covers integer-scaled conservative feasible set; "
                         "original_problem_upper_bound is a separate cardinality-relaxation bound",
        cost_rounding="ceil each cost; floor budget; may exclude originally feasible combinations",
        budget_rounding_max_extra_dollars=(n + 1) / scale,
        retained_nonpositive_action_options=True,
    )
    return selected
