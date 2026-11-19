"""Optional allocation challenger using feasible multi-start simulated annealing.

The deployment allocator is unchanged. This search compares fixed forecasts and
does not fit a model, inspect outcomes, or claim global optimality.
"""
from __future__ import annotations

from fractions import Fraction
import math

import numpy as np
import pandas as pd

from src.policy import (ACTIONS, _check_limits, _prepare, _selected_rows,
                        assert_feasible, optimize_policy)


def solve_annealing_policy(values, *, budget=62000, max_offer_share=.6,
                           max_pct20_share=.15, holdout_ids=(), seed=42,
                           iterations=30000, restarts=4):
    """Search feasible single-customer changes, paired changes, and exchanges.

    ``iterations`` is the number of proposals per restart. The starts cycle
    through greedy, all NO_OFFER, randomized feasible, and perturbed greedy.
    A geometric temperature schedule permits temporary objective decreases;
    the best feasible assignment found across all runs is returned. Greedy is
    always retained as an incumbent, so this challenger cannot return a lower
    objective than that incumbent. It may still miss the true optimum.

    Negative-gain individual offers are allowed: they can support a profitable
    PCT_20 offer under the share constraint. No infeasible move is accepted.
    """
    _check_limits(budget, max_offer_share, max_pct20_share)
    for name, value in (("iterations", iterations), ("restarts", restarts)):
        if isinstance(value, bool) or not isinstance(value, (int, np.integer)) or value < 1:
            raise ValueError(f"{name} must be a positive integer")
    v, hold = _prepare(values, holdout_ids)
    ids = v.customer_id.drop_duplicates().to_numpy()
    n = len(ids)
    gains = v.estimated_incremental_ncm.to_numpy(float).reshape(n, 4)
    costs = v.estimated_promo_cost.to_numpy(float).reshape(n, 4)
    rows = np.arange(n)
    eligible = np.array([i for i, cid in enumerate(ids) if cid not in hold], dtype=int)
    cap = math.floor(max_offer_share * n)
    fraction = Fraction(str(max_pct20_share))
    numerator, denominator = fraction.numerator, fraction.denominator
    constraints = dict(budget=budget, max_offer_share=max_offer_share,
                       max_pct20_share=max_pct20_share, holdout_ids=hold)
    greedy = optimize_policy(v, **constraints)
    action_codes = {action: i for i, action in enumerate(ACTIONS)}
    greedy_actions = greedy.recommended_action.map(action_codes).to_numpy(int)
    best = greedy_actions.copy()
    best_objective = float(gains[rows, best].sum())
    initial_objective = best_objective
    rng = np.random.default_rng(seed)
    # A within-customer action range reflects the loss needed to leave a local
    # choice; medians over individual gains can be dominated by tiny losses.
    magnitudes = np.ptp(gains[eligible], axis=1)
    magnitudes = magnitudes[magnitudes > 0]
    temperature_start = max(float(np.median(magnitudes)) if len(magnitudes) else 1., 1e-6)
    temperature_end = temperature_start * 1e-4
    cooling = (temperature_end / temperature_start) ** (1 / max(iterations - 1, 1))
    stats = dict(proposals=0, infeasible_proposals=0, unchanged_proposals=0,
                 feasible_proposals=0, accepted_moves=0, accepted_improving=0,
                 accepted_worsening=0, accepted_equal=0, rejected_by_temperature=0,
                 best_improvements=0, single_proposals=0, paired_proposals=0,
                 exchange_proposals=0)
    run_stats = []

    def totals(actions):
        return (float(costs[rows, actions].sum()), int(np.count_nonzero(actions)),
                int(np.count_nonzero(actions == 2)), float(gains[rows, actions].sum()))

    def feasible(spend, offered, twenty):
        return (spend <= budget + 1e-9 and offered <= cap
                and denominator * twenty <= numerator * offered)

    def random_start():
        actions = np.zeros(n, dtype=int)
        spend, offered, twenty = 0., 0, 0
        # Establish non-PCT_20 support first, then permit upgrades to PCT_20.
        for i in rng.permutation(eligible):
            action = int(rng.choice([0, 1, 3]))
            if action and feasible(spend + costs[i, action], offered + 1, twenty):
                actions[i] = action
                spend += costs[i, action]
                offered += 1
        for i in rng.permutation(eligible):
            old = actions[i]
            if rng.random() < .25:
                proposed_spend = spend + costs[i, 2] - costs[i, old]
                proposed_offered = offered + int(old == 0)
                if feasible(proposed_spend, proposed_offered, twenty + 1):
                    actions[i] = 2
                    spend, offered, twenty = proposed_spend, proposed_offered, twenty + 1
        return actions

    for run in range(restarts):
        kind = ["greedy", "all_no_offer", "random_feasible", "perturbed_greedy"][run % 4]
        if kind == "all_no_offer":
            chosen = np.zeros(n, dtype=int)
        elif kind == "random_feasible":
            chosen = random_start()
        else:
            chosen = greedy_actions.copy()
            if kind == "perturbed_greedy":
                # Remove PCT_20 before other offers so removal cannot break its
                # denominator constraint. This is a feasible perturbation.
                chosen[chosen == 2] = 0
                selected = np.flatnonzero(chosen)
                if len(selected):
                    chosen[rng.choice(selected, max(1, len(selected) // 5), replace=False)] = 0
        spend, offered, twenty, objective = totals(chosen)
        if not feasible(spend, offered, twenty):
            raise AssertionError("annealing initialization is infeasible")
        start_objective = objective
        if objective > best_objective + 1e-12:
            best, best_objective = chosen.copy(), objective
            stats["best_improvements"] += 1
        temperature = temperature_start
        run_accepted = 0
        for _ in range(iterations if len(eligible) else 0):
            stats["proposals"] += 1
            move = rng.random()
            i = int(eligible[rng.integers(len(eligible))])
            if move < .4 or len(eligible) < 2:
                indices = np.array([i])
                proposed = np.array([rng.integers(4)], dtype=int)
                stats["single_proposals"] += 1
            else:
                # Select a distinct second customer without rejection loops.
                first_position = int(np.searchsorted(eligible, i))
                second_position = int(rng.integers(len(eligible) - 1))
                second_position += int(second_position >= first_position)
                j = int(eligible[second_position])
                indices = np.array([i, j])
                if move < .7:
                    proposed = chosen[indices[::-1]].copy()
                    stats["exchange_proposals"] += 1
                else:
                    proposed = rng.integers(4, size=2)
                    stats["paired_proposals"] += 1
            old = chosen[indices]
            if np.array_equal(old, proposed):
                stats["unchanged_proposals"] += 1
                temperature *= cooling
                continue
            next_spend = math.fsum([spend, *costs[indices, proposed], *(-costs[indices, old])])
            next_offered = offered + int(np.count_nonzero(proposed)) - int(np.count_nonzero(old))
            next_twenty = twenty + int(np.count_nonzero(proposed == 2)) - int(np.count_nonzero(old == 2))
            if not feasible(next_spend, next_offered, next_twenty):
                stats["infeasible_proposals"] += 1
                temperature *= cooling
                continue
            stats["feasible_proposals"] += 1
            delta = float((gains[indices, proposed] - gains[indices, old]).sum())
            accept = delta >= 0 or rng.random() < math.exp(max(-745., delta / temperature))
            if accept:
                chosen[indices] = proposed
                spend, offered, twenty = next_spend, next_offered, next_twenty
                objective += delta
                stats["accepted_moves"] += 1
                run_accepted += 1
                stats["accepted_improving" if delta > 0 else "accepted_worsening" if delta < 0 else "accepted_equal"] += 1
                if objective > best_objective + 1e-12:
                    # Recompute before saving an improvement so accumulated
                    # floating-point deltas cannot manufacture a better score.
                    objective = float(gains[rows, chosen].sum())
                    if objective > best_objective + 1e-12:
                        best, best_objective = chosen.copy(), objective
                        stats["best_improvements"] += 1
            else:
                stats["rejected_by_temperature"] += 1
            temperature *= cooling
        final_spend, final_offered, final_twenty, final_objective = totals(chosen)
        if not feasible(final_spend, final_offered, final_twenty):
            raise AssertionError("annealing final state is infeasible")
        run_stats.append(dict(start=kind, initial_objective=start_objective,
                              final_objective=final_objective, accepted_moves=run_accepted))
    result = _selected_rows(v, pd.Series(np.asarray(ACTIONS)[best], index=ids), hold)
    assert_feasible(result, **constraints, expected_customers=ids)
    result.attrs.update(allocator="multi_start_simulated_annealing",
                        solver="Feasible multi-start simulated annealing",
                        solver_status="Feasible heuristic",
                        global_optimality_guaranteed=False,
                        objective=float(result.estimated_incremental_ncm.sum()),
                        initial_greedy_objective=initial_objective,
                        improvement_over_greedy=best_objective - initial_objective,
                        seed=int(seed), iterations_per_restart=int(iterations),
                        restarts=int(restarts), temperature_start=temperature_start,
                        temperature_end=temperature_end, search_statistics=stats,
                        runs=run_stats)
    return result
