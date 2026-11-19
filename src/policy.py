"""Shared, outcome-blind policy allocation for deployment and honest backtests.

All policies use the same population-level caps and preselected control IDs.
Dollar values are model projections, not an estimate from held-out outcomes.
"""
from __future__ import annotations

import math
from collections.abc import Iterable, Mapping

import numpy as np
import pandas as pd

ACTIONS = ["NO_OFFER", "PCT_10", "PCT_20", "FREE_SHIP"]
COMMERCIAL = ACTIONS[1:]
ALLOCATOR_NAME = "deterministic_incremental_ncm_greedy"


def _check_limits(budget, max_offer_share, max_pct20_share):
    if not np.isfinite(budget) or budget < 0:
        raise ValueError("budget must be finite and nonnegative")
    if not 0 <= max_offer_share <= 1 or not 0 <= max_pct20_share <= 1:
        raise ValueError("offer-share limits must be between zero and one")


def get_holdout(customer_ids: Iterable, share: float = 0.05, seed: int = 42) -> np.ndarray:
    """Randomize IDs independently of scores; stable under input reordering."""
    ids = pd.Series(list(customer_ids))
    if ids.isna().any() or not 0 <= share <= 1:
        raise ValueError("customer IDs must be non-null and holdout share in [0, 1]")
    ids = np.sort(ids.unique())
    n = math.ceil(share * len(ids))
    return np.sort(np.random.default_rng(seed).choice(ids, n, replace=False))


def _prepare(values: pd.DataFrame, holdout_ids=()):
    required = {"customer_id", "action", "predicted_ncm", "estimated_incremental_ncm", "estimated_promo_cost"}
    if not required.issubset(values):
        raise ValueError(f"missing value columns: {sorted(required - set(values))}")
    v = values.copy()
    if v.empty or v.customer_id.isna().any() or v.action.isna().any():
        raise ValueError("values must contain non-null customer IDs and actions")
    if v.duplicated(["customer_id", "action"]).any() or not v.action.isin(ACTIONS).all():
        raise ValueError("values must contain unique valid customer-action pairs")
    if not v.groupby("customer_id").action.nunique().eq(len(ACTIONS)).all():
        raise ValueError("every customer must have all four actions")
    numeric = v[["predicted_ncm", "estimated_incremental_ncm", "estimated_promo_cost"]].to_numpy(float)
    if not np.isfinite(numeric).all() or (v.estimated_promo_cost < 0).any():
        raise ValueError("values must be finite and promotion costs nonnegative")
    no = v.action.eq("NO_OFFER")
    if not np.allclose(v.loc[no, ["estimated_incremental_ncm", "estimated_promo_cost"]], 0, atol=1e-10):
        raise ValueError("NO_OFFER must have zero incremental value and promotion cost")
    hold = set(holdout_ids)
    if not hold.issubset(set(v.customer_id)):
        raise ValueError("holdout contains IDs outside the allocation population")
    v["_action_order"] = v.action.map({a: i for i, a in enumerate(ACTIONS)})
    v = v.sort_values(["customer_id", "_action_order"]).drop(columns="_action_order").reset_index(drop=True)
    return v, hold


def assert_feasible(policy, *, budget=62000, max_offer_share=0.6, max_pct20_share=0.15,
                    holdout_ids=(), expected_customers=None, atol=1e-6):
    """Fail on malformed allocations as well as financial/recipient violations."""
    _check_limits(budget, max_offer_share, max_pct20_share)
    if policy.empty or policy.customer_id.isna().any() or not policy.customer_id.is_unique:
        raise ValueError("policy must contain exactly one non-null row per customer")
    if expected_customers is not None and set(policy.customer_id) != set(expected_customers):
        raise ValueError("policy does not cover the complete population")
    if not policy.recommended_action.isin(ACTIONS).all():
        raise ValueError("policy contains invalid actions")
    cost = policy.estimated_promo_cost.to_numpy(float)
    if not np.isfinite(cost).all() or (cost < 0).any():
        raise ValueError("policy costs must be finite and nonnegative")
    n_off = int(policy.recommended_action.ne("NO_OFFER").sum())
    n20 = int(policy.recommended_action.eq("PCT_20").sum())
    hold = set(holdout_ids)
    if not hold.issubset(set(policy.customer_id)):
        raise ValueError("holdout IDs are missing from policy")
    if not policy.loc[policy.customer_id.isin(hold), "recommended_action"].eq("NO_OFFER").all():
        raise ValueError("reserved controls must receive NO_OFFER")
    if cost.sum() > budget + atol:
        raise ValueError("policy exceeds promotion budget")
    if n_off > math.floor(max_offer_share * len(policy)):
        raise ValueError("policy exceeds offer cap")
    if n20 > max_pct20_share * n_off + atol:
        raise ValueError("PCT_20 exceeds its share of commercial recipients")
    return True


def _selected_rows(v, actions, hold):
    chosen = pd.DataFrame({"customer_id": actions.index, "action": actions.to_numpy()})
    out = chosen.merge(v, on=["customer_id", "action"], how="left", validate="1:1")
    out = out.rename(columns={"action": "recommended_action"}).sort_values("customer_id").reset_index(drop=True)
    out["measurement_holdout"] = out.customer_id.isin(hold).astype(int)
    return out


def optimize_policy(values, *, budget=62000, max_offer_share=0.6, max_pct20_share=0.15,
                    holdout_ids=(), tie_tolerance=1e-7):
    """Return a deterministic feasible greedy allocation, not a global optimum.

    1. Rank positive PCT_10/FREE_SHIP options by incremental NCM, then lower
       cost and stable customer/action order; take affordable unassigned rows.
    2. Rank PCT_20 candidates by additional gain over their current action.
       Admit feasible upgrades, additions, or replacements of weaker recipients.
       Every move checks the PCT_20 share against the actual recipient count.
    3. Refill any remaining slots/budget with positive non-PCT_20 options.

    This bounded heuristic never uses an uplift/cost ratio, gives a nonpositive
    individual-gain offer, or sacrifices feasibility for a better objective.
    It can miss beneficial combinations when costs or quotas interact. In the
    current July case (one preferred offer and a slack budget), it reduces to
    selecting the largest positive gains. ``tie_tolerance`` remains accepted
    for caller compatibility; exact gain ordering is used instead of rounding.
    """
    _check_limits(budget, max_offer_share, max_pct20_share)
    if not np.isfinite(tie_tolerance) or tie_tolerance < 0:
        raise ValueError("tie_tolerance must be finite and nonnegative")
    v, hold = _prepare(values, holdout_ids)
    ids = v.customer_id.drop_duplicates().to_numpy()
    n = len(ids)
    gains = v.estimated_incremental_ncm.to_numpy(float).reshape(n, len(ACTIONS))
    costs = v.estimated_promo_cost.to_numpy(float).reshape(n, len(ACTIONS))
    reserved = np.array([cid in hold for cid in ids])
    cap = min(math.floor(max_offer_share * n), int((~reserved).sum()))
    chosen = np.zeros(n, dtype=int)
    spend, count, n20 = 0.0, 0, 0
    moves = dict(non_pct20_fill=0, pct20_upgrade=0, pct20_addition=0,
                 pct20_replacement=0, non_pct20_refill=0)
    non20 = sorted(((i, a) for i in range(n) if not reserved[i]
                    for a in (1, 3) if gains[i, a] > 0),
                   key=lambda x: (-gains[x], costs[x], x[0], x[1]))

    def fill_non20(phase):
        nonlocal spend, count
        for i, action in non20:
            if count >= cap:
                break
            if chosen[i] == 0 and spend + costs[i, action] <= budget + 1e-9:
                chosen[i] = action
                spend += costs[i, action]
                count += 1
                moves[phase] += 1

    fill_non20("non_pct20_fill")
    # Upgrade ranking is frozen at the beginning of this bounded pass. Each
    # candidate's gain and feasibility are recalculated before applying it.
    twenty = sorted((i for i in range(n) if not reserved[i] and gains[i, 2] > 0),
                    key=lambda i: (-(gains[i, 2] - gains[i, chosen[i]]),
                                   costs[i, 2] - costs[i, chosen[i]], i))
    for i in twenty:
        old = chosen[i]
        if old == 2 or gains[i, 2] <= gains[i, old]:
            continue
        if old != 0:
            new_spend = spend + costs[i, 2] - costs[i, old]
            if n20 + 1 <= max_pct20_share * count + 1e-12 and new_spend <= budget + 1e-9:
                chosen[i] = 2
                spend = new_spend
                n20 += 1
                moves["pct20_upgrade"] += 1
            continue
        if (count < cap and n20 + 1 <= max_pct20_share * (count + 1) + 1e-12
                and spend + costs[i, 2] <= budget + 1e-9):
            chosen[i] = 2
            spend += costs[i, 2]
            count += 1
            n20 += 1
            moves["pct20_addition"] += 1
            continue
        # A replacement keeps the offer count fixed. Removing an existing
        # PCT_20 also changes its numerator; removing another offer does not.
        existing = np.flatnonzero(chosen)
        if not len(existing):
            continue
        old_actions = chosen[existing]
        feasible = ((gains[existing, old_actions] < gains[i, 2])
                    & (spend - costs[existing, old_actions] + costs[i, 2] <= budget + 1e-9)
                    & (n20 + 1 - (old_actions == 2) <= max_pct20_share * count + 1e-12))
        victims = existing[feasible]
        if len(victims):
            victim = min(victims, key=lambda j: (gains[j, chosen[j]],
                                                -costs[j, chosen[j]], j))
            victim_action = chosen[victim]
            spend += costs[i, 2] - costs[victim, victim_action]
            n20 += 1 - int(victim_action == 2)
            chosen[victim], chosen[i] = 0, 2
            moves["pct20_replacement"] += 1
    fill_non20("non_pct20_refill")
    actions = pd.Series(np.asarray(ACTIONS)[chosen], index=ids)
    selected = _selected_rows(v, actions, hold)
    assert_feasible(selected, budget=budget, max_offer_share=max_offer_share,
                    max_pct20_share=max_pct20_share, holdout_ids=hold,
                    expected_customers=ids)
    objective = float(selected.estimated_incremental_ncm.sum())
    selected.attrs.update(solver_status="Feasible heuristic", allocator=ALLOCATOR_NAME,
                          solver="Deterministic incremental-NCM greedy allocation",
                          objective=objective, global_optimality_guaranteed=False,
                          allocation_moves=moves, tie_tolerance=tie_tolerance)
    return selected


def _mix_counts(n, mix, max_pct20_share):
    weights = pd.Series(mix, dtype=float).reindex(COMMERCIAL, fill_value=0)
    if not np.isfinite(weights).all() or (weights < 0).any() or weights.sum() <= 0:
        raise ValueError("historical mix must contain finite nonnegative commercial weights")
    weights /= weights.sum()
    raw = weights * n
    counts = np.floor(raw).astype(int)
    for a in (raw - counts).sort_values(ascending=False, kind="stable").index[:n - counts.sum()]:
        counts[a] += 1
    excess = max(0, int(counts["PCT_20"]) - math.floor(max_pct20_share * n))
    if excess:
        counts["PCT_20"] -= excess
        alternatives = weights.drop("PCT_20")
        if alternatives.sum() == 0:
            alternatives[:] = 1
        desired = alternatives / alternatives.sum() * excess
        extra = np.floor(desired).astype(int)
        for a in (desired - extra).sort_values(ascending=False, kind="stable").index[:excess - extra.sum()]:
            extra[a] += 1
        counts.loc[extra.index] += extra
    return counts


def build_baseline_policies(values, *, historical_mix: Mapping | None = None, budget=62000,
                            max_offer_share=0.6, max_pct20_share=0.15, holdout_ids=(), seed=42):
    """Return feasible, score-independent mix/free-ship and propensity baselines.

    Historical mix uses rounded commercial quotas (PCT_20 capped/reallocated),
    randomized customer order and the largest affordable prefix of that quota
    schedule. It approximates the historical mix; it does not optimize NCM.
    """
    _check_limits(budget, max_offer_share, max_pct20_share)
    v, hold = _prepare(values, holdout_ids)
    if "predicted_purchase_probability" not in v:
        raise ValueError("propensity baseline requires predicted_purchase_probability")
    base = v[v.action.eq("NO_OFFER")].set_index("customer_id")
    propensity = base.predicted_purchase_probability
    if not np.isfinite(propensity).all():
        raise ValueError("propensities must be finite")
    ids = base.index
    eligible = np.array([cid for cid in ids if cid not in hold])
    cap = min(math.floor(max_offer_share * len(ids)), len(eligible))
    cost = v.set_index(["customer_id", "action"]).estimated_promo_cost
    no = pd.Series("NO_OFFER", index=ids)
    p_prop = no.copy()
    spend = 0.0
    count = 0
    ranked = base.loc[eligible].sort_values("predicted_purchase_probability", ascending=False, kind="stable").index
    for cid in ranked:
        c = float(cost.loc[(cid, "PCT_10")])
        if count < cap and spend + c <= budget + 1e-9:
            p_prop.loc[cid] = "PCT_10"
            spend += c
            count += 1
    order = np.random.default_rng(seed).permutation(eligible)
    p_free = no.copy()
    spend = 0.0
    count = 0
    for cid in order:
        c = float(cost.loc[(cid, "FREE_SHIP")])
        if count < cap and spend + c <= budget + 1e-9:
            p_free.loc[cid] = "FREE_SHIP"
            spend += c
            count += 1
    mix = historical_mix if historical_mix is not None else {a: 1.0 for a in COMMERCIAL}
    p_mix = no.copy()
    # Recompute quotas when reducing recipients: dropping non-PCT_20 offers
    # without doing so can silently violate the denominator constraint.
    for count in range(cap, -1, -1):
        counts = _mix_counts(count, mix, max_pct20_share)
        actions = np.repeat(COMMERCIAL, counts.reindex(COMMERCIAL).to_numpy())
        keys = list(zip(order[:count], actions))
        spend = float(cost.reindex(pd.MultiIndex.from_tuples(keys, names=cost.index.names)).sum()) if keys else 0.0
        if spend <= budget + 1e-9:
            p_mix.loc[order[:count]] = actions
            break
    policies = {"ALL_NO_OFFER": no, "PROPENSITY_PCT10": p_prop,
                "HISTORICAL_MIX": p_mix, "RANDOM_FREE_SHIP": p_free}
    out = {}
    for name, actions in policies.items():
        selected = _selected_rows(v, actions, hold)
        assert_feasible(selected, budget=budget, max_offer_share=max_offer_share,
                        max_pct20_share=max_pct20_share, holdout_ids=hold, expected_customers=ids)
        selected.attrs["evaluation_basis"] = "model_projection"
        out[name] = selected
    return out


def allocate_selected_policy(values, policy_name, *, historical_mix=None, **constraints):
    """Apply a frozen allocation rule; simple baselines can beat optimization.

    Model-based rules require values from that rule's own forecast model. The
    caller must not substitute another model's scores after policy selection.
    """
    baseline_names = {"ALL_NO_OFFER", "PROPENSITY_PCT10", "HISTORICAL_MIX", "RANDOM_FREE_SHIP"}
    if policy_name in baseline_names:
        selected = build_baseline_policies(values, historical_mix=historical_mix, **constraints)[policy_name]
        selected.attrs.update(solver_status="Feasible baseline", objective=float(selected.estimated_incremental_ncm.sum()))
    else:
        from src.modeling import CANDIDATES
        from src.profit_models import PROFIT_CANDIDATES
        from src.policy_learning import POLICY_GRID, policy_name as segment_name
        known = set(CANDIDATES) | set(PROFIT_CANDIDATES) | {segment_name(c) for c in POLICY_GRID}
        if policy_name not in known:
            raise ValueError(f"Unknown allocation policy: {policy_name}")
        selected = optimize_policy(values, **constraints)
    selected.attrs["policy_name"] = policy_name
    return selected


def measurement_assignment(policy):
    """Export randomized experiment arms; NO_OFFER within POLICY stays POLICY."""
    required = {"customer_id", "recommended_action", "measurement_holdout"}
    if not required.issubset(policy) or not policy.customer_id.is_unique:
        raise ValueError("measurement design requires one policy row per customer and holdout flag")
    out = policy[["customer_id", "recommended_action", "measurement_holdout"]].copy()
    if not out.measurement_holdout.isin([0, 1]).all():
        raise ValueError("measurement_holdout must be binary")
    if not out.loc[out.measurement_holdout.eq(1), "recommended_action"].eq("NO_OFFER").all():
        raise ValueError("reserved controls must receive NO_OFFER")
    out["randomized_arm"] = np.where(out.measurement_holdout.eq(1), "CONTROL", "POLICY")
    return out.sort_values("customer_id").reset_index(drop=True)
