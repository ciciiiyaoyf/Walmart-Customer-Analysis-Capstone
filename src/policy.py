"""Shared, outcome-blind policy allocation for deployment and honest backtests.

All policies use the same population-level caps and preselected control IDs.
Dollar values are model projections, not an estimate from held-out outcomes.
"""
from __future__ import annotations

import math
from collections.abc import Iterable, Mapping

import numpy as np
import pandas as pd
from scipy.optimize import Bounds, LinearConstraint, milp
from scipy.sparse import coo_matrix, csr_matrix, vstack

ACTIONS = ["NO_OFFER", "PCT_10", "PCT_20", "FREE_SHIP"]
COMMERCIAL = ACTIONS[1:]


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
    """Maximize projected incremental NCM with scipy/HiGHS binary optimization.

    First optimize dollars. A second solve, restricted to within tie_tolerance
    dollars of that optimum, prefers fewer offers, then lower cost and stable
    sorted variable order. Sorting makes input row order irrelevant. The primary
    objective is never replaced by an arbitrary uplift/cost ratio.
    """
    _check_limits(budget, max_offer_share, max_pct20_share)
    if not np.isfinite(tie_tolerance) or tie_tolerance < 0:
        raise ValueError("tie_tolerance must be finite and nonnegative")
    v, hold = _prepare(values, holdout_ids)
    ids = v.customer_id.drop_duplicates().to_numpy()
    n, m = len(ids), len(v)
    commercial = v.action.ne("NO_OFFER").to_numpy(float)
    pct20 = v.action.eq("PCT_20").to_numpy(float)
    costs = v.estimated_promo_cost.to_numpy(float)
    gains = v.estimated_incremental_ncm.to_numpy(float)
    one_per_customer = coo_matrix((np.ones(m), (np.repeat(np.arange(n), len(ACTIONS)), np.arange(m))), shape=(n, m)).tocsr()
    matrix = vstack([one_per_customer, csr_matrix(costs[None, :]), csr_matrix(commercial[None, :]),
                     csr_matrix((pct20 - max_pct20_share * commercial)[None, :])], format="csc")
    lo = np.r_[np.ones(n), -np.inf, -np.inf, -np.inf]
    hi = np.r_[np.ones(n), budget, math.floor(max_offer_share * n), 0.0]
    upper = np.ones(m)
    upper[v.customer_id.isin(hold).to_numpy() & (commercial > 0)] = 0.0
    bounds = Bounds(np.zeros(m), upper)
    options = {"mip_rel_gap": 0.0, "presolve": True}
    first = milp(-gains, integrality=np.ones(m), bounds=bounds,
                 constraints=LinearConstraint(matrix, lo, hi), options=options)
    if first.status != 0 or first.x is None:
        raise RuntimeError(f"policy optimization failed: {first.message}")
    optimum = float(gains @ np.rint(first.x))
    secondary = commercial * (n + 1) + costs / max(1.0, costs.max())
    secondary += (np.arange(m) + 1) / (m + 1) ** 2
    matrix2 = vstack([matrix, csr_matrix(gains[None, :])], format="csc")
    second = milp(secondary, integrality=np.ones(m), bounds=bounds,
                  constraints=LinearConstraint(matrix2, np.r_[lo, optimum - tie_tolerance], np.r_[hi, np.inf]),
                  options=options)
    if second.status != 0 or second.x is None:
        raise RuntimeError(f"policy tie-breaking failed: {second.message}")
    chosen = v.loc[second.x > 0.5].copy().rename(columns={"action": "recommended_action"})
    chosen["measurement_holdout"] = chosen.customer_id.isin(hold).astype(int)
    chosen = chosen.reset_index(drop=True)
    actual = float(chosen.estimated_incremental_ncm.sum())
    if actual < optimum - max(tie_tolerance * 2, 1e-6):
        raise RuntimeError("tie-breaking degraded the primary objective")
    assert_feasible(chosen, budget=budget, max_offer_share=max_offer_share,
                    max_pct20_share=max_pct20_share, holdout_ids=hold, expected_customers=ids)
    chosen.attrs.update(solver_status="Optimal", solver="scipy.optimize.milp / HiGHS",
                        objective=actual, primary_objective=optimum, tie_tolerance=tie_tolerance)
    return chosen


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
