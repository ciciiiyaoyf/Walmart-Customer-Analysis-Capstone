"""Low-variance treatment effects for randomized offer targeting.

Campaign and segment main effects absorb baseline profit differences. Treatment
interactions are partially pooled toward the overall action effects with a ridge
penalty; subgroup features and penalties are selected on earlier campaigns only.
This estimates expected *net* contribution, so promotion costs must not be
subtracted again from the resulting action contrasts.
"""
import numpy as np
import pandas as pd

from src.modeling import ACTIONS


SEGMENTS = ("membership", "membership_tier", "membership_cart", "membership_history")
POLICY_GRID = [dict(segment=s, shrinkage=p) for s in SEGMENTS for p in (20., 100., 300.)]


def segment_keys(data, segment):
    if segment not in SEGMENTS:
        raise ValueError(f"Unknown segmentation: {segment}")
    member = data.is_member.fillna(-1).astype(str)
    if segment == "membership":
        return member
    if segment == "membership_tier":
        other = data.customer_tier.fillna("Missing").astype(str)
    elif segment == "membership_cart":
        other = pd.Series(np.where(data.cart_add_sessions_30d.fillna(0) > 0, "cart", "no_cart"), index=data.index)
    else:
        other = pd.Series(np.where(data.orders_90d.fillna(0) > 0, "recent_order", "no_recent_order"), index=data.index)
    return member + "|" + other


class SegmentEffectRegressor:
    """Regularized segment-by-treatment regression on randomized rows only.

    ``shrinkage`` is a ridge pseudo-count on treatment interactions; overall
    treatment effects are unpenalized. A centered group basis makes an unseen
    group fall back to the population-average action contrast. Campaign effects
    are nuisance terms used in fitting, with their training average used for
    prediction. No campaign identifier or treatment is required at prediction.
    """

    def __init__(self, segment="membership", shrinkage=100.):
        if segment not in SEGMENTS or not np.isfinite(shrinkage) or shrinkage <= 0:
            raise ValueError("A registered segment and positive finite shrinkage are required")
        self.segment, self.shrinkage = segment, float(shrinkage)

    @staticmethod
    def _centered(keys, levels, frequencies):
        index = {value: j for j, value in enumerate(levels)}
        z = np.zeros((len(keys), len(levels)))
        for i, key in enumerate(keys):
            if key in index:
                z[i] = -frequencies
                z[i, index[key]] += 1.
        return z

    def fit(self, data, target="ncm_14d"):
        if data.empty or not data.assignment_source.eq("exploration").all():
            raise ValueError("Treatment effects require nonempty randomized exploration rows only")
        if set(data.offer_type.unique()) != set(ACTIONS):
            raise ValueError("Every action needs randomized training support")
        y = data[target].to_numpy(float)
        if not np.isfinite(y).all() or data.campaign_id.isna().any():
            raise ValueError("Finite outcomes and observed campaign identifiers required")
        keys = segment_keys(data, self.segment)
        counts = keys.value_counts().sort_index()
        self.levels_ = counts.index.tolist()
        self.frequencies_ = counts.to_numpy(float) / len(data)
        z = self._centered(keys, self.levels_, self.frequencies_)
        campaigns = data.campaign_id.value_counts().sort_index()
        self.campaign_levels_ = campaigns.index.tolist()
        c = self._centered(data.campaign_id, self.campaign_levels_, campaigns.to_numpy(float) / len(data))
        t = np.column_stack([data.offer_type.eq(a).to_numpy(float) for a in ACTIONS[1:]])
        interactions = np.column_stack([z * t[:, j:j+1] for j in range(3)])
        design = np.column_stack([np.ones(len(data)), c, z, t, interactions])
        penalty = np.full(design.shape[1], 1e-8)
        penalty[0] = 0.
        penalty[-interactions.shape[1]:] = self.shrinkage
        self.coef_ = np.linalg.solve(design.T @ design + np.diag(penalty), design.T @ y)
        self.n_campaigns_ = len(campaigns)
        self.training_rows_ = len(data)
        self.target_ = target
        return self

    def predict_action(self, data, action):
        if action not in ACTIONS or not hasattr(self, "coef_"):
            raise ValueError("A fitted model and registered action are required")
        z = self._centered(segment_keys(data, self.segment), self.levels_, self.frequencies_)
        t = np.zeros((len(data), 3))
        if action != "NO_OFFER":
            t[:, ACTIONS.index(action)-1] = 1.
        interactions = np.column_stack([z * t[:, j:j+1] for j in range(3)])
        design = np.column_stack([np.ones(len(data)), np.zeros((len(data), self.n_campaigns_)), z, t, interactions])
        return design @ self.coef_

    def action_models(self):
        return {a: SegmentActionPredictor(self, a) for a in ACTIONS}


class SegmentActionPredictor:
    def __init__(self, model, action):
        self.model, self.action = model, action

    def predict(self, data):
        return self.model.predict_action(data, self.action)


def fit_segment_models(data, segment="membership", shrinkage=100., target="ncm_14d"):
    return SegmentEffectRegressor(segment, shrinkage).fit(data, target).action_models()


def policy_name(config):
    return f"SEGMENT_{config['segment'].upper()}_RIDGE_{config['shrinkage']:g}"


def stochastic_free_ship_scores(data, mu, costs, holdout_ids, budget, max_offer_share=.6):
    """DR components averaged over every feasible random recipient subset.

    Keeps the fixed measurement controls and averages over uniform random
    selection of exactly floor(max_offer_share*N) eligible recipients. This
    analytic calculation is exact only when *every* such subset is affordable;
    the largest possible predicted cost is checked before returning results.
    """
    mu, costs = np.asarray(mu, float), np.asarray(costs, float)
    if mu.shape != (len(data), len(ACTIONS)) or costs.shape != mu.shape:
        raise ValueError("Expected one potential-outcome and cost prediction per row/action")
    if not np.isfinite(mu).all() or not np.isfinite(costs).all() or (costs < 0).any():
        raise ValueError("Finite predictions and nonnegative costs required")
    if not 0 <= max_offer_share <= 1 or budget < 0:
        raise ValueError("Invalid allocation limits")
    eligible = ~data.customer_id.isin(holdout_ids).to_numpy()
    count = min(int(np.floor(max_offer_share*len(data))), int(eligible.sum()))
    worst_cost = np.sort(costs[eligible, ACTIONS.index("FREE_SHIP")])[-count:].sum() if count else 0.
    if worst_cost > budget + 1e-9:
        raise ValueError("Uniform recipient averaging requires a nonbinding budget for every subset")
    probability = eligible.astype(float) * (count / eligible.sum() if eligible.any() else 0.)
    observed = data.offer_type.to_numpy()
    observed_index = np.array([ACTIONS.index(a) for a in observed])
    residual = data.ncm_14d.to_numpy(float) - mu[np.arange(len(data)), observed_index]
    frame = data[["assignment_id", "customer_id", "campaign_id"]].copy()
    frame["observed_action"] = observed
    frame["policy"] = "RANDOM_FREE_SHIP_EXPECTATION"
    frame["offer_probability"] = probability
    frame["direct_component"] = probability * (mu[:, ACTIONS.index("FREE_SHIP")]-mu[:, 0])
    frame["residual_component"] = probability * (observed == "FREE_SHIP").astype(float) * residual
    frame["residual_component"] -= probability * (observed == "NO_OFFER").astype(float) * residual
    frame["worst_case_predicted_spend"] = worst_cost
    return frame
