"""Analysis of the pre-randomized July policy-versus-NO_OFFER experiment."""
import numpy as np
import pandas as pd
from scipy.stats import t


def estimate_policy_itt(assignments, outcomes, outcome="ncm_14d", confidence=0.95):
    """Difference in mean NCM for every POLICY customer versus every CONTROL.

    Outcomes must be finalized for the complete randomized population. Missing
    outcomes are not silently treated as zero or dropped. In-window nonbuyers
    should have explicit zero outcomes from the same audited outcome builder.
    The Welch interval describes the randomized arm contrast; it does not test
    offer recipients against controls or identify each offer's causal effect.
    """
    if not 0 < confidence < 1:
        raise ValueError("confidence must be between zero and one")
    if not {"customer_id", "randomized_arm"}.issubset(assignments):
        raise ValueError("assignments require customer_id and randomized_arm")
    if not {"customer_id", outcome}.issubset(outcomes):
        raise ValueError("outcomes require customer_id and the outcome column")
    if not assignments.customer_id.is_unique or not outcomes.customer_id.is_unique:
        raise ValueError("one assignment and one outcome row per customer are required")
    if assignments.customer_id.isna().any() or outcomes.customer_id.isna().any():
        raise ValueError("customer IDs cannot be missing")
    if set(assignments.customer_id) != set(outcomes.customer_id):
        raise ValueError("complete randomized-population outcomes are required")
    if not assignments.randomized_arm.isin(["POLICY", "CONTROL"]).all():
        raise ValueError("unknown randomized arm")
    joined = assignments[["customer_id", "randomized_arm"]].merge(
        outcomes[["customer_id", outcome]], on="customer_id", validate="1:1")
    if not np.isfinite(joined[outcome].to_numpy(float)).all():
        raise ValueError("outcomes must be finite and fully observed")
    y1 = joined.loc[joined.randomized_arm.eq("POLICY"), outcome]
    y0 = joined.loc[joined.randomized_arm.eq("CONTROL"), outcome]
    if min(len(y1), len(y0)) < 2:
        raise ValueError("at least two observed customers in each arm are required")
    v1, v0 = y1.var(ddof=1) / len(y1), y0.var(ddof=1) / len(y0)
    se = float(np.sqrt(v1 + v0))
    dof = (v1 + v0) ** 2 / (v1 ** 2 / (len(y1) - 1) + v0 ** 2 / (len(y0) - 1)) if se else np.inf
    delta = float(y1.mean() - y0.mean())
    radius = float(t.ppf((1 + confidence) / 2, dof) * se) if se else 0.0
    policy_share = len(y1) / len(joined)
    return pd.DataFrame([{"estimand": "policy_intention_to_treat", "outcome": outcome,
                          "n_policy": len(y1), "n_control": len(y0), "policy_mean": y1.mean(),
                          "control_mean": y0.mean(), "incremental_ncm_per_customer": delta,
                          "effect_denominator": "customer assigned to POLICY arm",
                          "policy_assignment_share": policy_share,
                          "incremental_ncm_per_randomized_customer": policy_share * delta,
                          "estimated_campaign_incremental_ncm": len(y1) * delta,
                          "standard_error": se, "confidence_level": confidence,
                          "ci_low": delta - radius, "ci_high": delta + radius}])
