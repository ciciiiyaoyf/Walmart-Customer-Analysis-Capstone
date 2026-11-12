"""Descriptive randomized-arm comparisons with customer-clustered uncertainty.

Campaign-standardized results condition on the observed campaign distribution.
They do not estimate uncertainty from deploying in an entirely new campaign.
Subgroup intervals are exploratory and are not multiplicity-adjusted.
"""
import numpy as np
import pandas as pd
from scipy.stats import t as student_t


def clustered_difference_in_means(data, outcome, treatment, control="NO_OFFER",
                                 action_col="offer_type", cluster_col="customer_id",
                                 campaign_col=None, confidence=0.95):
    """Estimate an arm contrast with a customer-cluster sandwich interval.

    With ``campaign_col``, standardize within-campaign contrasts to that column's
    distribution in ``data`` (including its other randomized arms). Campaign
    weights are fixed at their observed values. Both arms must be represented
    in every target campaign; unsupported comparisons fail explicitly.
    """
    if treatment == control or not 0 < confidence < 1:
        raise ValueError("Distinct treatment/control and 0 < confidence < 1 are required")
    cols = [outcome, action_col, cluster_col] + ([campaign_col] if campaign_col else [])
    if data[cols].isna().any().any():
        raise ValueError("Inference columns must not contain missing values")
    d = data.loc[data[action_col].isin([treatment, control]), cols].copy().reset_index(drop=True)
    if not np.isfinite(d[outcome].to_numpy(float)).all():
        raise ValueError("Outcomes must be finite")
    n_treat = int(d[action_col].eq(treatment).sum())
    n_control = int(d[action_col].eq(control).sum())
    n_clusters = d[cluster_col].nunique()
    if not n_treat or not n_control or n_clusters < 2:
        raise ValueError("Both arms and at least two customer clusters are required")
    # Each influence term already includes the arm-mean denominator. Summing
    # within customer retains covariance between all of that customer's rows.
    influence = pd.Series(0., index=d.index)
    estimate = 0.
    if campaign_col is None:
        strata = [(None, 1., d)]
    else:
        weights = data[campaign_col].value_counts(normalize=True, sort=False)
        strata = [(campaign, float(weight), d[d[campaign_col] == campaign])
                  for campaign, weight in weights.items()]
    for campaign, weight, part in strata:
        tr = part[part[action_col] == treatment]
        co = part[part[action_col] == control]
        if tr.empty or co.empty:
            raise ValueError(f"Both arms required in target campaign {campaign!r}")
        mu_t, mu_c = tr[outcome].mean(), co[outcome].mean()
        estimate += weight * (mu_t - mu_c)
        influence.loc[tr.index] = weight * (tr[outcome] - mu_t) / len(tr)
        influence.loc[co.index] = -weight * (co[outcome] - mu_c) / len(co)
    cluster_scores = influence.groupby(d[cluster_col]).sum()
    variance = n_clusters / (n_clusters - 1) * np.square(cluster_scores).sum()
    se = float(np.sqrt(variance))
    critical = student_t.ppf((1 + confidence) / 2, df=n_clusters - 1)
    return {
        "estimate": float(estimate), "standard_error": se,
        "ci_low": float(estimate - critical * se), "ci_high": float(estimate + critical * se),
        "n_treat": n_treat, "n_control": n_control, "n_clusters": int(n_clusters),
        "n_campaigns": len(strata) if campaign_col else None,
        "inference_method": "customer_cluster_sandwich_t",
        "campaign_adjustment": "fixed_distribution_standardization" if campaign_col else "none",
        "confidence": confidence,
    }
