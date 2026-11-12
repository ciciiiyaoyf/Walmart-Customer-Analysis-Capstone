import numpy as np
import pandas as pd
import pytest

from src.inference import clustered_difference_in_means


def test_repeated_observations_do_not_create_artificial_precision():
    original = pd.DataFrame({"customer_id": list("abcdefgh"), "offer_type": ["A"] * 4 + ["NO_OFFER"] * 4,
                              "y": [2., 4., 8., 10., 1., 3., 5., 7.]})
    base = clustered_difference_in_means(original, "y", "A")
    repeated = pd.concat([original, original], ignore_index=True)
    repeat = clustered_difference_in_means(repeated, "y", "A")
    assert repeat["estimate"] == base["estimate"]
    assert repeat["standard_error"] == pytest.approx(base["standard_error"])
    repeated.loc[len(original):, "customer_id"] += "_new"
    independent = clustered_difference_in_means(repeated, "y", "A")
    assert independent["standard_error"] < base["standard_error"]


def test_campaign_standardization_removes_campaign_mix_difference():
    # Every arm has zero effect within campaign, but campaign mixes differ.
    rows = []
    for campaign, level, nt, nc in [("low", 0., 9, 1), ("high", 100., 1, 9)]:
        rows += [{"customer_id": f"{campaign}-t{i}", "campaign_id": campaign, "offer_type": "A", "y": level}
                 for i in range(nt)]
        rows += [{"customer_id": f"{campaign}-c{i}", "campaign_id": campaign, "offer_type": "NO_OFFER", "y": level}
                 for i in range(nc)]
    data = pd.DataFrame(rows)
    pooled = clustered_difference_in_means(data, "y", "A")
    adjusted = clustered_difference_in_means(data, "y", "A", campaign_col="campaign_id")
    assert pooled["estimate"] == -80.
    assert adjusted["estimate"] == 0.
    assert adjusted["n_campaigns"] == 2


def test_campaign_with_no_control_fails_explicitly():
    data = pd.DataFrame({"customer_id": ["a", "b", "c"], "campaign_id": ["1", "1", "2"],
                          "offer_type": ["A", "NO_OFFER", "A"], "y": [1., 2., 3.]})
    with pytest.raises(ValueError, match="Both arms required in target campaign"):
        clustered_difference_in_means(data, "y", "A", campaign_col="campaign_id")


def test_interval_is_order_invariant_and_allows_cross_arm_customers():
    data = pd.DataFrame({"customer_id": list("aabbccdd"), "offer_type": ["A", "NO_OFFER"] * 4,
                          "y": [5., 2., 8., 4., 12., 6., 8., 3.]})
    result = clustered_difference_in_means(data, "y", "A")
    shuffled = clustered_difference_in_means(data.sample(frac=1, random_state=5), "y", "A")
    for key in ["estimate", "standard_error", "ci_low", "ci_high"]:
        assert shuffled[key] == pytest.approx(result[key])
    assert result["n_clusters"] == 4
    assert result["ci_low"] < result["estimate"] < result["ci_high"]
