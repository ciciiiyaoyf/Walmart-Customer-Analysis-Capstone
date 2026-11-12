import pandas as pd
import pytest

from src.measurement import estimate_policy_itt


def test_itt_includes_no_offer_customers_in_policy_arm():
    assignments = pd.DataFrame({"customer_id": list("abcde"),
                                "randomized_arm": ["POLICY"] * 3 + ["CONTROL"] * 2,
                                "recommended_action": ["NO_OFFER", "NO_OFFER", "PCT_10", "NO_OFFER", "NO_OFFER"]})
    outcomes = pd.DataFrame({"customer_id": list("abcde"), "ncm_14d": [0, 0, 100, 10, 10]})
    result = estimate_policy_itt(assignments, outcomes).iloc[0]
    assert result.n_policy == 3
    assert result.incremental_ncm_per_customer == pytest.approx(100 / 3 - 10)
    assert result.estimated_campaign_incremental_ncm == pytest.approx(70)
    assert result.incremental_ncm_per_randomized_customer == pytest.approx(14)


def test_missing_customer_outcomes_are_not_dropped_or_imputed_zero():
    assignments = pd.DataFrame({"customer_id": list("abcd"), "randomized_arm": ["POLICY"] * 2 + ["CONTROL"] * 2})
    outcomes = pd.DataFrame({"customer_id": list("abc"), "ncm_14d": [1, 2, 3]})
    with pytest.raises(ValueError, match="complete"):
        estimate_policy_itt(assignments, outcomes)
