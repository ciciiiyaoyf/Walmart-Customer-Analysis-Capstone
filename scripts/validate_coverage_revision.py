"""Record coverage evidence and compare numerical outputs with the prior revision."""
import io
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.features import FEATURES, OBSERVED_FEATURE_RENAMES
from src.diagnostics import verified_inactive_mask

REFERENCE = "54cf2ff6618e7bffab69f7d45f0a53f5f0d4aebc"


def previous(path):
    git = ["git", "-c", f"safe.directory={ROOT.as_posix()}", "-C", str(ROOT)]
    raw = io.BytesIO(subprocess.check_output(git + ["show", f"{REFERENCE}:{path}"]))
    return pd.read_parquet(raw) if path.endswith(".parquet") else pd.read_csv(raw)


def aligned(frame, key):
    if not frame[key].is_unique:
        raise ValueError(f"Duplicate {key}")
    return frame.set_index(key).sort_index()


def main():
    tables = ROOT / "outputs/tables"
    result = {"comparison_commit": REFERENCE, "feature_scope_renames": OBSERVED_FEATURE_RENAMES}
    feature_checks = []
    for cohort, key in [("historical", "assignment_id"), ("july", "customer_id")]:
        path = f"data/processed/{cohort}_features.parquet"
        before = aligned(previous(path).rename(columns=OBSERVED_FEATURE_RENAMES), key)
        after = aligned(pd.read_parquet(ROOT / path), key)
        pd.testing.assert_frame_equal(before[FEATURES], after[FEATURES], check_exact=True)
        coverage = pd.read_csv(tables / f"{cohort}_transaction_coverage.csv")
        assert coverage[key].is_unique and set(coverage[key]) == set(after.index)
        for days in [30, 90]:
            c = f"observed_orders_{days}d"
            np.testing.assert_array_equal(after[c], aligned(coverage, key)[c])
            assert coverage[f"orders_{days}d_coverage_status"].eq("unverified").all()
            assert coverage[f"complete_orders_{days}d"].isna().all()
        assert not verified_inactive_mask(coverage).any()
        feature_checks.append({"cohort": cohort, "rows": len(after), "predictor_count": len(FEATURES),
                               "all_predictor_values_identical_after_rename": True})
    result["feature_checks"] = feature_checks
    output_checks = []
    for path, key in [
        ("outputs/tables/logistic_test_predictions.csv", "assignment_id"),
        ("outputs/tables/xgboost_test_predictions.csv", "assignment_id"),
        ("outputs/tables/july_xgboost_ncm_scores.csv", "customer_id"),
        ("outputs/tables/july_action_ncm_scores.csv", "customer_id"),
        ("policy_submission.csv", "customer_id"),
    ]:
        before, after = aligned(previous(path), key), aligned(pd.read_csv(ROOT / path), key)
        pd.testing.assert_index_equal(before.index, after.index)
        pd.testing.assert_index_equal(before.columns, after.columns)
        numeric = after.select_dtypes(include="number").columns
        delta = (before[numeric] - after[numeric]).abs()
        same_numeric = np.isclose(before[numeric], after[numeric], rtol=1e-10, atol=1e-10, equal_nan=True)
        other = after.columns.difference(numeric)
        same_other = before[other].fillna("__missing__").eq(after[other].fillna("__missing__")).to_numpy()
        changed = ~(same_numeric.all(axis=1) & same_other.all(axis=1))
        output_checks.append({"file": path, "rows": len(after), "changed_rows": int(changed.sum()),
                              "max_absolute_numeric_change": float(delta.max().max()),
                              "comparison_tolerance": "rtol=1e-10, atol=1e-10"})
    result["output_checks"] = output_checks
    result["no_measured_performance_gain_from_coverage_revision"] = not any(r["changed_rows"] for r in output_checks)
    coverage = pd.read_csv(tables / "july_transaction_coverage.csv")
    reference = pd.read_csv(tables / "july_organic_reference_window.csv")
    july = pd.read_parquet(ROOT / "data/processed/july_features.parquet")
    assert reference.customer_id.is_unique and set(reference.customer_id) == set(july.customer_id)
    assert july.policy_as_of_ts.nunique() == 1
    for column in ["policy_as_of_ts", "window_30d_start", "window_30d_end",
                   "window_90d_start", "window_90d_end", "window_end_inclusive"]:
        assert coverage[column].nunique(dropna=False) == 1, column
    assert coverage.window_end_inclusive.eq(True).all()
    for column in ["reference_window_start", "reference_window_end", "reference_end_inclusive",
                   "reference_available", "reference_age_days"]:
        assert reference[column].nunique(dropna=False) == 1, column
    assert reference.reference_available.eq(True).all() and reference.reference_end_inclusive.eq(False).all()
    summary = {
        "july_customers": len(july),
        "policy_as_of_ts": str(july.policy_as_of_ts.iloc[0]),
        "window_30d_start": coverage.window_30d_start.iloc[0],
        "window_90d_start": coverage.window_90d_start.iloc[0],
        "window_end_inclusive": True,
        "source_latest_observed_organic_order_ts": coverage.source_latest_observed_organic_order_ts.max(),
        "july_observed_recent_90d_orders": int(coverage.observed_orders_90d.sum()),
        "july_verified_complete_90d_windows": int(coverage.orders_90d_coverage_status.eq("verified_complete").sum()),
        "july_complete_90d_counts_unknown": int(coverage.complete_orders_90d.isna().sum()),
        "july_verified_inactive_customers": int(verified_inactive_mask(coverage).sum()),
        "july_customers_with_any_observed_completed_order": int(july.observed_prior_order_count.gt(0).sum()),
        "reference_window_start": reference.reference_window_start.iloc[0],
        "reference_window_end_exclusive": reference.reference_window_end.iloc[0],
        "reference_customers_with_observed_completed_order": int(reference.observed_organic_reference_orders.gt(0).sum()),
        "reference_observed_completed_orders": int(reference.observed_organic_reference_orders.sum()),
        "reference_age_days_at_july_cutoff": float(reference.reference_age_days.iloc[0]),
        "interpretation": "Observed zeros are valid extract counts, not verified complete-history zeros. Last event timestamps do not certify feed coverage. The fixed historical reference is audit-only, not a recent-window replacement or model predictor.",
    }
    (tables / "transaction_coverage_summary.json").write_text(json.dumps(summary, indent=2)+"\n", encoding="utf-8")
    (tables / "coverage_revision_comparison.json").write_text(json.dumps(result, indent=2)+"\n", encoding="utf-8")
    print(json.dumps({"coverage": summary, "output_checks": output_checks}, indent=2))


if __name__ == "__main__":
    main()
