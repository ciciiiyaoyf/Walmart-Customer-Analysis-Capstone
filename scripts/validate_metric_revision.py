"""Verify the metrics revision against the preceding coverage-correction commit."""
import io
import json
import subprocess
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
REFERENCE = "2cec21b9b6200d1b4da1c0728da44cfffe63f784"


def main():
    git = ["git", "-c", f"safe.directory={ROOT.as_posix()}", "-C", str(ROOT)]
    checks = []
    for name, key in [
        ("outputs/tables/logistic_test_predictions.csv", "assignment_id"),
        ("outputs/tables/xgboost_test_predictions.csv", "assignment_id"),
        ("outputs/tables/july_xgboost_ncm_scores.csv", "customer_id"),
        ("outputs/tables/july_action_ncm_scores.csv", "customer_id"),
        ("policy_submission.csv", "customer_id"),
    ]:
        before = pd.read_csv(io.BytesIO(subprocess.check_output(git + ["show", f"{REFERENCE}:{name}"])))
        after = pd.read_csv(ROOT / name)
        assert before[key].is_unique and after[key].is_unique
        before, after = before.set_index(key).sort_index(), after.set_index(key).sort_index()
        pd.testing.assert_index_equal(before.index, after.index)
        pd.testing.assert_index_equal(before.columns, after.columns)
        numeric = after.select_dtypes(include="number").columns
        other = after.columns.difference(numeric)
        numeric_equal = np.isclose(before[numeric], after[numeric], rtol=1e-10, atol=1e-10, equal_nan=True)
        other_equal = before[other].fillna("__missing__").eq(after[other].fillna("__missing__")).to_numpy()
        changed = ~(numeric_equal.all(axis=1) & other_equal.all(axis=1))
        checks.append({"file": name, "rows": len(after), "changed_rows": int(changed.sum()),
                       "max_absolute_numeric_change": float((before[numeric]-after[numeric]).abs().max().max())})
    result = {"comparison_commit": REFERENCE, "output_checks": checks,
              "predictions_and_submission_unchanged": not any(row["changed_rows"] for row in checks),
              "interpretation": "Additional diagnostics measure existing predictions. This comparison distinguishes a reporting change from a model-performance improvement."}
    path = ROOT / "outputs/tables/metric_revision_comparison.json"
    path.write_text(json.dumps(result, indent=2)+"\n", encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
