"""Compare the new allocator with frozen previous outputs on identical inputs."""
from pathlib import Path
import hashlib
import json
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.policy import ACTIONS, assert_feasible, get_holdout, optimize_policy


def main():
    root = Path(__file__).resolve().parents[1]
    tables = root / "outputs/tables"
    reference = root / "outputs/reference/allocator_before_20261118"
    provenance = json.loads((reference / "provenance.json").read_text())
    for relative, expected in provenance["unchanged_input_hashes"].items():
        actual = hashlib.sha256((root / relative).read_bytes()).hexdigest()
        if actual != expected:
            raise AssertionError(f"Allocation-only revision changed a saved input/model: {relative}")
    values = pd.read_csv(tables / "july_customer_action_values.csv")
    new = pd.read_csv(tables / "selected_policy_detailed.csv").set_index("customer_id").sort_index()
    previous_milp = pd.read_csv(reference / "milp_policy_detailed.csv").set_index("customer_id").sort_index()
    previous_submission = pd.read_csv(reference / "policy_submission.csv").set_index("customer_id").sort_index()
    old_controls = pd.read_csv(reference / "measurement_holdout.csv").customer_id.to_numpy()
    controls = get_holdout(values.customer_id.unique())
    np.testing.assert_array_equal(controls, old_controls)
    pd.testing.assert_index_equal(new.index, previous_milp.index)
    assert assert_feasible(new.reset_index(), holdout_ids=controls, expected_customers=values.customer_id.unique())
    rerun = optimize_policy(values, holdout_ids=controls).set_index("customer_id").sort_index()
    pd.testing.assert_series_equal(new.recommended_action, rerun.recommended_action)
    expected_columns = ["estimated_incremental_ncm", "estimated_promo_cost", "predicted_ncm"]
    np.testing.assert_allclose(new[expected_columns], rerun[expected_columns], rtol=0, atol=1e-10)
    candidates = values.set_index(["customer_id", "action"])

    def summarize(frame):
        rows = candidates.loc[list(zip(frame.index, frame.recommended_action))]
        return {"incremental_ncm": float(rows.estimated_incremental_ncm.sum()),
                "total_ncm": float(rows.predicted_ncm.sum()),
                "promo_cost": float(rows.estimated_promo_cost.sum()),
                "action_counts": {a: int(frame.recommended_action.eq(a).sum()) for a in ACTIONS}}

    after, old_milp, old_submission = map(summarize, [new, previous_milp, previous_submission])
    previous_historical = pd.read_csv(reference / "heldout_policy_comparison.csv")
    current_historical = pd.read_csv(tables / "heldout_policy_comparison.csv")
    selection = json.loads((root / "outputs/models/policy_selection.json").read_text())
    selected = selection["selected_policy"]
    metric_columns = ["policy", "dr_incremental_ncm_per_customer", "ci_low", "ci_high"]
    result = {
        "date": "2026-11-18", "comparison_commit": provenance["comparison_commit"],
        "allocator": selection["allocator"], "selected_policy": selected,
        "deployment_selection_basis": selection["deployment_selection_basis"],
        "validation_winner": selection["validation_selected_policy"],
        "unchanged_forecast_model_and_input_hash_count": len(provenance["unchanged_input_hashes"]),
        "controls_unchanged": True, "control_count": len(controls), "all_constraints_pass": True,
        "new_greedy": after, "previous_milp": old_milp, "previous_submission": old_submission,
        "gain_vs_previous_milp": after["incremental_ncm"] - old_milp["incremental_ncm"],
        "gain_vs_previous_submission": after["incremental_ncm"] - old_submission["incremental_ncm"],
        "action_changes_vs_previous_milp": int(new.recommended_action.ne(previous_milp.recommended_action).sum()),
        "action_changes_vs_previous_submission": int(new.recommended_action.ne(previous_submission.recommended_action).sum()),
        "historical_selected_policy_before": previous_historical.loc[previous_historical.policy.eq(selected), metric_columns].to_dict("records"),
        "historical_selected_policy_after": current_historical.loc[current_historical.policy.eq(selected), metric_columns].to_dict("records"),
        "evaluation_basis": "July model projection; CP07 evaluation is retrospective, not an untouched new test",
        "global_optimality_guaranteed": False,
        "control_design_change": "None; stratified and optimize-first designs remain options for the project manager",
    }
    (root / "outputs/allocator_revision_validation.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
