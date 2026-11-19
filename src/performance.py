"""Rebuild model searches, honest comparisons, and separate deployment decisions."""
import json
from pathlib import Path
import joblib
import numpy as np
import pandas as pd

from src.modeling import ACTIONS, predict_actions, fit_action_models
from src.profit_models import fit_profit_models
from src.policy import ALLOCATOR_NAME
from src.evaluation import cluster_dr_interval, _paired_dr_frame, tune_on_prior_campaigns


def load_allocation_policy_config(root=None):
    """Load the explicit deployment choice independently of policy validation."""
    root = Path(root) if root is not None else Path(__file__).resolve().parents[1]
    config = json.loads((root / "config/allocation_policy.json").read_text(encoding="utf-8"))
    if config.get("mode") != "forecast_champion_greedy":
        raise ValueError("Unknown allocation-policy mode")
    if config.get("allocator") != ALLOCATOR_NAME:
        raise ValueError("Unknown deployment allocator")
    return config


def publish_policy_validation(history, policy, profit, tables, models_dir, draws=2000):
    """Publish saved policy results without refitting unchanged profit models.

    ``policy`` and ``profit`` are the already saved selection records. Return
    the combined deployment selection and the CP07 policy summary. This helper
    neither runs a selection search nor changes a fitted model artifact. The
    explicit deployment configuration may differ from the development winner;
    retain both decisions rather than implying the requested policy won.
    """
    tables, models_dir = Path(tables), Path(models_dir)
    root = tables.parents[1]
    policy_out = root / "outputs/experiments/policy"
    forecast = profit["selected_model"]
    config = load_allocation_policy_config(root)
    if policy.get("model_allocator") != config["allocator"]:
        raise ValueError("Rerun policy validation using the configured greedy allocator before publication")
    selected = forecast
    selection = {**policy, "selected_policy": selected, "selected_method": selected,
                 "selection_role": "Configured deployment; development comparison preserved in validation_policy_selection",
                 "selected_config": profit["config"], "forecast_model": forecast,
                 "allocation_model": forecast, "allocator": config["allocator"],
                 "selection_metric": config["selection_rule"],
                 "rule": config["selection_rule"],
                 "deployment_selection_basis": config["selection_rule"],
                 "allocation_policy_config": config,
                 "validation_selected_policy": policy["selected_policy"],
                 "validation_policy_selection": policy,
                 "prediction_selection": profit,
                 "distinction": "Forecast model minimizes factual RMSE. Deployment uses its predictions with the user-requested greedy allocator; the development policy winner remains a separate comparison."}
    summary = pd.read_csv(policy_out / "retrospective_cp07_summary.csv")
    scores = pd.read_csv(policy_out / "retrospective_cp07_scores.csv")
    allocations = pd.read_csv(policy_out / "retrospective_cp07_allocations.csv")
    if selected not in set(summary.policy) or selected not in set(scores.policy) or selected not in set(allocations.policy):
        raise ValueError("The configured deployment model must have a completed greedy-policy audit")
    for frame in (summary, allocations):
        if ("allocation_algorithm" not in frame or
                not frame.loc[frame.policy.eq(selected), "allocation_algorithm"].eq(config["allocator"]).all()):
            raise ValueError("Saved deployment audit rows use a stale allocation algorithm")
    (models_dir / "policy_selection.json").write_text(json.dumps(selection, indent=2)+"\n", encoding="utf-8")
    summary["selected_for_july"] = summary.policy.eq(selected)
    summary["selected_by_development"] = summary.policy.eq(policy["selected_policy"])
    summary.to_csv(tables / "heldout_policy_comparison.csv", index=False)
    scores.to_csv(tables / "heldout_policy_dr_scores.csv", index=False)
    allocations.to_csv(tables / "heldout_policy_assignments.csv", index=False)
    pairs = []
    for reference in scores.policy.unique():
        point, low, high = cluster_dr_interval(_paired_dr_frame(scores, selected, reference), draws=draws)
        pairs.append(dict(policy=selected, reference=reference, paired_dr_difference=point, ci_low=low, ci_high=high))
    pd.DataFrame(pairs).to_csv(tables / "heldout_policy_paired_comparisons.csv", index=False)
    test = history[history.campaign_id.eq("CP07") & history.assignment_source.eq("exploration")]
    seen = set(history.loc[history.campaign_id.lt("CP07"), "customer_id"])
    scores["unseen_customer"] = ~scores.customer_id.isin(seen)
    scores["unseen_no_observed_recent_orders"] = scores.first_exposure_no_observed_recent_orders & scores.unseen_customer
    scores.to_csv(tables / "heldout_policy_dr_scores.csv", index=False)
    subgroups = []
    for group in ["first_exposure_no_observed_recent_orders", "unseen_customer", "unseen_no_observed_recent_orders"]:
        for name, part in scores.groupby("policy"):
            included = part[group].to_numpy(bool)
            point, low, high = cluster_dr_interval(part, included=included, draws=draws) if included.any() else (np.nan,)*3
            subgroups.append(dict(subgroup=group, policy=name, n=int(included.sum()),
                matched_rows=int(part.loc[included,"matched"].sum()),
                dr_incremental_ncm_per_customer=point, ci_low=low, ci_high=high,
                logging_population="entire evaluation campaign, not subgroup"))
    pd.DataFrame(subgroups).to_csv(tables / "heldout_cold_start_evaluation.csv", index=False)
    for source, target in [("development_ranking.csv", "policy_candidate_selection.csv"),
                           ("development_fold_summary.csv", "rolling_policy_validation.csv")]:
        pd.read_csv(policy_out / source).to_csv(tables / target, index=False)
    summary.assign(policy_scope="feasible constrained policy").to_csv(tables / "tlearner_policy_value_cp07.csv", index=False)
    return selection, summary


def run_performance_validation(history, num, cat, tables, models_dir, draws=2000):
    """Select prediction, compare policies, and apply the deployment choice.

    CP07 has been inspected before this revision. It is never an input to either
    selection rule, but the resulting audit is explicitly retrospective.
    """
    from scripts.experiment_profit import run_profit_search, evaluate_profit_selection
    from scripts.experiment_policy import run_policy_search

    tables, models_dir = Path(tables), Path(models_dir)
    root = tables.parents[1]
    profit_out = root / "outputs/experiments/profit"
    policy_out = root / "outputs/experiments/policy"
    profit = run_profit_search(history, num, cat, profit_out)
    run_policy_search(history, num, cat, policy_out, draws=draws)
    # Development results are serialized before their final-model fits. The
    # deployment choice is explicit configuration, not a claim it won policy search.
    predictive_metrics, predictions = evaluate_profit_selection(history, num, cat, profit, profit_out)
    policy = json.loads((policy_out / "selection.json").read_text())
    selection, summary = publish_policy_validation(history, policy, profit, tables, models_dir, draws=draws)
    forecast, allocation_model = selection["forecast_model"], selection["allocation_model"]
    pd.read_csv(profit_out / "development_fold_metrics.csv").to_csv(tables / "rolling_model_metrics.csv", index=False)
    predictive_metrics.assign(split="CP07").to_csv(tables / "model_comparison_cp07.csv", index=False)
    chosen_metrics = predictive_metrics[predictive_metrics.model.eq(forecast)]
    chosen_metrics.to_csv(tables / "xgboost_model_metrics.csv", index=False)
    chosen_metrics.drop(columns="model").to_csv(tables / "tlearner_test_metrics.csv", index=False)
    chosen_predictions = predictions[predictions.model.eq(forecast)].copy()
    for action in ACTIONS:
        chosen_predictions[f"incremental_ncm_{action}"] = chosen_predictions[f"predicted_ncm_{action}"] - chosen_predictions.predicted_ncm_NO_OFFER
    pcols = [f"predicted_ncm_{a}" for a in ACTIONS]
    chosen_predictions["recommended_action_unconstrained"] = np.array(ACTIONS)[chosen_predictions[pcols].to_numpy().argmax(axis=1)]
    for name in ["xgboost_test_predictions.csv", "tlearner_test_predictions.csv"]:
        chosen_predictions.to_csv(tables / name, index=False)
    final = fit_profit_models(history[history.assignment_source.eq("exploration")], num, cat, profit["config"])
    exp = history[history.assignment_source.eq("exploration")]
    params, _ = tune_on_prior_campaigns(exp[exp.campaign_id.lt("CP07")], num, cat)
    allocation_models = final if allocation_model == forecast else fit_action_models(exp, num, cat, params, method=allocation_model)
    for action, model in final.items():
        # Legacy filenames retained for existing notebooks and saved dashboards.
        joblib.dump(model, models_dir / f"tlearner_ncm_{action}.joblib")
        joblib.dump(model, models_dir / f"xgboost_ncm_{action}.joblib")
        joblib.dump(allocation_models[action], models_dir / f"allocation_ncm_{action}.joblib")
    return selection, predictive_metrics, summary, final
