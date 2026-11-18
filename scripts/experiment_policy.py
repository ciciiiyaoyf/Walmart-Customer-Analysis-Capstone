"""Select targeting rules on CP04-06, then assess the locked rule on CP07.

Run from the repository root: python scripts/experiment_policy.py --draws 2000
CP07 has been inspected in earlier project work, so it is a retrospective audit,
not a pristine holdout. It never enters selection in this experiment.
"""
import argparse
import faulthandler
import json
import multiprocessing
import os
from pathlib import Path
import sys
import traceback

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits
from src.modeling import (ACTIONS, feature_lists, fit_action_models, predict_actions, reg_metrics,
                          fit_logistic_baseline, load_logistic_baseline_config)
from src.profit_models import PROFIT_CANDIDATES, fit_profit_models
from src.evaluation import (JULY_BUDGET, JULY_SIZE, tune_on_prior_campaigns, fit_cost_models,
                            predict_costs, value_table, evaluate_policy, cluster_dr_interval,
                            _paired_dr_frame)
from src.policy import optimize_policy, build_baseline_policies, get_holdout
from src.policy_learning import POLICY_GRID, fit_segment_models, policy_name, stochastic_free_ship_scores


def _campaign_worker(connection, history, campaign, num, cat, draws, candidate_names):
    """Importable spawn target; native crashes cannot take down the notebook."""
    faulthandler.enable()
    try:
        result = evaluate_campaign(history, campaign, num, cat, draws, candidate_names=candidate_names)
        connection.send(("ok", result))
    except BaseException:
        connection.send(("error", traceback.format_exc()))
    finally:
        connection.close()


def isolated_evaluate_campaign(history, campaign, num, cat, draws, candidate_names=None, timeout=300.):
    """Evaluate one campaign in a fresh process, preserving all native settings.

    Repeated XGBoost fitting in a long-lived Windows notebook kernel caused a
    native crash during CP07. A spawn worker limits native-library state to one
    fold. The parent receives only ordinary data frames, with a bounded wait and
    explicit errors for Python failures, native crashes, or stalled workers.
    """
    if not np.isfinite(timeout) or timeout <= 0:
        raise ValueError("Worker timeout must be positive and finite")
    context = multiprocessing.get_context("spawn")
    receiver, sender = context.Pipe(duplex=False)
    worker = context.Process(target=_campaign_worker,
                             args=(sender, history, campaign, num, cat, draws, candidate_names),
                             name=f"policy-{campaign}")
    started = False
    try:
        worker.start()
        started = True
        sender.close()
        if not receiver.poll(timeout):
            raise TimeoutError(f"Policy evaluation {campaign} exceeded {timeout:g} seconds")
        try:
            status, payload = receiver.recv()
        except (EOFError, OSError) as error:
            worker.join(timeout=1.)
            raise RuntimeError(f"Policy worker {campaign} exited without a result (exit code {worker.exitcode})") from error
        worker.join(timeout=10.)
        if worker.is_alive():
            raise RuntimeError(f"Policy worker {campaign} did not shut down after returning its result")
        if status != "ok":
            raise RuntimeError(f"Policy evaluation {campaign} failed in its worker:\n{payload}")
        if worker.exitcode != 0:
            raise RuntimeError(f"Policy worker {campaign} exited with code {worker.exitcode}")
        return payload
    finally:
        sender.close()
        receiver.close()
        # Only the process created by this call is ever terminated.
        if started and worker.is_alive():
            worker.terminate()
            worker.join(timeout=10.)
        if started and not worker.is_alive():
            worker.close()


def evaluate_campaign(history, campaign, num, cat, draws, candidate_names=None):
    train_all = history[history.campaign_id < campaign].copy()
    train = train_all[train_all.assignment_source.eq("exploration")].copy()
    test = history[history.campaign_id.eq(campaign) & history.assignment_source.eq("exploration")].sort_values("customer_id").copy()
    if train.send_ts.max() + pd.Timedelta(days=14) > test.send_ts.min():
        raise ValueError("Training labels have not matured")
    params, inner_campaign = tune_on_prior_campaigns(train, num, cat)
    original = fit_action_models(train, num, cat, params)
    nuisance = predict_actions(original, test[num+cat])
    costs = predict_costs(fit_cost_models(train, num, cat, params), test[num+cat])
    kwargs = dict(budget=JULY_BUDGET*len(test)/JULY_SIZE, holdout_ids=get_holdout(test.customer_id))
    # Use the original, fixed logistic specification requested for the July
    # purchase-propensity baseline. Each fold fits earlier campaigns only;
    # there is no separate feature-transform or regularization search here.
    # Dense logistic fits do not benefit from the machine-wide BLAS pool here.
    # Scope the limit to this stage and restore it before XGBoost/HiGHS; changing
    # OpenMP settings globally can change XGBoost's randomized training results.
    with threadpool_limits(limits=4, user_api="blas"):
        purchase_config = load_logistic_baseline_config()["config"]
        purchase = fit_logistic_baseline(train_all, num, cat).predict_proba(test[num+cat])[:, 1]
    reference_values = value_table(test, nuisance, costs, purchase)
    modelsets = {"ORIGINAL_TLEARNER": original,
                 "ACTION_MEAN": fit_action_models(train, num, cat, params, method="ACTION_MEAN")}
    for method in ["SHRUNK_TLEARNER", "POOLED_XGBOOST"]:
        if candidate_names is None or method in candidate_names:
            modelsets[method] = fit_action_models(train, num, cat, params, method=method, original=original)
    if candidate_names is None or "HURDLE_XGB_HALF" in candidate_names:
        modelsets["HURDLE_XGB_HALF"] = fit_profit_models(train, num, cat, PROFIT_CANDIDATES["HURDLE_XGB_HALF"])
    modelsets.update({policy_name(config): fit_segment_models(train, **config) for config in POLICY_GRID
                     if candidate_names is None or policy_name(config) in candidate_names})
    summaries, scores, metrics, allocations = [], [], [], []
    for name, models in modelsets.items():
        predictions = predict_actions(models, test[num+cat])
        allocation = optimize_policy(value_table(test, predictions, costs, purchase), **kwargs)
        summary, individual = evaluate_policy(test, allocation, nuisance, costs, name, campaign, draws=draws)
        summary["inner_validation_campaign"] = inner_campaign
        summary["fold_parameters"] = json.dumps(params, sort_keys=True)
        summaries.append(summary)
        scores.append(individual)
        allocations.append(allocation.assign(policy=name, evaluation_campaign=campaign))
        factual = predictions[np.arange(len(test)), [ACTIONS.index(a) for a in test.offer_type]]
        metrics.append(dict(split=campaign, model=name, **reg_metrics(test.ncm_14d, factual)))
    hist_mix = train_all.loc[train_all.assignment_source.eq("business_rule") & train_all.offer_type.ne("NO_OFFER"), "offer_type"].value_counts(normalize=True).to_dict()
    for name, allocation in build_baseline_policies(reference_values, historical_mix=hist_mix, **kwargs).items():
        summary, individual = evaluate_policy(test, allocation, nuisance, costs, name, campaign, draws=draws)
        summaries.append(summary)
        scores.append(individual)
        allocations.append(allocation.assign(policy=name, evaluation_campaign=campaign))
    scores = pd.concat(scores, ignore_index=True)
    scores["unseen_customer"] = ~scores.customer_id.isin(train_all.customer_id)
    scores["unseen_no_observed_recent_orders"] = scores.first_exposure_no_observed_recent_orders & scores.unseen_customer
    stochastic = stochastic_free_ship_scores(test, nuisance, costs, kwargs["holdout_ids"], kwargs["budget"])
    summaries = pd.DataFrame(summaries)
    summaries["purchase_baseline_config"] = json.dumps(purchase_config, sort_keys=True)
    summaries["purchase_validation_campaigns"] = "[]"
    summaries["purchase_baseline_selection"] = "user-fixed original logistic; earlier-campaign fit only; no nested search"
    return (summaries, scores, pd.DataFrame(metrics),
            pd.concat(allocations, ignore_index=True), stochastic)


def save_randomization_robustness(development_stochastic, audit_stochastic, development_scores, audit_scores, out, draws):
    out = Path(out)
    robustness = []
    for split, frame, references in [("CP04-CP06", development_stochastic, development_scores), ("CP07", audit_stochastic, audit_scores)]:
        point, low, high = cluster_dr_interval(frame, draws=draws)
        robustness.append(dict(split=split, comparison="versus NO_OFFER", gain=point, ci_low=low, ci_high=high))
        combined = pd.concat([frame, references], ignore_index=True)
        for reference in ["ORIGINAL_TLEARNER", "RANDOM_FREE_SHIP"]:
            point, low, high = cluster_dr_interval(_paired_dr_frame(combined, "RANDOM_FREE_SHIP_EXPECTATION", reference), draws=draws)
            robustness.append(dict(split=split, comparison=f"versus {reference}", gain=point, ci_low=low, ci_high=high))
    robustness = pd.DataFrame(robustness)
    robustness["policy"] = "RANDOM_FREE_SHIP_EXPECTATION"
    robustness["interpretation"] = "averages recipient randomization; fixed seed-42 controls; never used for selection"
    robustness.to_csv(out/"stochastic_free_ship_robustness.csv", index=False)
    pd.concat([development_stochastic, audit_stochastic], ignore_index=True).to_csv(out/"stochastic_free_ship_scores.csv", index=False)
    return robustness


def run_randomization_robustness(history, num, cat, out, draws=2000):
    """Recompute the diagnostic only, preserving the locked model/policy choices."""
    stochastic, references = {}, {}
    for campaign in ["CP04", "CP05", "CP06", "CP07"]:
        print(f"Randomization robustness {campaign}", flush=True)
        train = history[(history.campaign_id < campaign) & history.assignment_source.eq("exploration")]
        test = history[history.campaign_id.eq(campaign) & history.assignment_source.eq("exploration")].sort_values("customer_id")
        params, _ = tune_on_prior_campaigns(train, num, cat)
        nuisance = predict_actions(fit_action_models(train, num, cat, params), test[num+cat])
        costs = predict_costs(fit_cost_models(train, num, cat, params), test[num+cat])
        kwargs = dict(budget=JULY_BUDGET*len(test)/JULY_SIZE, holdout_ids=get_holdout(test.customer_id))
        values = value_table(test, nuisance, costs, np.zeros(len(test)))
        policies = dict(ORIGINAL_TLEARNER=optimize_policy(values, **kwargs),
                        RANDOM_FREE_SHIP=build_baseline_policies(values, **kwargs)["RANDOM_FREE_SHIP"])
        references[campaign] = pd.concat([evaluate_policy(test, allocation, nuisance, costs, name, campaign, draws=2)[1]
                                         for name, allocation in policies.items()], ignore_index=True)
        stochastic[campaign] = stochastic_free_ship_scores(test, nuisance, costs, kwargs["holdout_ids"], kwargs["budget"])
    report = save_randomization_robustness(pd.concat([stochastic[c] for c in ["CP04", "CP05", "CP06"]], ignore_index=True),
        stochastic["CP07"], pd.concat([references[c] for c in ["CP04", "CP05", "CP06"]], ignore_index=True),
        references["CP07"], out, draws)
    print(report.to_string(index=False), flush=True)
    return report


def run_policy_search(history, num, cat, out, draws=2000):
    """Freeze development winner, then return its CP07 audit and fixed references."""
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    development = []
    for campaign in ["CP04", "CP05", "CP06"]:
        print(f"Evaluating development {campaign}", flush=True)
        development.append(isolated_evaluate_campaign(history, campaign, num, cat, draws))
        development[-1][0].to_csv(out/f"{campaign}_summary.csv", index=False)
    summary = pd.concat([r[0] for r in development], ignore_index=True)
    scores = pd.concat([r[1] for r in development], ignore_index=True)
    metrics = pd.concat([r[2] for r in development], ignore_index=True)
    ranking = scores.groupby("policy").dr_incremental_score.agg(["size", "mean"]).reset_index()
    ranking = ranking.rename(columns={"mean": "development_dr_incremental_ncm", "size": "n"})
    ranking = ranking.sort_values(["development_dr_incremental_ncm", "policy"], ascending=[False, True]).reset_index(drop=True)
    selected = ranking.iloc[0].policy
    config = next((c for c in POLICY_GRID if policy_name(c) == selected), None)
    decision = dict(selected_policy=selected, selected_config=config,
                    selection_campaigns=["CP04", "CP05", "CP06"],
                    selection_metric="maximum pooled customer-weighted DR incremental NCM under common feasible constraints",
                    candidate_count=len(ranking), holdout_used_for_selection=False,
                    audit_campaign="CP07", audit_status="retrospective: already inspected in prior project work",
                    uncertainty="Validation winner is selection-biased; intervals condition on fixed models and allocations.",
                    grid=POLICY_GRID, additional_profit_model=PROFIT_CANDIDATES["HURDLE_XGB_HALF"],
                    random_seed=42, bootstrap_draws=draws, OMP_NUM_THREADS=os.environ.get("OMP_NUM_THREADS", "unset"))
    decision["purchase_baseline_selection"] = "user-fixed original logistic from config/logistic_baseline.json; earlier-campaign fit only; no nested search"
    decision["execution"] = "fresh spawn process per campaign; 300-second worker timeout; native OpenMP settings unchanged"
    # This file is serialized before any CP07 target values are evaluated.
    (out/"selection.json").write_text(json.dumps(decision, indent=2)+"\n")
    ranking.to_csv(out/"development_ranking.csv", index=False)
    summary.to_csv(out/"development_fold_summary.csv", index=False)
    metrics.to_csv(out/"development_factual_metrics.csv", index=False)
    print(f"Locked selection: {selected}\n{ranking.to_string(index=False)}", flush=True)
    paired = _paired_dr_frame(scores, selected, "ORIGINAL_TLEARNER")
    dev_gain = cluster_dr_interval(paired, draws=draws)
    # Audit only the locked choice plus prespecified references.
    print("Auditing locked selection on CP07", flush=True)
    keep = [selected, "HURDLE_XGB_HALF", "ORIGINAL_TLEARNER", "ACTION_MEAN", "RANDOM_FREE_SHIP",
            "ALL_NO_OFFER", "PROPENSITY_PCT10", "HISTORICAL_MIX"]
    audit_summary, audit_scores, audit_metrics, audit_allocations, audit_stochastic = isolated_evaluate_campaign(history, "CP07", num, cat, draws, candidate_names=keep)
    audit_summary.to_csv(out/"retrospective_cp07_summary.csv", index=False)
    audit_metrics[audit_metrics.model.isin(keep)].to_csv(out/"retrospective_cp07_factual_metrics.csv", index=False)
    audit_allocations[audit_allocations.policy.isin(keep)].to_csv(out/"retrospective_cp07_allocations.csv", index=False)
    audit_scores.to_csv(out/"retrospective_cp07_scores.csv", index=False)
    comparisons = []
    for reference in ["ORIGINAL_TLEARNER", "ACTION_MEAN", "HURDLE_XGB_HALF", "RANDOM_FREE_SHIP"]:
        if reference == selected:
            continue
        point, low, high = cluster_dr_interval(_paired_dr_frame(audit_scores, selected, reference), draws=draws)
        comparisons.append(dict(split="CP07", policy=selected, reference=reference, gain=point, ci_low=low, ci_high=high))
    comparisons.append(dict(split="CP04-CP06", policy=selected, reference="ORIGINAL_TLEARNER", gain=dev_gain[0], ci_low=dev_gain[1], ci_high=dev_gain[2]))
    pd.DataFrame(comparisons).to_csv(out/"paired_comparisons.csv", index=False)
    development_stochastic = pd.concat([r[4] for r in development], ignore_index=True)
    robustness = save_randomization_robustness(development_stochastic, audit_stochastic, scores, audit_scores, out, draws)
    print(audit_summary[["policy", "dr_incremental_ncm_per_customer", "ci_low", "ci_high", "offered_share"]].to_string(index=False), flush=True)
    print(pd.DataFrame(comparisons).to_string(index=False), flush=True)
    print(robustness.to_string(index=False), flush=True)
    return decision, dict(summary=audit_summary, scores=audit_scores, metrics=audit_metrics,
                         allocations=audit_allocations, development_ranking=ranking,
                         development_summary=summary, development_scores=scores,
                         paired_comparisons=pd.DataFrame(comparisons), stochastic_robustness=robustness)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--draws", type=int, default=2000)
    parser.add_argument("--robustness-only", action="store_true")
    args = parser.parse_args()
    history = pd.read_parquet("data/processed/historical_feature_outcome.parquet")
    num, cat = feature_lists()
    runner = run_randomization_robustness if args.robustness_only else run_policy_search
    runner(history, num, cat, "outputs/experiments/policy", draws=args.draws)


if __name__ == "__main__":
    main()
