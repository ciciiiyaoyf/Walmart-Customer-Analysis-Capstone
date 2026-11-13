"""Temporal, feasible policy evaluation using logged randomized outcomes.

Policies and reward/cost nuisance models use earlier campaigns only. CP04-06 are
development folds; CP07 is evaluated after the selection decision is serialized.
Customer-cluster bootstrap intervals re-estimate logging frequencies within
campaign on every draw, including subgroup analyses. They condition on trained
outcome/cost models and fixed batch allocations and do not capture future
campaign shifts or the full uncertainty of model retraining.
"""
import json
from pathlib import Path
import numpy as np
import pandas as pd
import joblib
from src.modeling import (ACTIONS, COMMERCIAL, CANDIDATES, XGB_GRID, ConstantPredictor, fit_action_models,
                          predict_actions, make_xgb_pipe, make_propensity_pipe, reg_metrics)
from src.policy import optimize_policy, build_baseline_policies, get_holdout

JULY_SIZE = 2776
JULY_BUDGET = 62000.0


def cluster_mean_interval(values, customers, alpha=0.05, draws=2000, seed=42):
    """Customer bootstrap, retaining every campaign observation in a sampled cluster."""
    values = np.asarray(values, dtype=float)
    customers = np.asarray(customers)
    if not len(values) or len(values) != len(customers) or not np.isfinite(values).all():
        raise ValueError("Finite nonempty observations and matching customer IDs required")
    if not 0 < alpha < 1 or draws < 2:
        raise ValueError("Invalid interval parameters")
    _, inv = np.unique(customers, return_inverse=True)
    k = inv.max() + 1
    if k < 2:
        return float(values.mean()), np.nan, np.nan
    sums = np.bincount(inv, weights=values, minlength=k)
    counts = np.bincount(inv, minlength=k)
    rng = np.random.default_rng(seed)
    boot = np.empty(draws)
    for start in range(0, draws, 100):
        idx = rng.integers(0, k, size=(min(100, draws-start), k))
        boot[start:start+len(idx)] = sums[idx].sum(axis=1) / counts[idx].sum(axis=1)
    lo, hi = np.quantile(boot, [alpha/2, 1-alpha/2])
    return float(values.mean()), float(lo), float(hi)


def cluster_dr_interval(frame, direct_column="direct_component", residual_column="residual_component",
                        included=None, alpha=0.05, draws=2000, seed=42):
    """Bootstrap a DR mean while re-estimating each campaign's logging policy.

    Each row contributes ``direct + residual / p(observed_action | campaign)``.
    Sample customers with all their campaign observations. The optional boolean
    ``included`` mask limits the outcome mean (e.g. a cold-start subgroup), but
    logging probabilities always use the entire resampled population supplied
    in ``frame``. Pass one row per decision, not stacked competing policies.

    Draws lacking an observed treatment arm in a represented campaign, or lacking
    subgroup observations, cannot identify the requested estimate. They are
    skipped; if too many draws fail support, return undefined interval endpoints
    rather than claim precision from a heavily conditioned bootstrap.
    """
    required = {"customer_id", "campaign_id", "observed_action", direct_column, residual_column}
    if not required.issubset(frame):
        raise ValueError(f"Missing DR bootstrap columns: {sorted(required - set(frame))}")
    if frame.empty or frame[["customer_id", "campaign_id", "observed_action"]].isna().any().any():
        raise ValueError("DR bootstrap requires nonempty rows and non-null identifiers/actions")
    if "assignment_id" in frame and not frame.assignment_id.is_unique:
        raise ValueError("DR bootstrap requires one row per decision, not stacked policies")
    if not frame.observed_action.isin(ACTIONS).all() or not 0 < alpha < 1 or draws < 2:
        raise ValueError("Invalid logging actions or interval parameters")
    direct = frame[direct_column].to_numpy(float)
    residual = frame[residual_column].to_numpy(float)
    if not np.isfinite(direct).all() or not np.isfinite(residual).all():
        raise ValueError("DR bootstrap components must be finite")
    mask = np.ones(len(frame), dtype=bool) if included is None else np.asarray(included)
    if mask.shape != (len(frame),) or mask.dtype.kind != "b" or not mask.any():
        raise ValueError("included must be a nonempty matching boolean mask")
    _, customer = np.unique(frame.customer_id.to_numpy(), return_inverse=True)
    _, campaign = np.unique(frame.campaign_id.to_numpy(), return_inverse=True)
    action = frame.observed_action.map({a: j for j, a in enumerate(ACTIONS)}).to_numpy()
    k, campaigns = customer.max() + 1, campaign.max() + 1
    groups = campaigns * len(ACTIONS)
    group = campaign * len(ACTIONS) + action
    # Aggregate once, then each sampled customer retains every decision row.
    cell = customer * groups + group
    counts = np.bincount(cell, minlength=k * groups).reshape(k, groups)
    residuals = np.bincount(cell[mask], weights=residual[mask], minlength=k * groups).reshape(k, groups)
    directs = np.bincount(customer[mask], weights=direct[mask], minlength=k)
    included_counts = np.bincount(customer[mask], minlength=k)
    observed_counts = counts.sum(axis=0).reshape(campaigns, len(ACTIONS))
    if (observed_counts == 0).any():
        raise ValueError("Every evaluation campaign needs positive support for all four actions")
    inverse_p = np.repeat(observed_counts.sum(axis=1), len(ACTIONS)) / observed_counts.ravel()
    point = float((directs.sum() + residuals.sum(axis=0) @ inverse_p) / included_counts.sum())
    if k < 2:
        return point, np.nan, np.nan
    if np.all(direct[mask] == direct[mask][0]) and np.all(residual[mask] == 0):
        return point, point, point
    rng = np.random.default_rng(seed)
    samples = []
    attempted = 0
    valid_count = 0
    while valid_count < draws and attempted < draws * 10:
        batch = min(100, draws - valid_count)
        idx = rng.integers(0, k, size=(batch, k))
        action_counts = counts[idx].sum(axis=1).reshape(batch, campaigns, len(ACTIONS))
        campaign_counts = action_counts.sum(axis=2)
        denominator = included_counts[idx].sum(axis=1)
        supported = (((action_counts > 0).all(axis=2)) | (campaign_counts == 0)).all(axis=1)
        supported &= denominator > 0
        inverse = np.divide(np.repeat(campaign_counts, len(ACTIONS), axis=1), action_counts.reshape(batch, groups),
                            out=np.zeros((batch, groups), dtype=float), where=action_counts.reshape(batch, groups) > 0)
        numerator = directs[idx].sum(axis=1) + (residuals[idx].sum(axis=1) * inverse).sum(axis=1)
        if supported.any():
            samples.append(numerator[supported] / denominator[supported])
            valid_count += int(supported.sum())
        attempted += batch
    # With the supplied campaign sizes, arm disappearance is negligibly rare.
    # This guard avoids spuriously tight intervals in tiny unsupported examples.
    if valid_count < draws or valid_count / attempted < 0.8:
        return point, np.nan, np.nan
    low, high = np.quantile(np.concatenate(samples)[:draws], [alpha / 2, 1 - alpha / 2])
    return point, float(low), float(high)


def _paired_dr_frame(scores, policy, reference):
    """One full-population row per decision, with paired DR components."""
    index = ["assignment_id", "customer_id", "campaign_id"]
    left = scores.loc[scores.policy.eq(policy)].set_index(index)
    right = scores.loc[scores.policy.eq(reference)].set_index(index)
    if not left.index.is_unique or not right.index.is_unique or set(left.index) != set(right.index):
        raise ValueError("Compared policies must cover exactly the same unique decisions")
    right = right.reindex(left.index)
    if not left.observed_action.equals(right.observed_action):
        raise ValueError("Compared policies disagree on logged actions")
    paired = left[["observed_action"]].copy()
    paired["direct_component"] = left.direct_component - right.direct_component
    paired["residual_component"] = left.residual_component - right.residual_component
    return paired.reset_index()


def dr_components(y, observed_action, policy_action, mu, propensity):
    """Per-row DR reward score; all four potential-outcome predictions are required."""
    y, mu, p = np.asarray(y, float), np.asarray(mu, float), np.asarray(propensity, float)
    observed_action, policy_action = np.asarray(observed_action), np.asarray(policy_action)
    n = len(y)
    if y.ndim != 1 or observed_action.shape != (n,) or policy_action.shape != (n,):
        raise ValueError("Outcomes and both action arrays must have matching one-dimensional shapes")
    if mu.shape != (n, len(ACTIONS)) or p.shape != (n,):
        raise ValueError("Prediction/propensity dimensions do not match logged outcomes")
    if not np.isfinite(mu).all() or not np.isfinite(y).all() or not np.isfinite(p).all() or (p <= 0).any() or (p > 1).any():
        raise ValueError("Finite outcomes and positive logging probabilities are required")
    if not set(observed_action).issubset(ACTIONS) or not set(policy_action).issubset(ACTIONS):
        raise ValueError("Invalid action")
    observed = np.array([ACTIONS.index(a) for a in observed_action])
    policy = np.array([ACTIONS.index(a) for a in policy_action])
    weights = (observed_action == policy_action).astype(float) / p
    phi = mu[np.arange(n), policy] + weights * (y - mu[np.arange(n), observed])
    return phi, weights


def evaluate_policy(data, allocation, mu, cost_mu, policy_name, split, draws=2000):
    if not data.customer_id.is_unique or not allocation.customer_id.is_unique:
        raise ValueError("One decision per customer required within an evaluation campaign")
    if set(data.customer_id) != set(allocation.customer_id):
        raise ValueError("Policy must cover exactly the evaluation population")
    if data.campaign_id.nunique() != 1:
        raise ValueError("evaluate_policy expects one campaign; pool decomposed scores only after evaluation")
    chosen = allocation.set_index("customer_id").reindex(data.customer_id)
    if chosen.recommended_action.isna().any():
        raise ValueError("Policy does not cover the evaluation population")
    observed = data.offer_type.to_numpy()
    policy = chosen.recommended_action.to_numpy()
    # No logging probabilities are supplied. Empirical frequencies are valid only
    # under unconditional randomization within campaign; uniform assignment is a sensitivity.
    counts = data.offer_type.value_counts(normalize=True)
    if set(counts.index) != set(ACTIONS):
        raise ValueError("All four actions need positive evaluation support")
    p = data.offer_type.map(counts).to_numpy()
    y = data.ncm_14d.to_numpy(float)
    no = np.full(len(data), "NO_OFFER")
    phi, w = dr_components(y, observed, policy, mu, p)
    phi0, w0 = dr_components(y, observed, no, mu, p)
    delta = phi - phi0
    observed_index = np.array([ACTIONS.index(a) for a in observed])
    policy_index = np.array([ACTIONS.index(a) for a in policy])
    observed_mu = np.asarray(mu)[np.arange(len(data)), observed_index]
    individual = data[["assignment_id", "customer_id", "campaign_id"]].copy()
    individual["policy"] = policy_name
    individual["recommended_action"] = policy
    individual["outcome"] = y
    individual["observed_action"] = observed
    individual["direct_component"] = np.asarray(mu)[np.arange(len(data)), policy_index] - np.asarray(mu)[:, 0]
    individual["residual_component"] = ((observed == policy).astype(float) - (observed == no).astype(float)) * (y - observed_mu)
    val, low, high = cluster_dr_interval(individual, draws=draws)
    if not np.isclose(val, delta.mean()):
        raise AssertionError("DR decomposition does not reproduce the point estimate")
    cost_phi, _ = dr_components(data.promo_cost_14d, observed, policy, cost_mu, p)
    cost_components = individual[["assignment_id", "customer_id", "campaign_id", "observed_action"]].copy()
    cost_components["direct_component"] = np.asarray(cost_mu)[np.arange(len(data)), policy_index]
    cost_components["residual_component"] = (observed == policy).astype(float) * (
        data.promo_cost_14d.to_numpy(float) - np.asarray(cost_mu)[np.arange(len(data)), observed_index])
    cost_value, cost_low, cost_high = cluster_dr_interval(cost_components, draws=draws)
    if not np.isclose(cost_value, cost_phi.mean()):
        raise AssertionError("Cost DR decomposition does not reproduce the point estimate")
    uniform, _ = dr_components(y, observed, policy, mu, np.full(len(data), 0.25))
    uniform0, _ = dr_components(y, observed, no, mu, np.full(len(data), 0.25))
    hajek = np.sum(w*y)/np.sum(w) - np.sum(w0*y)/np.sum(w0) if w.sum() else np.nan
    summary = {"split": split, "policy": policy_name, "n": len(data),
               "dr_ncm_per_customer": float(phi.mean()), "dr_incremental_ncm_per_customer": val,
               "ci_low": low, "ci_high": high, "hajek_incremental_ncm_per_customer": float(hajek),
               "uniform_propensity_dr_incremental_ncm": float(np.mean(uniform-uniform0)),
               "estimated_promo_spend": float(chosen.estimated_promo_cost.sum()),
               "dr_promo_spend": cost_value*len(data), "dr_promo_spend_ci_low": cost_low*len(data),
               "dr_promo_spend_ci_high": cost_high*len(data),
               "offered_share": float(np.mean(policy != "NO_OFFER")),
               "matched_rows": int(np.sum(observed == policy)),
               "effective_sample_size": float(w.sum()**2 / np.sum(w*w)) if np.sum(w*w) else 0.,
               "budget": JULY_BUDGET*len(data)/JULY_SIZE,
               "propensity_source": "empirical_within_campaign_unconditional_randomization_assumed",
               "interval_method": "customer_cluster_bootstrap_reestimating_campaign_logging_frequencies"}
    individual["dr_score"] = phi
    individual["dr_no_offer_score"] = phi0
    individual["dr_incremental_score"] = delta
    individual["matched"] = observed == policy
    individual["cold_history"] = data.prior_emails_delivered.eq(0).to_numpy() & data.orders_90d.eq(0).to_numpy()
    return summary, individual


def value_table(data, mu, costs, purchase_probability):
    return pd.concat([data[["customer_id", "campaign_id", "send_ts"]].rename(columns={"send_ts":"policy_as_of_ts"}).assign(
        action=a, predicted_ncm=mu[:, j], estimated_incremental_ncm=mu[:, j]-mu[:, 0],
        estimated_promo_cost=costs[:, j], predicted_purchase_probability=purchase_probability)
        for j,a in enumerate(ACTIONS)], ignore_index=True)


def fit_cost_models(train, num, cat, params, selection_campaign=None):
    """Choose XGB or the action mean on the final prior campaign, then refit.

    For July training on CP01-07, explicitly freeze selection_campaign='CP06'
    to keep CP07 outside model-family selection, as in notebook 08.
    """
    validation_campaign = selection_campaign or max(train.campaign_id)
    if validation_campaign not in set(train.campaign_id):
        raise ValueError("Cost selection campaign must be in the training period")
    result = {}
    for a in COMMERCIAL:
        rows = train[train.offer_type == a]
        inner_train=rows[rows.campaign_id < validation_campaign]
        inner_valid=rows[rows.campaign_id == validation_campaign]
        if len(inner_train) == 0 or len(inner_valid) == 0:
            raise ValueError("Cost selection requires earlier fitting rows and validation support")
        candidate=make_xgb_pipe(num,cat,params).fit(inner_train[num+cat],inner_train.promo_cost_14d)
        xgb_rmse=reg_metrics(inner_valid.promo_cost_14d,np.maximum(candidate.predict(inner_valid[num+cat]),0))["rmse"]
        mean_rmse=reg_metrics(inner_valid.promo_cost_14d,np.full(len(inner_valid),inner_train.promo_cost_14d.mean()))["rmse"]
        result[a] = (make_xgb_pipe(num, cat, params).fit(rows[num+cat], rows.promo_cost_14d)
                     if xgb_rmse < mean_rmse else ConstantPredictor(rows.promo_cost_14d.mean()))
    return result


def predict_costs(models, X):
    return np.column_stack([np.zeros(len(X))] + [np.maximum(models[a].predict(X), 0) for a in COMMERCIAL])


def tune_on_prior_campaigns(train, num, cat):
    """Nested temporal tuning: the final *prior* campaign is the inner validation set."""
    campaigns = sorted(train.campaign_id.unique())
    if len(campaigns) < 2:
        raise ValueError("At least two prior campaigns required for inner temporal tuning")
    inner_train = train[train.campaign_id < campaigns[-1]]
    inner_valid = train[train.campaign_id == campaigns[-1]]
    rows=[]
    for cfg in XGB_GRID:
        models = fit_action_models(inner_train,num,cat,cfg)
        mu = predict_actions(models,inner_valid[num+cat])
        factual = mu[np.arange(len(inner_valid)),[ACTIONS.index(a) for a in inner_valid.offer_type]]
        rows.append({**cfg,**reg_metrics(inner_valid.ncm_14d,factual)})
    scores=pd.DataFrame(rows)
    best=scores.sort_values(["rmse","mae","max_depth","n_estimators"]).iloc[0]
    params={"n_estimators":int(best.n_estimators),"max_depth":int(best.max_depth),"learning_rate":float(best.learning_rate)}
    return params, campaigns[-1]


def evaluate_fold(history, campaign, num, cat, params, draws=2000):
    train_all = history[history.campaign_id < campaign].copy()
    train = train_all[train_all.assignment_source == "exploration"]
    test = history[(history.campaign_id == campaign) & (history.assignment_source == "exploration")].sort_values("customer_id").copy()
    if not len(test) or not len(train) or train.send_ts.max()+pd.Timedelta(days=14) > test.send_ts.min():
        raise ValueError("Training outcomes must mature before the evaluation campaign")
    params, inner_validation = tune_on_prior_campaigns(train,num,cat)
    X = test[num+cat]
    original = fit_action_models(train, num, cat, params)
    reward_nuisance = predict_actions(original, X)
    cost_models = fit_cost_models(train, num, cat, params)
    costs = predict_costs(cost_models, X)
    purchase = make_propensity_pipe(num, cat).fit(train_all[num+cat], train_all.purchase_14d).predict_proba(X)[:,1]
    holdout = get_holdout(test.customer_id, seed=42)
    kwargs = dict(budget=JULY_BUDGET*len(test)/JULY_SIZE, holdout_ids=holdout)
    summaries, individuals, metrics, assignments = [], [], [], []
    modelsets = {}
    predictions = {}
    for method in CANDIDATES:
        models = fit_action_models(train, num, cat, params, method=method, original=original)
        modelsets[method] = models
        mu = predict_actions(models, X)
        predictions[method] = mu
        values = value_table(test, mu, costs, purchase)
        allocation = optimize_policy(values, **kwargs)
        summary, individual = evaluate_policy(test, allocation, reward_nuisance, costs, method, campaign, draws)
        summaries.append(summary); individuals.append(individual)
        assignments.append(allocation.assign(policy=method, evaluation_campaign=campaign))
        factual = mu[np.arange(len(test)), [ACTIONS.index(a) for a in test.offer_type]]
        metrics.append({"split":campaign, "model":method, "scope":"pooled", **reg_metrics(test.ncm_14d, factual)})
        for action in ACTIONS:
            mask = test.offer_type.eq(action).to_numpy()
            metrics.append({"split":campaign, "model":method, "scope":action, **reg_metrics(test.loc[mask,"ncm_14d"], factual[mask])})
    hist_mix = train_all.loc[(train_all.assignment_source=="business_rule") & train_all.offer_type.ne("NO_OFFER"),"offer_type"].value_counts(normalize=True).to_dict()
    reference_values = value_table(test, reward_nuisance, costs, purchase)
    for name, allocation in build_baseline_policies(reference_values, historical_mix=hist_mix, **kwargs).items():
        summary, individual = evaluate_policy(test, allocation, reward_nuisance, costs, name, campaign, draws)
        summaries.append(summary); individuals.append(individual)
        assignments.append(allocation.assign(policy=name, evaluation_campaign=campaign))
    individual = pd.concat(individuals, ignore_index=True)
    individual["unseen_customer"] = ~individual.customer_id.isin(train_all.customer_id)
    individual["cold_and_unseen_customer"] = individual.cold_history & individual.unseen_customer
    summary_table=pd.DataFrame(summaries)
    summary_table["inner_validation_campaign"]=inner_validation
    summary_table["fold_parameters"]=json.dumps(params,sort_keys=True)
    return {"summary":summary_table, "scores":individual, "metrics":pd.DataFrame(metrics),
            "allocations":pd.concat(assignments, ignore_index=True), "models":modelsets,
            "predictions":predictions, "test":test, "costs":costs, "cost_models":cost_models}


def select_candidate(development_scores, draws=2000):
    """Never accepts test data; three paired comparisons use Bonferroni intervals."""
    if not set(development_scores.campaign_id).issubset({"CP04","CP05","CP06"}):
        raise ValueError("Only CP04-CP06 development predictions may select a model")
    index = ["assignment_id","customer_id","campaign_id"]
    wide = development_scores.pivot(index=index, columns="policy", values="dr_incremental_score")
    rows=[]
    for method in CANDIDATES:
        delta = wide[method]-wide["ORIGINAL_TLEARNER"]
        if {"observed_action", "direct_component", "residual_component"}.issubset(development_scores):
            paired = _paired_dr_frame(development_scores, method, "ORIGINAL_TLEARNER")
            mean, low, high = cluster_dr_interval(paired, alpha=0.05/(len(CANDIDATES)-1), draws=draws)
        else:
            # Compatibility for synthetic score-only unit fixtures; production
            # evaluation always supplies decomposition and re-estimates p.
            mean, low, high = cluster_mean_interval(delta, wide.index.get_level_values("customer_id"),
                                                   alpha=0.05/(len(CANDIDATES)-1), draws=draws)
        rows.append({"candidate":method, "paired_dr_gain_vs_original":mean,
                     "adjusted_ci_low":low, "adjusted_ci_high":high,
                     "eligible_to_replace":method != "ORIGINAL_TLEARNER" and low > 0})
    comparisons=pd.DataFrame(rows)
    eligible=comparisons[comparisons.eligible_to_replace]
    selected="ORIGINAL_TLEARNER" if eligible.empty else eligible.sort_values(
        ["paired_dr_gain_vs_original","candidate"],ascending=[False,True]).iloc[0].candidate
    return selected, comparisons


def run_policy_validation(history, num, cat, params, tables, models_dir, draws=2000):
    tables, models_dir = Path(tables), Path(models_dir)
    dev=[]
    for campaign in ["CP04","CP05","CP06"]:
        print(f"Policy development fold {campaign}", flush=True)
        dev.append(evaluate_fold(history, campaign, num, cat, params, draws))
    devscores=pd.concat([r["scores"] for r in dev],ignore_index=True)
    selected, comparisons=select_candidate(devscores,draws)
    comparisons.to_csv(tables/"policy_candidate_selection.csv",index=False)
    pd.concat([r["summary"] for r in dev],ignore_index=True).to_csv(tables/"rolling_policy_validation.csv",index=False)
    pd.concat([r["metrics"] for r in dev],ignore_index=True).to_csv(tables/"rolling_model_metrics.csv",index=False)
    # Commit the development choice before any CP07 evaluation or outcomes are accessed below.
    selection={"selected_method":selected,"selection_campaigns":["CP04","CP05","CP06"],
               "test_campaign":"CP07","candidate_methods":CANDIDATES,"parameters":params,
               "rule":"retain original unless paired customer-bootstrap lower bound >0 after 3-comparison Bonferroni adjustment",
               "test_status":"retrospective held-out reanalysis; CP07 was inspected in the earlier audit",
               "interval_scope":"customer bootstrap re-estimates campaign logging probabilities; conditional on fitted outcome/cost models and fixed allocations; not future campaign drift",
               "logging_propensity_assumption":"unconditional randomized offers within campaign; empirical frequencies; uniform sensitivity also reported"}
    (models_dir/"policy_selection.json").write_text(json.dumps(selection,indent=2),encoding="utf-8")
    print(f"Selection locked: {selected}; now evaluating CP07",flush=True)
    test=evaluate_fold(history,"CP07",num,cat,params,draws)
    test["summary"]["selected_for_july"]=test["summary"].policy.eq(selected)
    test["summary"].to_csv(tables/"heldout_policy_comparison.csv",index=False)
    test["metrics"].to_csv(tables/"model_comparison_cp07.csv",index=False)
    test["allocations"].to_csv(tables/"heldout_policy_assignments.csv",index=False)
    test["scores"].to_csv(tables/"heldout_policy_dr_scores.csv",index=False)
    pairs=[]
    wide=test["scores"].pivot(index=["assignment_id","customer_id"],columns="policy",values="dr_incremental_score")
    for ref in wide.columns:
        paired = _paired_dr_frame(test["scores"], selected, ref)
        m,lo,hi=cluster_dr_interval(paired,draws=draws)
        pairs.append({"policy":selected,"reference":ref,"paired_dr_difference":m,"ci_low":lo,"ci_high":hi})
    pd.DataFrame(pairs).to_csv(tables/"heldout_policy_paired_comparisons.csv",index=False)
    subgroup=[]
    for group in ["cold_history","unseen_customer","cold_and_unseen_customer"]:
        for policy,s in test["scores"].groupby("policy"):
            included = s[group].to_numpy(bool)
            m,lo,hi = cluster_dr_interval(s, included=included, draws=draws) if included.any() else (np.nan,np.nan,np.nan)
            subgroup.append({"subgroup":group,"policy":policy,"n":int(included.sum()),
                             "matched_rows":int(s.loc[included,"matched"].sum()),
                             "dr_incremental_ncm_per_customer":m,"ci_low":lo,"ci_high":hi,
                             "logging_population":"entire evaluation campaign, not subgroup"})
    pd.DataFrame(subgroup).to_csv(tables/"heldout_cold_start_evaluation.csv",index=False)
    exp=history[history.assignment_source=="exploration"]
    final=fit_action_models(exp,num,cat,params,method=selected)
    for a,model in final.items():
        joblib.dump(model,models_dir/f"tlearner_ncm_{a}.joblib")
    return selection,test


def run_return_sensitivity(history, july, clean, num, cat, params, method, tables, draws=2000, policy_rule=None):
    """Full re-fit/allocation sensitivity, never a reason to change the primary target.

    Compare 14-day and 45-day accrued-return labels on identical mature cohorts.
    Retrospective sensitivity uses CP06, whose 45-day labels mature before July's
    decision cutoff. Because method selection also uses CP06, this is not a new
    untouched confirmatory test. CP07 is excluded from the mature July fit.
    """
    from src.diagnostics import build_accrued_target
    from src.policy import allocate_selected_policy
    from src.propensity import select_purchase_config, fit_purchase_model
    policy_rule = policy_rule or method
    tables=Path(tables)
    cutoff=pd.to_datetime(july.policy_as_of_ts).min()
    if pd.to_datetime(july.policy_as_of_ts).nunique() != 1:
        raise ValueError("Sensitivity currently requires one shared July decision cutoff")
    mature=build_accrued_target(history,clean["orders"],clean["returns"],horizon_days=45,as_of_ts=cutoff)
    accrued="ncm_accrued_45d"
    exp=mature[mature.assignment_source=="exploration"]
    final_costs=fit_cost_models(history[history.assignment_source=="exploration"],num,cat,params,selection_campaign="CP06")
    july_cost=predict_costs(final_costs,july[num+cat])
    purchase_config, _, _ = select_purchase_config(history,num,cat)
    purchase=fit_purchase_model(history,num,cat,purchase_config).predict_proba(july[num+cat])[:,1]
    juli=july.rename(columns={"policy_as_of_ts":"send_ts"})
    hold=get_holdout(july.customer_id)
    mix=history.loc[history.assignment_source.eq('business_rule') & history.offer_type.ne('NO_OFFER'),'offer_type'].value_counts(normalize=True).to_dict()
    july_policies={}; july_summary=[]
    for name,target in [("PRIMARY_14D_MATURE_COHORT","ncm_14d"),("ACCRUED_45D_MATURE_COHORT",accrued)]:
        models=fit_action_models(exp,num,cat,params,method=method,target=target)
        mu=predict_actions(models,july[num+cat])
        pol=allocate_selected_policy(value_table(juli,mu,july_cost,purchase),policy_rule,historical_mix=mix,budget=JULY_BUDGET,holdout_ids=hold)
        july_policies[name]=pol
        july_summary.append({"scenario":name,"target":target,"policy_rule":policy_rule,"training_rows":len(exp),
            "training_campaigns":",".join(sorted(exp.campaign_id.unique())),"label_as_of_ts":str(cutoff),
            "projected_incremental_ncm":float(pol.estimated_incremental_ncm.sum()),
            "projected_spend":float(pol.estimated_promo_cost.sum()),
            **{f"n_{a}":int(pol.recommended_action.eq(a).sum()) for a in ACTIONS}})
    base=july_policies["PRIMARY_14D_MATURE_COHORT"].set_index("customer_id").recommended_action
    alt=july_policies["ACCRUED_45D_MATURE_COHORT"].set_index("customer_id").recommended_action.reindex(base.index)
    for row in july_summary:
        row["actions_changed_between_targets"]=int(base.ne(alt).sum())
        row["interpretation"]="scenario projections under different targets, not evidence of causal improvement"
    pd.DataFrame(july_summary).to_csv(tables/"return_sensitivity_july.csv",index=False)
    pd.concat([p.assign(scenario=name) for name,p in july_policies.items()],ignore_index=True).to_csv(tables/"return_sensitivity_assignments.csv",index=False)
    # Earlier campaign sensitivity is evaluated only when its 45-day outcomes are mature.
    campaign="CP06"
    asof=history.loc[history.campaign_id.eq(campaign),"send_ts"].min()
    training=build_accrued_target(history,clean["orders"],clean["returns"],horizon_days=45,as_of_ts=asof)
    tr=training[training.assignment_source.eq("exploration")]
    te=mature[mature.campaign_id.eq(campaign) & mature.assignment_source.eq("exploration")].sort_values("customer_id")
    cfg, _=tune_on_prior_campaigns(tr,num,cat)
    costs=predict_costs(fit_cost_models(tr,num,cat,cfg),te[num+cat])
    purchase_config, _, _ = select_purchase_config(training,num,cat)
    purchase=fit_purchase_model(training,num,cat,purchase_config).predict_proba(te[num+cat])[:,1]
    mix=training.loc[training.assignment_source.eq('business_rule') & training.offer_type.ne('NO_OFFER'),'offer_type'].value_counts(normalize=True).to_dict()
    allocation={}; nuisance={}
    for label,target in [("PRIMARY_14D","ncm_14d"),("ACCRUED_45D",accrued)]:
        fitted=fit_action_models(tr,num,cat,cfg,method=method,target=target)
        mu=predict_actions(fitted,te[num+cat])
        allocation[label]=allocate_selected_policy(value_table(te,mu,costs,purchase),policy_rule,historical_mix=mix,budget=JULY_BUDGET*len(te)/JULY_SIZE,
                                           holdout_ids=get_holdout(te.customer_id))
        nuisance[label]=predict_actions(fit_action_models(tr,num,cat,cfg,target=target),te[num+cat])
    rows=[]
    for evaluation_label,target in [("PRIMARY_14D","ncm_14d"),("ACCRUED_45D",accrued)]:
        outcomes=te.assign(ncm_14d=te[target])
        for name,pol in allocation.items():
            summary,_=evaluate_policy(outcomes,pol,nuisance[evaluation_label],costs,name,campaign,draws)
            summary.update({"evaluation_target":target,"training_rows":len(tr),
                            "training_campaigns":",".join(sorted(tr.campaign_id.unique())),"label_as_of_ts":str(cutoff),
                            "evaluation_status":"retrospective sensitivity; CP06 also used for method selection"})
            rows.append(summary)
    pd.DataFrame(rows).to_csv(tables/"return_sensitivity_cp06.csv",index=False)
    return pd.DataFrame(july_summary)
