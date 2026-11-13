"""Build the validation report from the current completed pipeline artifacts."""
import json
from pathlib import Path
import pandas as pd

ROOT=Path(__file__).resolve().parents[1]
T=ROOT/'outputs/tables'
E=ROOT/'outputs/experiments'


def money(value):
    return ('-' if value<0 else '')+f'${abs(value):.2f}'


def table(frame):
    def fmt(value):
        if isinstance(value,float):return f'{value:.4f}' if pd.notna(value) else 'NA'
        return str(value).replace('|','/')
    lines=['| '+' | '.join(frame.columns)+' |','| '+' | '.join(['---']*len(frame.columns))+' |']
    lines += ['| '+' | '.join(fmt(v) for v in row)+' |' for row in frame.itertuples(index=False,name=None)]
    return '\n'.join(lines)


def main():
    selection=json.loads((ROOT/'outputs/models/policy_selection.json').read_text())
    selected=selection['selected_method']
    profit_selection=json.loads((E/'profit/selection.json').read_text())
    policy_search=json.loads((E/'policy/selection.json').read_text())
    stochastic=pd.read_csv(E/'policy/stochastic_free_ship_robustness.csv').query("split == 'CP07'")
    stochastic_gain=stochastic.set_index('comparison').loc['versus NO_OFFER']
    stochastic_change=stochastic.set_index('comparison').loc['versus ORIGINAL_TLEARNER']
    if selected!=policy_search['selected_policy']:
        raise ValueError('Rebuild pipeline outputs before reporting the new policy selection')
    champion=profit_selection['selected_model']
    compare=pd.read_csv(T/'before_after_model_metrics.csv')
    old=compare.set_index('version').loc['original_refit_current_environment']
    new=compare.set_index('version').loc['corrected_predictive_champion']
    if new['model']!=champion:raise ValueError('Reference comparison has a stale predictive champion')
    xgb_ci=pd.read_csv(E/'profit/cp07_paired_loss.csv').set_index('loss')
    logistic=pd.read_csv(E/'propensity/cp07_comparison.csv')
    log_ci=pd.read_csv(E/'propensity/cp07_paired_bootstrap.csv').set_index('metric')
    development=pd.read_csv(E/'policy/development_ranking.csv')
    policies=pd.read_csv(T/'heldout_policy_comparison.csv')
    policy=policies.set_index('policy').loc[selected]
    original_policy=policies.set_index('policy').loc['ORIGINAL_TLEARNER']
    paired=pd.read_csv(T/'heldout_policy_paired_comparisons.csv').rename(columns={'gain':'paired_dr_difference'})
    change=paired.query("reference == 'ORIGINAL_TLEARNER'").iloc[0]
    final=pd.read_csv(T/'optimization_summary.csv').iloc[0]
    cold=pd.read_csv(T/'heldout_cold_start_evaluation.csv').query('policy == @selected')
    returns=pd.read_csv(T/'return_sensitivity_july.csv')
    costs=pd.read_csv(T/'promo_cost_metrics.csv').query("split == 'test_CP07'")
    data_check=json.loads((T/'before_after_data_check.json').read_text())
    behavior_change=json.loads((T/'performance_revision_changes.json').read_text())
    coverage=json.loads((T/'transaction_coverage_summary.json').read_text())
    coverage_change=json.loads((T/'coverage_revision_comparison.json').read_text())
    environment=json.loads((ROOT/'outputs/validation_environment.json').read_text())
    test_evidence=(f"The recorded verification run passed **{environment['focused_tests_passed']} focused tests**."
                   if 'focused_tests_passed' in environment else 'The completed test run is recorded with the revision.')
    cold_text=(table(cold[['subgroup','n','matched_rows','dr_incremental_ncm_per_customer','ci_low','ci_high']])
               if len(cold) else 'No selected-policy subgroup estimate was saved; July transport remains unvalidated.')
    text=f'''# Correction and performance validation - November 13, 2026

**The prior performance revision improved prediction modestly; the current coverage correction prevents unsupported inactivity claims.** Policy superiority remains uncertain. Earlier corrections fixed accounting, timing, constraints, and misleading evaluation while retaining the original model. The subsequent performance revision changed the fitted models and selected allocation independently on development campaigns. The current correction preserves those numeric predictor values and properly labels their observation scope.

## Transaction coverage correction

The decision cutoff remains **{coverage['policy_as_of_ts']}**. The actual July 30-day window starts **{coverage['window_30d_start']}** and the 90-day window starts **{coverage['window_90d_start']}**, with an inclusive decision endpoint. The old date arithmetic was correct; shifting the recent window back to January would answer a different question.

The supplied ordinary-order records end at an observed event on **{coverage['source_latest_observed_organic_order_ts']}**. Later supplied orders are campaign-attributed; July customers have no historical campaign assignments. All **{coverage['july_customers']:,}** July customers therefore have zero recent completed orders **in this extract**. No export contract or coverage watermark certifies that the extract contains every purchase. Thus `observed_orders_90d = 0` is accurate, while `complete_orders_90d` is **unknown for {coverage['july_complete_90d_counts_unknown']:,} customers**. Verified inactive customers: **{coverage['july_verified_inactive_customers']}**. The same coverage limitation applies to historical windows.

The code uses 19 explicit `observed_` predictor names, preserves all 43 predictor values/order, and exports per-decision requested windows, source counts, completeness status, and nullable complete-history counts. `config/transaction_coverage.json` contains no invented certified intervals. Complete counts become known only when explicit all-source coverage spans the entire requested window and was available by the decision time. Metadata is audit context, excluded from model predictors. Unknown counts never qualify as verified inactivity.

Separately, **{coverage['july_customers_with_any_observed_completed_order']}** July customers have older observed completed purchases. The fixed organic reference interval **[{coverage['reference_window_start']}, {coverage['reference_window_end_exclusive']})** contains **{coverage['reference_observed_completed_orders']}** completed orders from **{coverage['reference_customers_with_observed_completed_order']}** July customers. This reference is about **{coverage['reference_age_days_at_july_cutoff']:.1f} days stale** at the decision cutoff. It is an observed-only audit, not a replacement recent window, a completeness claim, or a new predictor.

Numerical comparison against the preceding performance commit `{coverage_change['comparison_commit'][:7]}`:

{table(pd.DataFrame(coverage_change['output_checks'])[['file','rows','changed_rows','max_absolute_numeric_change']])}

This semantic correction adds **no measured model-performance gain** when predictions are unchanged. Its benefit is a valid distinction between observed zero and unknown full history. The existing dataset cannot resolve whether missing ordinary purchases reflect the supplied extraction design or actual inactivity. Historical accuracy and July projections remain conditional on this limited source coverage. See `transaction_coverage_summary.json`, `coverage_revision_comparison.json`, and the historical/July coverage and reference CSVs.

## Prior model-performance gains retained

- Expected-NCM prediction: CP07 RMSE falls from **{old.rmse:.4f} to {new.rmse:.4f}** ({100*(old.rmse-new.rmse)/old.rmse:.2f}%), and MAE from **{old.mae:.4f} to {new.mae:.4f}** ({100*(old.mae-new.mae)/old.mae:.2f}%).
- Logistic regression: small improvements in average precision and probability quality; paired uncertainty includes zero improvement.
- Feasible allocation: **{selected}** wins CP04-CP06 development selection. Averaging feasible random recipient sets, CP07 DR incremental NCM is **${stochastic_gain.gain:.2f} per customer** versus no offer; the paired difference versus the original feasible policy is **${stochastic_change.gain:.2f}**, 95% interval **[{money(stochastic_change.ci_low)}, {money(stochastic_change.ci_high)}]**. The interval includes zero.

CP07 was already inspected in earlier project work. These are **retrospective comparisons**, not a new untouched confirmatory test. They are not realized July gains. The policy result favors a simple offer rule; it does not demonstrate a benefit from personalization.

## Fair predictive comparison

The reference is commit `ea950d0e9011ff0e105f5d93758ab54944a7c75f`. The original model is refitted in the current environment and compared on identical CP07 assignments, observed actions, and targets. Saved original metrics are retained for provenance, not used to attribute runtime differences to this revision.

{table(compare)}

Predictive champion: **{champion}**. Allocation policy: **{selected}**. These are different selections: prediction accuracy does not establish good treatment-effect ranking. Primary 14-day NCM unchanged: **{data_check['primary_ncm_unchanged']}**. Feature columns with changed numerical/category values: **{len(data_check['feature_differences'])}**; separately renamed observation-scope fields: **{len(data_check.get('feature_scope_renames', {}))}**. Factual CP07 predictions changed for **{data_check['factual_predictions_changed_rows']}** assignments. Unchanged source data do not imply unchanged fitted predictions.

In the prior model-performance revision, compared with reliability revision `{behavior_change['comparison_commit'][:7]}`, **{behavior_change['july_actions_changed']:,} of {behavior_change['july_customers']:,} July actions changed**. That is a concrete behavior change, not proof of higher realized profit. The comparison and input fingerprint are saved in `performance_revision_changes.json`.

## XGBoost: why the change matters

The original fits four independent regressors. About 88% of randomized training rows are nonbuyers with zero NCM, and some purchaser margins are negative. The selected two-part model shares information across action arms and estimates purchase probability times signed purchaser margin. The margin forecast blends pooled XGBoost with each training arm's purchaser mean, using 50% weight on each. It neither logs the target nor clips negative profit. Future purchase is a training label, never a scoring feature.

Fourteen fixed configurations compare original, pooled, shared-baseline, and two-part estimators. Rolling CP04-CP06 validation fits earlier campaigns only; minimum pooled factual RMSE selects the winner. The configuration is serialized before CP07 evaluation. The code and full search results are in `src/profit_models.py` and `outputs/experiments/profit/`.

On CP07, old-minus-new MSE is **{xgb_ci.loc['mse','old_minus_new']:.2f}**, paired customer-bootstrap 95% interval **[{xgb_ci.loc['mse','ci95_lower']:.2f}, {xgb_ci.loc['mse','ci95_upper']:.2f}]**. MAE improvement is **{xgb_ci.loc['mae','old_minus_new']:.3f}**, interval **[{xgb_ci.loc['mae','ci95_lower']:.3f}, {xgb_ci.loc['mae','ci95_upper']:.3f}]**. Five thousand draws hold fits fixed and omit selection, retraining, and future-campaign uncertainty. Three arm-specific RMSEs improve, while NO_OFFER worsens slightly. R² remains low, so realized individual profit remains difficult to predict.

RMSE is appropriate for expected dollar NCM and squared-error fitting. MSE adds no independent ranking information because RMSE is its square root; on identical observations R² gives the same ranking. MAE describes absolute dollar errors, but minimizing it instead targets a conditional median, often zero in this sparse outcome. None of these metrics proves causal uplift or policy profit.

## Logistic regression: improvement is small

Rolling CP04-CP06 development compares transformations, compact interactions, regularization, and nested temporal probability calibration. Selection maximizes mean average precision subject to log-loss/Brier safeguards. The winner uses signed-log transforms and compact interactions, C = 0.01, and no additional calibration. Raw and improved models use the same training/test populations and current runtime.

{table(logistic[['model','roc_auc','pr_auc','log_loss','brier','observed_mean','predicted_mean']])}

The `pr_auc` column is **average precision**, not trapezoidal PR area. Its paired CP07 change is **{log_ci.loc['pr_auc','difference_selected_minus_original']:.6f}**, with 95% interval **[{log_ci.loc['pr_auc','ci_low']:.6f}, {log_ci.loc['pr_auc','ci_high']:.6f}]**. All four paired metric intervals include zero improvement. Retaining the development winner is reasonable, but a major reliable predictive advance is not established. Purchase propensity describes buying under the historical offer mixture; it is not untreated purchase probability or causal uplift.

## Select the complete feasible policy separately

Development compares {policy_search['candidate_count']} allocation rules, including model optimizers, pooled/segment shrinkage rules, and simple baselines. Every fold uses earlier campaigns for reward and cost fitting, with the same budget, offer cap, PCT_20 cap, and reserved controls. The winner maximizes pooled customer-weighted development DR value. A rule need not clear a multiple-comparison-adjusted confidence threshold to replace the original; uncertainty is reported separately.

{table(development.head(8))}

Development ranking is selection-biased. The frozen winner is evaluated on CP07 with common reward nuisance predictions for all policies. CP07 uses 1,060 randomized assignments and the population-scaled budget. The former unconstrained +$8.73 diagnostic offered promotions to 87.2% of customers and is not the baseline for this comparison.

{table(policies[['policy','dr_incremental_ncm_per_customer','ci_low','ci_high','hajek_incremental_ncm_per_customer','offered_share']])}

{table(paired[['reference','paired_dr_difference','ci_low','ci_high']])}

The saved seed-42 allocation has DR gain **${policy.dr_incremental_ncm_per_customer:.2f}**, interval **[${policy.ci_low:.2f}, ${policy.ci_high:.2f}]**, and paired difference versus the original **${change.paired_dr_difference:.2f}**, interval **[${change.ci_low:.2f}, ${change.ci_high:.2f}]**. That fixed recipient set is unusually favorable. A diagnostic averages feasible random recipient sets with the same controls:

{table(stochastic[['comparison','gain','ci_low','ci_high']])}

The averaged paired interval includes zero: **a clear policy improvement over the original is not established across recipient randomizations**. The seed was fixed before testing, but that does not make its favorable recipient realization representative. This diagnostic does not retune selection. The simple rule wins development; personalization has not shown a dependable advantage. A new randomized campaign is needed to assess future performance.

Actual logging probabilities were not supplied. Primary estimates assume unconditional randomized assignment within campaign and use empirical action frequencies; uniform-0.25 sensitivity is also saved. Customer-cluster bootstrap draws retain repeated-customer observations and re-estimate logging frequencies. Policy intervals use 2,000 draws, condition on fitted models and fixed allocations, and omit future campaign drift and full retraining uncertainty. Nominal test intervals are not adjusted for repeated earlier inspection of CP07.

## July support, accounting, and cost remain material limits

July has no historical campaign exposure, no observed recent orders in the supplied extract, and seven entirely missing predictor columns. Complete recent purchase history is unknown. Imputation permits scoring; it does not validate transport. Selected-policy subgroup evidence uses observed-history labels, not verified inactivity:

{cold_text}

Subgroups are small and uncertain. Profile fields without effective-date history still require the explicit static-snapshot assumption.

Earlier fixes remain meaningful: eligible events precede decisions; duplicate refund events no longer duplicate purchase features; early/late/total refunds reconcile; the allocator enforces the actual constraints; and prospective measurement compares the full randomized POLICY arm with controls. These improve validity even when prediction accuracy is unchanged.

The 45-day accrued-return sensitivity uses identical mature training cohorts and excludes CP07 from July training. It retains the selected allocation rule and re-estimates values under both financial targets. It is an accounting stress test, not a verified new business rule or fresh test.

{table(returns[['scenario','training_rows','projected_incremental_ncm','projected_spend','actions_changed_between_targets']])}

The random allocation does not change between targets. Its mature-cohort projected increment changes sign under accrued returns: that is a material limitation, not an improvement. Different-target projections cannot establish that one model is better, and differ from the primary July projection because this sensitivity excludes CP07 training.

Cost models are selected by action on prior-campaign RMSE before CP07 testing. Mean bias matters for aggregate spend even when individual RMSE improves.

{table(costs[['action','candidate','rmse','mae','obs_mean','pred_mean','relative_mean_bias']])}

The July submission contains {int(final.n_FREE_SHIP)} FREE_SHIP, {int(final.n_PCT_10)} PCT_10, {int(final.n_PCT_20)} PCT_20, and {int(final.n_NO_OFFER)} NO_OFFER customers. Its {int(final.holdout_customers)} reserved controls remain NO_OFFER. Forecast incremental NCM is **${final.objective_total_expected_incremental_ncm:,.2f}**, and forecast promotion spend is **${final.expected_promo_spend:,.2f}**. These use the improved predictor to describe the selected allocation; they are not observed profit. Comparing this forecast with an old optimizer forecast under different models is not evidence of better or worse realized profit. Estimated-cost feasibility does not guarantee realized spending.

## Reproduction and prospective measurement

Raw CSVs are intentionally absent. A clean-data rebuild validates transformations after cleaning, not the unavailable raw inputs. The original reference commit must be available locally for the fair comparison. Core versions are pinned in `requirements.txt`; `outputs/validation_environment.json` records the run environment.

Policy-validation campaigns execute in separate worker processes with deadlines and cleanup, isolating native learner state across folds. A long-lived-process rerun encountered a native XGBoost failure while standalone evaluation succeeded. Process isolation is a practical safeguard; it does not establish the underlying library bug or its cause. Preserve recorded seeds and native-thread settings when comparing numerical results.

```text
python -m pytest -q
python run_pipeline.py --from-clean
python scripts/compare_reference.py
python scripts/validate_coverage_revision.py
python scripts/build_validation_report.py
python docs/build_walkthrough_pdf.py
```

{test_evidence} Focused checks cover signed-margin factorization, exploration-only fitting, serialization, assignment alignment, feature timing, observation-scope parity, explicit coverage contracts/unknown zeros, refund accounting/maturity, DR/bootstrap logic, feasible allocation, and ITT analysis. Recorded execution scope: {environment['execution']} The commands above reproduce the checks; generating this report itself does not rerun them. The walkthrough PDF explains prediction and allocation separately.

Before launch, predefine the financial horizon, randomization, power, monitoring, and analysis. Compare **all** customers randomized to the POLICY arm, including those given NO_OFFER, with the randomized control arm using `src.measurement.estimate_policy_itt`. A 5% minimum control share satisfies the project constraint but does not guarantee adequate power.
'''
    path=ROOT/'docs/VALIDATION_REPORT.md'
    path.write_text(text,encoding='utf-8')
    print(path)


if __name__=='__main__':main()
