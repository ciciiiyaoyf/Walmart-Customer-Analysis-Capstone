# Correction and performance validation - November 13, 2026

**The prior performance revision improved prediction modestly; the current coverage correction prevents unsupported inactivity claims.** Policy superiority remains uncertain. Earlier corrections fixed accounting, timing, constraints, and misleading evaluation while retaining the original model. The subsequent performance revision changed the fitted models and selected allocation independently on development campaigns. The current correction preserves those numeric predictor values and properly labels their observation scope.

## Transaction coverage correction

The decision cutoff remains **2025-06-20 10:00:00**. The actual July 30-day window starts **2025-05-21 10:00:00** and the 90-day window starts **2025-03-22 10:00:00**, with an inclusive decision endpoint. The old date arithmetic was correct; shifting the recent window back to January would answer a different question.

The supplied ordinary-order records end at an observed event on **2025-01-09 21:42:00**. Later supplied orders are campaign-attributed; July customers have no historical campaign assignments. All **2,776** July customers therefore have zero recent completed orders **in this extract**. No export contract or coverage watermark certifies that the extract contains every purchase. Thus `observed_orders_90d = 0` is accurate, while `complete_orders_90d` is **unknown for 2,776 customers**. Verified inactive customers: **0**. The same coverage limitation applies to historical windows.

The code uses 19 explicit `observed_` predictor names, preserves all 43 predictor values/order, and exports per-decision requested windows, source counts, completeness status, and nullable complete-history counts. `config/transaction_coverage.json` contains no invented certified intervals. Complete counts become known only when explicit all-source coverage spans the entire requested window and was available by the decision time. Metadata is audit context, excluded from model predictors. Unknown counts never qualify as verified inactivity.

Separately, **510** July customers have older observed completed purchases. The fixed organic reference interval **[2024-10-12, 2025-01-10)** contains **279** completed orders from **256** July customers. This reference is about **161.4 days stale** at the decision cutoff. It is an observed-only audit, not a replacement recent window, a completeness claim, or a new predictor.

Numerical comparison against the preceding performance commit `54cf2ff`:

| file | rows | changed_rows | max_absolute_numeric_change |
| --- | --- | --- | --- |
| outputs/tables/logistic_test_predictions.csv | 3800 | 0 | 0.0000 |
| outputs/tables/xgboost_test_predictions.csv | 1060 | 0 | 0.0000 |
| outputs/tables/july_xgboost_ncm_scores.csv | 2776 | 0 | 0.0000 |
| outputs/tables/july_action_ncm_scores.csv | 2776 | 0 | 0.0000 |
| policy_submission.csv | 2776 | 0 | 0.0000 |

This semantic correction adds **no measured model-performance gain** when predictions are unchanged. Its benefit is a valid distinction between observed zero and unknown full history. The existing dataset cannot resolve whether missing ordinary purchases reflect the supplied extraction design or actual inactivity. Historical accuracy and July projections remain conditional on this limited source coverage. See `transaction_coverage_summary.json`, `coverage_revision_comparison.json`, and the historical/July coverage and reference CSVs.

## Prior model-performance gains retained

- Expected-NCM prediction: CP07 RMSE falls from **63.3694 to 62.7385** (1.00%), and MAE from **27.4111 to 26.4806** (3.39%).
- Logistic regression: small improvements in average precision and probability quality; paired uncertainty includes zero improvement.
- Feasible allocation: **RANDOM_FREE_SHIP** wins CP04-CP06 development selection. Averaging feasible random recipient sets, CP07 DR incremental NCM is **$7.90 per customer** versus no offer; the paired difference versus the original feasible policy is **$3.93**, 95% interval **[-$2.12, $10.44]**. The interval includes zero.

CP07 was already inspected in earlier project work. These are **retrospective comparisons**, not a new untouched confirmatory test. They are not realized July gains. The policy result favors a simple offer rule; it does not demonstrate a benefit from personalization.

## Fair predictive comparison

The reference is commit `ea950d0e9011ff0e105f5d93758ab54944a7c75f`. The original model is refitted in the current environment and compared on identical CP07 assignments, observed actions, and targets. Saved original metrics are retained for provenance, not used to attribute runtime differences to this revision.

| version | role | model | rmse | mae | r2 |
| --- | --- | --- | --- | --- | --- |
| original_saved_snapshot | profit_prediction | ORIGINAL_TLEARNER | 63.3774 | 27.4112 | 0.0056 |
| original_refit_current_environment | profit_prediction | ORIGINAL_TLEARNER | 63.3694 | 27.4111 | 0.0059 |
| corrected_predictive_champion | profit_prediction | HURDLE_XGB_HALF | 62.7385 | 26.4806 | 0.0256 |

Predictive champion: **HURDLE_XGB_HALF**. Allocation policy: **RANDOM_FREE_SHIP**. These are different selections: prediction accuracy does not establish good treatment-effect ranking. Primary 14-day NCM unchanged: **True**. Feature columns with changed numerical/category values: **0**; separately renamed observation-scope fields: **19**. Factual CP07 predictions changed for **1060** assignments. Unchanged source data do not imply unchanged fitted predictions.

In the prior model-performance revision, compared with reliability revision `1221dd1`, **1,403 of 2,776 July actions changed**. That is a concrete behavior change, not proof of higher realized profit. The comparison and input fingerprint are saved in `performance_revision_changes.json`.

## XGBoost: why the change matters

The original fits four independent regressors. About 88% of randomized training rows are nonbuyers with zero NCM, and some purchaser margins are negative. The selected two-part model shares information across action arms and estimates purchase probability times signed purchaser margin. The margin forecast blends pooled XGBoost with each training arm's purchaser mean, using 50% weight on each. It neither logs the target nor clips negative profit. Future purchase is a training label, never a scoring feature.

Fourteen fixed configurations compare original, pooled, shared-baseline, and two-part estimators. Rolling CP04-CP06 validation fits earlier campaigns only; minimum pooled factual RMSE selects the winner. The configuration is serialized before CP07 evaluation. The code and full search results are in `src/profit_models.py` and `outputs/experiments/profit/`.

On CP07, old-minus-new MSE is **79.57**, paired customer-bootstrap 95% interval **[17.45, 140.35]**. MAE improvement is **0.931**, interval **[0.452, 1.423]**. Five thousand draws hold fits fixed and omit selection, retraining, and future-campaign uncertainty. Three arm-specific RMSEs improve, while NO_OFFER worsens slightly. R² remains low, so realized individual profit remains difficult to predict.

RMSE is appropriate for expected dollar NCM and squared-error fitting. MSE adds no independent ranking information because RMSE is its square root; on identical observations R² gives the same ranking. MAE describes absolute dollar errors, but minimizing it instead targets a conditional median, often zero in this sparse outcome. None of these metrics proves causal uplift or policy profit.

## Logistic regression: improvement is small

Rolling CP04-CP06 development compares transformations, compact interactions, regularization, and nested temporal probability calibration. Selection maximizes mean average precision subject to log-loss/Brier safeguards. The winner uses signed-log transforms and compact interactions, C = 0.01, and no additional calibration. Raw and improved models use the same training/test populations and current runtime.

| model | roc_auc | pr_auc | log_loss | brier | observed_mean | predicted_mean |
| --- | --- | --- | --- | --- | --- | --- |
| original | 0.6587 | 0.2323 | 0.3711 | 0.1097 | 0.1318 | 0.1289 |
| selected | 0.6589 | 0.2367 | 0.3707 | 0.1096 | 0.1318 | 0.1306 |

The `pr_auc` column is **average precision**, not trapezoidal PR area. Its paired CP07 change is **0.004384**, with 95% interval **[-0.004445, 0.013209]**. All four paired metric intervals include zero improvement. Retaining the development winner is reasonable, but a major reliable predictive advance is not established. Purchase propensity describes buying under the historical offer mixture; it is not untreated purchase probability or causal uplift.

## Select the complete feasible policy separately

Development compares 21 allocation rules, including model optimizers, pooled/segment shrinkage rules, and simple baselines. Every fold uses earlier campaigns for reward and cost fitting, with the same budget, offer cap, PCT_20 cap, and reserved controls. The winner maximizes pooled customer-weighted development DR value. A rule need not clear a multiple-comparison-adjusted confidence threshold to replace the original; uncertainty is reported separately.

| policy | n | development_dr_incremental_ncm |
| --- | --- | --- |
| RANDOM_FREE_SHIP | 3237 | 3.8176 |
| SEGMENT_MEMBERSHIP_HISTORY_RIDGE_100 | 3237 | 3.7374 |
| SEGMENT_MEMBERSHIP_HISTORY_RIDGE_20 | 3237 | 3.7374 |
| HISTORICAL_MIX | 3237 | 3.3395 |
| ACTION_MEAN | 3237 | 2.7685 |
| SEGMENT_MEMBERSHIP_HISTORY_RIDGE_300 | 3237 | 2.7338 |
| SEGMENT_MEMBERSHIP_CART_RIDGE_100 | 3237 | 2.4945 |
| SEGMENT_MEMBERSHIP_RIDGE_100 | 3237 | 2.0515 |

Development ranking is selection-biased. The frozen winner is evaluated on CP07 with common reward nuisance predictions for all policies. CP07 uses 1,060 randomized assignments and the population-scaled budget. The former unconstrained +$8.73 diagnostic offered promotions to 87.2% of customers and is not the baseline for this comparison.

| policy | dr_incremental_ncm_per_customer | ci_low | ci_high | hajek_incremental_ncm_per_customer | offered_share |
| --- | --- | --- | --- | --- | --- |
| ORIGINAL_TLEARNER | 3.9714 | -2.2332 | 9.6825 | 3.0755 | 0.6000 |
| ACTION_MEAN | 5.5713 | -2.0978 | 13.6234 | 5.9460 | 0.6000 |
| HURDLE_XGB_HALF | 8.8888 | 0.2528 | 18.0299 | 8.9169 | 0.6000 |
| ALL_NO_OFFER | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| PROPENSITY_PCT10 | 3.4417 | -2.6335 | 9.6202 | 2.9177 | 0.6000 |
| HISTORICAL_MIX | 4.6599 | -0.4839 | 9.7587 | 3.0303 | 0.6000 |
| RANDOM_FREE_SHIP | 12.7414 | 3.5899 | 22.6637 | 12.2317 | 0.6000 |

| reference | paired_dr_difference | ci_low | ci_high |
| --- | --- | --- | --- |
| ORIGINAL_TLEARNER | 8.7700 | 0.3639 | 18.3279 |
| ACTION_MEAN | 7.1701 | -0.4220 | 15.9273 |
| HURDLE_XGB_HALF | 3.8527 | -4.1208 | 12.5708 |
| ALL_NO_OFFER | 12.7414 | 3.5899 | 22.6637 |
| PROPENSITY_PCT10 | 9.2997 | -0.9828 | 20.3793 |
| HISTORICAL_MIX | 8.0816 | -0.5327 | 18.3042 |
| RANDOM_FREE_SHIP | 0.0000 | 0.0000 | 0.0000 |

The saved seed-42 allocation has DR gain **$12.74**, interval **[$3.59, $22.66]**, and paired difference versus the original **$8.77**, interval **[$0.36, $18.33]**. That fixed recipient set is unusually favorable. A diagnostic averages feasible random recipient sets with the same controls:

| comparison | gain | ci_low | ci_high |
| --- | --- | --- | --- |
| versus NO_OFFER | 7.9018 | 1.3563 | 15.3476 |
| versus ORIGINAL_TLEARNER | 3.9304 | -2.1150 | 10.4401 |
| versus RANDOM_FREE_SHIP | -4.8396 | -9.7655 | -0.0505 |

The averaged paired interval includes zero: **a clear policy improvement over the original is not established across recipient randomizations**. The seed was fixed before testing, but that does not make its favorable recipient realization representative. This diagnostic does not retune selection. The simple rule wins development; personalization has not shown a dependable advantage. A new randomized campaign is needed to assess future performance.

Actual logging probabilities were not supplied. Primary estimates assume unconditional randomized assignment within campaign and use empirical action frequencies; uniform-0.25 sensitivity is also saved. Customer-cluster bootstrap draws retain repeated-customer observations and re-estimate logging frequencies. Policy intervals use 2,000 draws, condition on fitted models and fixed allocations, and omit future campaign drift and full retraining uncertainty. Nominal test intervals are not adjusted for repeated earlier inspection of CP07.

## July support, accounting, and cost remain material limits

July has no historical campaign exposure, no observed recent orders in the supplied extract, and seven entirely missing predictor columns. Complete recent purchase history is unknown. Imputation permits scoring; it does not validate transport. Selected-policy subgroup evidence uses observed-history labels, not verified inactivity:

| subgroup | n | matched_rows | dr_incremental_ncm_per_customer | ci_low | ci_high |
| --- | --- | --- | --- | --- | --- |
| first_exposure_no_observed_recent_orders | 93 | 23 | 13.4072 | -23.1388 | 69.1737 |
| unseen_customer | 83 | 22 | 15.3732 | -25.5895 | 78.6668 |
| unseen_no_observed_recent_orders | 83 | 22 | 15.3732 | -25.5895 | 78.6668 |

Subgroups are small and uncertain. Profile fields without effective-date history still require the explicit static-snapshot assumption.

Earlier fixes remain meaningful: eligible events precede decisions; duplicate refund events no longer duplicate purchase features; early/late/total refunds reconcile; the allocator enforces the actual constraints; and prospective measurement compares the full randomized POLICY arm with controls. These improve validity even when prediction accuracy is unchanged.

The 45-day accrued-return sensitivity uses identical mature training cohorts and excludes CP07 from July training. It retains the selected allocation rule and re-estimates values under both financial targets. It is an accounting stress test, not a verified new business rule or fresh test.

| scenario | training_rows | projected_incremental_ncm | projected_spend | actions_changed_between_targets |
| --- | --- | --- | --- | --- |
| PRIMARY_14D_MATURE_COHORT | 6467 | 124.2953 | 167.4434 | 0 |
| ACCRUED_45D_MATURE_COHORT | 6467 | -569.0519 | 167.4434 | 0 |

The random allocation does not change between targets. Its mature-cohort projected increment changes sign under accrued returns: that is a material limitation, not an improvement. Different-target projections cannot establish that one model is better, and differ from the primary July projection because this sensitivity excludes CP07 training.

Cost models are selected by action on prior-campaign RMSE before CP07 testing. Mean bias matters for aggregate spend even when individual RMSE improves.

| action | candidate | rmse | mae | obs_mean | pred_mean | relative_mean_bias |
| --- | --- | --- | --- | --- | --- | --- |
| PCT_10 | xgb | 20.8975 | 8.2918 | 4.9054 | 4.5943 | -0.0634 |
| PCT_20 | xgb | 31.2833 | 16.4322 | 10.8787 | 8.5411 | -0.2149 |
| FREE_SHIP | action_mean | 0.8903 | 0.2258 | 0.1351 | 0.0950 | -0.2972 |

The July submission contains 1665 FREE_SHIP, 0 PCT_10, 0 PCT_20, and 1111 NO_OFFER customers. Its 139 reserved controls remain NO_OFFER. Forecast incremental NCM is **$1,999.37**, and forecast promotion spend is **$167.44**. These use the improved predictor to describe the selected allocation; they are not observed profit. Comparing this forecast with an old optimizer forecast under different models is not evidence of better or worse realized profit. Estimated-cost feasibility does not guarantee realized spending.

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

The recorded verification run passed **106 focused tests**. Focused checks cover signed-margin factorization, exploration-only fitting, serialization, assignment alignment, feature timing, observation-scope parity, explicit coverage contracts/unknown zeros, refund accounting/maturity, DR/bootstrap logic, feasible allocation, and ITT analysis. Recorded execution scope: Coverage revision executed notebooks 02-10 from the committed cleaned snapshot, rebuilt and reloaded model artifacts, ran all 106 tests, original-model comparison and exact predictor/output comparison against the prior performance revision. Raw CSV validation not rerun. The commands above reproduce the checks; generating this report itself does not rerun them. The walkthrough PDF explains prediction and allocation separately.

Before launch, predefine the financial horizon, randomization, power, monitoring, and analysis. Compare **all** customers randomized to the POLICY arm, including those given NO_OFFER, with the randomized control arm using `src.measurement.estimate_policy_itt`. A 5% minimum control share satisfies the project constraint but does not guarantee adequate power.
