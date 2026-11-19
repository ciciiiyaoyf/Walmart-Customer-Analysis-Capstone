# Correction and performance validation

Updated November 18, 2026: **the current deployment uses HURDLE_XGB_HALF forecasts with a deterministic greedy allocator**, replacing MILP at the project owner's request. The configured allocator is `deterministic_incremental_ncm_greedy`. It produces a feasible allocation without a general global-optimality guarantee. The development-policy winner is separately recorded as **RANDOM_FREE_SHIP**; using the greedy deployment does not establish that it outperforms that comparator. The choice and unchanged measurement-control design are documented in `config/allocation_policy.json`.

The main workflow retains **the original logistic regression as its one purchase-propensity baseline**. The complex logistic version remains an experiment record; restoring the original does not establish a new performance gain. Execution checks for that earlier restoration are recorded in `outputs/original_logistic_validation.json`.


## Allocator replacement on identical July inputs

| allocation | projected_incremental_ncm | expected_promo_cost |
| --- | --- | --- |
| Previous random submission | 1999.3748 | 167.4434 |
| Previous MILP reference | 2501.7834 | 167.4434 |
| Current greedy allocation | 2501.7834 | 167.4434 |

Compared with the previous MILP, projected incremental NCM changed by **$0.00**, with **0 changed assignments**. Compared with the earlier random submission, the change is **$502.41** and **1234 changed assignments**. That latter difference comes from deploying model-ranked recipients instead of random recipients; it is not an improvement over MILP.

The audit verifies 29 unchanged input/model hashes, unchanged controls, matching saved and recomputed greedy assignments, and all constraints. The same model-driven CP07 policy point estimate is unchanged. The heuristic can be suboptimal on other inputs; a regression test explicitly demonstrates a binding-budget counterexample. These are forecasts and retrospective diagnostics, not realized July gains.

The two proposed measurement-control designs are documented in `docs/CONTROL_GROUP_OPTIONS.md`; neither was adopted in this revision.

The allocator revision passed **159 tests**, plus 1,000 randomized feasibility/order-invariance scenarios.


**The prior performance revision improved prediction modestly; coverage handling prevents unsupported inactivity claims; the current metrics revision adds purchase recall and detailed profit diagnostics.** Policy superiority remains uncertain. Earlier corrections fixed accounting, timing, constraints, and misleading evaluation while retaining the original model. The subsequent performance revision changed the fitted models and selected allocation independently on development campaigns. The coverage correction preserves those numeric predictor values and properly labels their observation scope. The metrics revision measures the existing models more completely; adding metrics is not a new improvement in their predictions.

## Purchase headline metrics and expanded diagnostics

**Average precision and recall** are the headline purchase-classification measures. Average precision summarizes the precision-recall ranking without a fixed threshold. Recall is the fraction of actual purchasers classified as purchasers at a frozen threshold. The logistic baseline's threshold is chosen by maximum F1 using only CP04-CP06 out-of-time predictions, then serialized before CP07 evaluation. The hurdle purchase component follows the same development-only rule with its own global threshold. The resulting recall is an operating-point diagnostic, not an offer-allocation rule.

The baseline uses the original `make_propensity_pipe` specification: all 43 registered predictors, numeric imputation and standardization, categorical encoding, and `C=0.01`. It adds no log transformations, interactions, or extra probability calibration. Its configuration is recorded in `config/logistic_baseline.json`. The evaluation fit uses CP01-CP06 and the deployment fit uses CP01-CP07; only the latter supplies July propensity scores.

| model | average_precision | recall | precision | threshold | predicted_positive_share |
| --- | --- | --- | --- | --- | --- |
| logistic_baseline | 0.2323 | 0.6088 | 0.2020 | 0.1290 | 0.3974 |

The earlier logistic restoration reproduced all **3,800 original CP07 probabilities** within the recorded tolerance. It changed July probabilities for **2,776 customers** and changed **88 actions** in the comparison policy that gives PCT_10 to high-propensity customers. At that revision, the final selected-policy submission remained unchanged: **True**. The subsequent allocator revision is a separate change. Saved profit models and their forecasts are separate from the logistic restoration, whose recorded run passed **151 tests** and its dashboard smoke test.

Categorical missing values are normalized to one consistent representation before encoding. This preserves the original learned missing category and makes live feature joins and saved Parquet inputs produce identical probabilities. The 43 base inputs and original CP07 predictions remain unchanged by that representation fix.

Precision, flagged-customer share and the threshold accompany recall so that predicting everyone to purchase cannot masquerade as a useful gain. Log loss, Brier score and ROC-AUC remain supporting diagnostics; the existing model-selection safeguards are retained. `average_precision` is the canonical name; `pr_auc` remains an identical compatibility alias, not trapezoidal PR area.

For the earlier metrics-only revision, numerical comparison with the preceding coverage commit `2cec21b` confirms predictions and the submission unchanged: **True**. Full row-aligned differences are saved in `metric_revision_comparison.json`.

Implemented outputs:

- `purchase_headline_metrics.csv`, `purchase_thresholds.json`: one logistic baseline's average precision/recall and reproducible threshold.
- `profit_bias_by_offer.csv`: signed dollar bias, RMSE/MAE/R2, and observed/predicted means overall and by actual offer, for original and selected models.
- `profit_calibration_by_offer.csv`: predicted-versus-observed NCM in prediction bands, without arbitrarily splitting tied predictions.
- `hurdle_purchase_metrics.csv`, `hurdle_threshold_selection.json`: purchase-component average precision/recall and development threshold provenance.
- `purchaser_margin_metrics.csv`: conditional margin prediction error among actual purchasers, preserving negative margins.
- `hurdle_component_predictions.csv`: assignment-level probability and conditional margin, whose product must match saved expected-NCM predictions.

The existing held-out policy evaluation remains the business measure: incremental NCM per customer versus no offer and feasible alternatives, under the full constraints and with uncertainty. Purchase metrics do not apply to continuous NCM forecasts and do not replace policy evaluation. None of these diagnostics supplies missing July outcomes or certifies transaction coverage. Definitions follow [average precision](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.average_precision_score.html), [recall](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.recall_score.html), and [separate threshold tuning](https://scikit-learn.org/stable/modules/classification_threshold.html).

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
- Historical logistic experiment: the more complex version produced small changes in average precision and probability quality, with uncertainty including zero improvement. Recall was lower at its separately chosen threshold because fewer customers were flagged. The current baseline restores the original specification for simplicity.
- Feasible allocation: **RANDOM_FREE_SHIP** wins the rerun CP04-CP06 development comparison. Production instead uses **HURDLE_XGB_HALF with the user-requested greedy allocator**. For the separate random-free-shipping comparator, averaging feasible random recipient sets gives CP07 DR incremental NCM of **$7.90 per customer** versus no offer; its paired difference versus the original-model greedy policy is **$3.93**, 95% interval **[-$2.12, $10.44]**.

CP07 was already inspected in earlier project work. These are **retrospective comparisons**, not a new untouched confirmatory test. They are not realized July gains. The choice of a model-driven deployment is explicit and is not evidence that personalization improves profit.

## Fair predictive comparison

The reference is commit `ea950d0e9011ff0e105f5d93758ab54944a7c75f`. The original model is refitted in the current environment and compared on identical CP07 assignments, observed actions, and targets. Saved original metrics are retained for provenance, not used to attribute runtime differences to this revision.

| version | role | model | rmse | mae | r2 |
| --- | --- | --- | --- | --- | --- |
| original_saved_snapshot | profit_prediction | ORIGINAL_TLEARNER | 63.3774 | 27.4112 | 0.0056 |
| original_refit_current_environment | profit_prediction | ORIGINAL_TLEARNER | 63.3694 | 27.4111 | 0.0059 |
| corrected_predictive_champion | profit_prediction | HURDLE_XGB_HALF | 62.7385 | 26.4806 | 0.0256 |

Predictive champion: **HURDLE_XGB_HALF**. Deployment forecast model: **HURDLE_XGB_HALF**, used by the greedy allocator. Development-policy winner: **RANDOM_FREE_SHIP**. Prediction accuracy does not establish good treatment-effect ranking. Primary 14-day NCM unchanged: **True**. Feature columns with changed numerical/category values: **0**; separately renamed observation-scope fields: **19**. Factual CP07 predictions changed for **1060** assignments in the earlier prediction revision. Replacing the allocator does not change those fitted predictions.

In the prior model-performance revision, compared with reliability revision `1221dd1`, **1,403 of 2,776 July actions changed**. That is a concrete behavior change, not proof of higher realized profit. The comparison and input fingerprint are saved in `performance_revision_changes.json`.

## XGBoost: why the change matters

The original fits four independent regressors. About 88% of randomized training rows are nonbuyers with zero NCM, and some purchaser margins are negative. The selected two-part model shares information across action arms and estimates purchase probability times signed purchaser margin. The margin forecast blends pooled XGBoost with each training arm's purchaser mean, using 50% weight on each. It neither logs the target nor clips negative profit. Future purchase is a training label, never a scoring feature.

Fourteen fixed configurations compare original, pooled, shared-baseline, and two-part estimators. Rolling CP04-CP06 validation fits earlier campaigns only; minimum pooled factual RMSE selects the winner. The configuration is serialized before CP07 evaluation. The code and full search results are in `src/profit_models.py` and `outputs/experiments/profit/`.

On CP07, old-minus-new MSE is **79.57**, paired customer-bootstrap 95% interval **[17.45, 140.35]**. MAE improvement is **0.931**, interval **[0.452, 1.423]**. Five thousand draws hold fits fixed and omit selection, retraining, and future-campaign uncertainty. Three arm-specific RMSEs improve, while NO_OFFER worsens slightly. R² remains low, so realized individual profit remains difficult to predict.

RMSE is appropriate for expected dollar NCM and squared-error fitting. MSE adds no independent ranking information because RMSE is its square root; on identical observations R² gives the same ranking. MAE describes absolute dollar errors, but minimizing it instead targets a conditional median, often zero in this sparse outcome. None of these metrics proves causal uplift or policy profit.

## Historical logistic experiment: improvement was small

The earlier optional experiment compared transformations, compact interactions, regularization, and nested temporal probability calibration on rolling CP04-CP06 development. Selection maximized mean average precision subject to log-loss/Brier safeguards. The transformed/interacted configuration was tried, but the current baseline now uses the original specification by the project owner's explicit choice. The comparison below is read from `outputs/experiments/propensity/`; Step 05 does not repeat it or require two baselines. Both versions in this archived comparison used the same training/test populations and runtime.

| model | average_precision | recall | precision | threshold | predicted_positive_share |
| --- | --- | --- | --- | --- | --- |
| original | 0.2323 | 0.6088 | 0.2020 | 0.1290 | 0.3974 |
| selected | 0.2367 | 0.5010 | 0.2149 | 0.1588 | 0.3074 |

The earlier original specification caught **60.88%** of actual purchasers at its development threshold, versus **50.10%** for the more complex experimental configuration at its own threshold. The latter was not uniformly better: it used a stricter operating point, flagged fewer customers and had higher precision. This archived diagnostic did not retune either threshold using CP07 outcomes.

The archived experimental model's average-precision difference versus the original is **0.004384**, with paired 95% interval **[-0.004445, 0.013209]**. Its recall difference at the respective frozen development thresholds is **-0.107784**, interval **[-0.135243, -0.082807]**. The recall comparison includes different operating thresholds and must be read together with precision and flagged share. Thresholds are held fixed in the customer bootstrap; intervals omit threshold/model selection, refitting and future-campaign uncertainty. Purchase propensity describes buying under the historical offer mixture; it is not untreated purchase probability or causal uplift.

## Profit bias, calibration and two-part diagnostics

Signed mean prediction bias is **predicted minus observed** NCM: positive values indicate overprediction in this sample, negative values underprediction. Opposite errors across offers can cancel in the pooled average. These are factual observed-action diagnostics; they do not verify individual treatment effects or counterfactual profits.

| model | scope | n | obs_mean | pred_mean | mean_prediction_bias | bias_ci_low | bias_ci_high |
| --- | --- | --- | --- | --- | --- | --- | --- |
| HURDLE_XGB_HALF | pooled | 1060 | 15.3148 | 15.9756 | 0.6608 | -3.1867 | 4.3962 |
| HURDLE_XGB_HALF | NO_OFFER | 257 | 10.2449 | 18.2170 | 7.9721 | 2.4969 | 12.6264 |
| HURDLE_XGB_HALF | PCT_10 | 294 | 16.1856 | 15.6729 | -0.5127 | -9.8746 | 6.5416 |
| HURDLE_XGB_HALF | PCT_20 | 243 | 11.3396 | 11.0982 | -0.2414 | -4.9524 | 4.0894 |
| HURDLE_XGB_HALF | FREE_SHIP | 266 | 22.8823 | 18.6002 | -4.2822 | -14.3930 | 4.4293 |
| ORIGINAL_TLEARNER | pooled | 1060 | 15.3148 | 16.5737 | 1.2589 | -2.6246 | 5.0393 |
| ORIGINAL_TLEARNER | NO_OFFER | 257 | 10.2449 | 17.8076 | 7.5627 | 2.0756 | 12.2334 |
| ORIGINAL_TLEARNER | PCT_10 | 294 | 16.1856 | 14.7944 | -1.3911 | -10.9617 | 5.8341 |
| ORIGINAL_TLEARNER | PCT_20 | 243 | 11.3396 | 9.6422 | -1.6974 | -6.5514 | 2.7534 |
| ORIGINAL_TLEARNER | FREE_SHIP | 266 | 22.8823 | 23.6802 | 0.7978 | -9.4685 | 9.5394 |

Calibration-band details are saved in `profit_calibration_by_offer.csv` and displayed in the dashboard. Means and their bias intervals are estimated within each actual offer and prediction band. Small groups and extreme signed profits make these diagnostics noisy. Customer bootstrap intervals condition on the fitted models and bins; they do not establish future calibration or adjust for many comparisons. No CP07-derived recalibration is applied.

The hurdle purchase-component classification results use only the **1,060 randomized CP07 assignments** and each customer's actual offer. Logistic evaluation above uses all 3,800 CP07 assignments under their historical offer mixture. Their average precision and recall values are therefore **not a like-for-like contest between algorithms**.

| scope | n | buyers | average_precision | recall | precision | threshold | predicted_positive_share |
| --- | --- | --- | --- | --- | --- | --- | --- |
| pooled | 1060 | 135 | 0.2341 | 0.4296 | 0.2222 | 0.1520 | 0.2462 |
| NO_OFFER | 257 | 26 | 0.2237 | 0.5385 | 0.2121 | 0.1520 | 0.2568 |
| PCT_10 | 294 | 37 | 0.2386 | 0.4054 | 0.2206 | 0.1520 | 0.2313 |
| PCT_20 | 243 | 36 | 0.2791 | 0.3889 | 0.2414 | 0.1520 | 0.2387 |
| FREE_SHIP | 266 | 36 | 0.2676 | 0.4167 | 0.2174 | 0.1520 | 0.2594 |

Conditional purchaser-margin performance is evaluated only among actual purchasers, using the model's blended margin forecast before multiplying by purchase probability. It is not the expected-NCM forecast restricted to purchasers. Errors on this selected subgroup answer a different question from whole-population profit error:

| scope | n | rmse | mae | r2 | obs_mean | pred_mean | mean_prediction_bias |
| --- | --- | --- | --- | --- | --- | --- | --- |
| pooled | 135 | 131.1006 | 85.8259 | 0.1000 | 120.2498 | 125.9236 | 5.6738 |
| NO_OFFER | 26 | 91.1329 | 79.0555 | -0.2302 | 101.2669 | 140.6253 | 39.3583 |
| PCT_10 | 37 | 166.8146 | 91.5532 | 0.0575 | 128.6097 | 131.2195 | 2.6098 |
| PCT_20 | 36 | 65.0245 | 49.8181 | 0.0663 | 76.5422 | 90.8900 | 14.3478 |
| FREE_SHIP | 36 | 160.0819 | 120.8369 | 0.0593 | 169.0750 | 144.8962 | -24.1788 |

Negative margins remain valid. MAPE is unsuitable for the many zero-NCM outcomes; changing metrics cannot resolve the underlying noise or prove profitable targeting.

## Compare feasible policies and document the deployment choice

Development compares 21 allocation rules, including greedy allocations from model forecasts, pooled/segment shrinkage rules, and simple baselines. Every fold uses earlier campaigns for reward and cost fitting, with the same budget, offer cap, PCT_20 cap, and reserved controls. All model-based candidates now use `deterministic_incremental_ncm_greedy`. The development winner maximizes pooled customer-weighted DR value. The separate deployment decision is: **User-requested model-driven allocation using the predictive champion and deterministic greedy allocation; not a claim of historical policy superiority.**

| policy | n | development_dr_incremental_ncm |
| --- | --- | --- |
| RANDOM_FREE_SHIP | 3237 | 3.8176 |
| HISTORICAL_MIX | 3237 | 3.3395 |
| SEGMENT_MEMBERSHIP_HISTORY_RIDGE_100 | 3237 | 2.0797 |
| SEGMENT_MEMBERSHIP_HISTORY_RIDGE_20 | 3237 | 2.0797 |
| ACTION_MEAN | 3237 | 1.4569 |
| SEGMENT_MEMBERSHIP_HISTORY_RIDGE_300 | 3237 | 0.9126 |
| POOLED_XGBOOST | 3237 | 0.4500 |
| SEGMENT_MEMBERSHIP_CART_RIDGE_100 | 3237 | 0.3786 |

Development ranking is selection-biased. The development winner and the configured deployment candidate are evaluated on CP07 with common reward nuisance predictions for all policies. CP07 uses 1,060 randomized assignments and the population-scaled budget. The former unconstrained +$8.73 diagnostic offered promotions to 87.2% of customers and is not the baseline for this comparison. `selected_for_july` and `selected_by_development` are distinct flags in the saved comparison.

| policy | dr_incremental_ncm_per_customer | ci_low | ci_high | hajek_incremental_ncm_per_customer | offered_share |
| --- | --- | --- | --- | --- | --- |
| ORIGINAL_TLEARNER | 3.9714 | -2.2311 | 9.6737 | 3.0755 | 0.6000 |
| ACTION_MEAN | 12.5233 | 3.1494 | 22.5177 | 11.6142 | 0.6000 |
| HURDLE_XGB_HALF | 8.8888 | 0.2528 | 18.0299 | 8.9169 | 0.6000 |
| ALL_NO_OFFER | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| PROPENSITY_PCT10 | 3.6510 | -2.4921 | 9.7598 | 3.1123 | 0.6000 |
| HISTORICAL_MIX | 4.6599 | -0.4839 | 9.7587 | 3.0303 | 0.6000 |
| RANDOM_FREE_SHIP | 12.7414 | 3.5899 | 22.6637 | 12.2317 | 0.6000 |

| reference | paired_dr_difference | ci_low | ci_high |
| --- | --- | --- | --- |
| ORIGINAL_TLEARNER | 4.9174 | -2.7122 | 13.2738 |
| ACTION_MEAN | -3.6346 | -11.5797 | 4.0218 |
| HURDLE_XGB_HALF | 0.0000 | 0.0000 | 0.0000 |
| ALL_NO_OFFER | 8.8888 | 0.2528 | 18.0299 |
| PROPENSITY_PCT10 | 5.2378 | -3.0086 | 14.8580 |
| HISTORICAL_MIX | 4.2289 | -4.7588 | 14.1547 |
| RANDOM_FREE_SHIP | -3.8527 | -12.5708 | 4.1208 |

The configured **HURDLE_XGB_HALF greedy allocation** has CP07 DR gain **$8.89**, interval **[$0.25, $18.03]**, and paired difference versus the original-model greedy allocation of **$4.92**, interval **[$-2.71, $13.27]**. These results evaluate a policy, not prediction RMSE.

The separate seed-42 RANDOM_FREE_SHIP comparator has CP07 DR gain **$12.74**, interval **[$3.59, $22.66]**. The following diagnostic averages random recipient sets with the same controls; its numbers belong to the random comparator, not the deployed greedy allocation:

| comparison | gain | ci_low | ci_high |
| --- | --- | --- | --- |
| versus NO_OFFER | 7.9018 | 1.3563 | 15.3476 |
| versus ORIGINAL_TLEARNER | 3.9304 | -2.1177 | 10.4405 |
| versus RANDOM_FREE_SHIP | -4.8396 | -9.7655 | -0.0505 |

The random-recipient diagnostic does not retune selection and is not evidence that the requested greedy allocation is superior. A fixed seed alone does not make one random recipient realization representative. The deployment choice is explicit configuration, while the development ranking and uncertainty remain visible. A new randomized campaign is needed to assess future performance.

Actual logging probabilities were not supplied. Primary estimates assume unconditional randomized assignment within campaign and use empirical action frequencies; uniform-0.25 sensitivity is also saved. Customer-cluster bootstrap draws retain repeated-customer observations and re-estimate logging frequencies. Policy intervals use 2,000 draws, condition on fitted models and fixed allocations, and omit future campaign drift and full retraining uncertainty. Nominal test intervals are not adjusted for repeated earlier inspection of CP07.

## July support, accounting, and cost remain material limits

July has no historical campaign exposure, no observed recent orders in the supplied extract, and seven entirely missing predictor columns. Complete recent purchase history is unknown. Imputation permits scoring; it does not validate transport. Selected-policy subgroup evidence uses observed-history labels, not verified inactivity:

| subgroup | n | matched_rows | dr_incremental_ncm_per_customer | ci_low | ci_high |
| --- | --- | --- | --- | --- | --- |
| first_exposure_no_observed_recent_orders | 93 | 25 | -0.0467 | -4.5609 | 4.7769 |
| unseen_customer | 83 | 24 | -0.0523 | -5.0809 | 5.3057 |
| unseen_no_observed_recent_orders | 83 | 24 | -0.0523 | -5.0809 | 5.3057 |

Subgroups are small and uncertain. Profile fields without effective-date history still require the explicit static-snapshot assumption.

Earlier fixes remain meaningful: eligible events precede decisions; duplicate refund events no longer duplicate purchase features; early/late/total refunds reconcile; the allocator enforces the actual constraints; and prospective measurement compares the full randomized POLICY arm with controls. These improve validity even when prediction accuracy is unchanged.

The 45-day accrued-return sensitivity uses identical mature training cohorts and excludes CP07 from July training. It retains the selected allocation rule and re-estimates values under both financial targets. It is an accounting stress test, not a verified new business rule or fresh test.

| scenario | training_rows | projected_incremental_ncm | projected_spend | actions_changed_between_targets |
| --- | --- | --- | --- | --- |
| PRIMARY_14D_MATURE_COHORT | 6467 | 220.0368 | 167.4434 | 1665 |
| ACCRUED_45D_MATURE_COHORT | 6467 | 0.0000 | 0.0000 | 1665 |

The configured allocation changes **1,665 customer actions** between targets, and its mature-cohort projected increment **does not cross zero between the two financial targets**. These are accounting sensitivities, not measured improvements. Different-target projections cannot establish that one model is better, and differ from the primary July projection because this sensitivity excludes CP07 training.

Cost models are selected by action on prior-campaign RMSE before CP07 testing. Mean bias matters for aggregate spend even when individual RMSE improves.

FREE_SHIP cost now uses the project owner's explicit **$75 ordinary free-shipping cutoff**, recorded in `config/shipping_cost_assumptions.json`. The source observations bracket a threshold above $74.99 and at or below $75.08 in gross basket value; the exact tariff is not documented in the supplied data package. All 601 positive customer-paid shipping fees are $5.99. The code assumes a redeemed FREE_SHIP order below $75 would otherwise have paid that fee, while a basket of $75 or more already qualifies for ordinary free shipping. This is a counterfactual waived-fee assumption, separate from physical shipping expense.

The source counts and boundary evidence are saved in `outputs/tables/shipping_fee_evidence.json`. Changing the operational cutoff from $75.08 to the explicit $75 assumption leaves all 26,600 historical outcome rows and joined feature/outcome rows unchanged: no supplied completed basket lies between the observed bounds. The boundary test checks $74.99 as chargeable and $75.00 or more as already free. See `outputs/shipping_cutoff_validation.json`; this clarification adds no model-accuracy gain.

| action | candidate | rmse | mae | obs_mean | pred_mean | relative_mean_bias |
| --- | --- | --- | --- | --- | --- | --- |
| PCT_10 | xgb | 20.8975 | 8.2918 | 4.9054 | 4.5943 | -0.0634 |
| PCT_20 | xgb | 31.2833 | 16.4322 | 10.8787 | 8.5411 | -0.2149 |
| FREE_SHIP | action_mean | 0.8903 | 0.2258 | 0.1351 | 0.0950 | -0.2972 |

The July submission contains 1665 FREE_SHIP, 0 PCT_10, 0 PCT_20, and 1111 NO_OFFER customers. Its 139 reserved controls remain NO_OFFER. Forecast incremental NCM is **$2,501.78**, and forecast promotion spend is **$167.44**. These use the improved predictor to describe the selected allocation; they are not observed profit. Comparing this forecast with an old optimizer forecast under different models is not evidence of better or worse realized profit. Estimated-cost feasibility does not guarantee realized spending.

## Reproduction and prospective measurement

Raw CSVs are intentionally excluded from GitHub. The original ZIP was subsequently found and Step 01 revalidated all 11 raw tables; see `outputs/data_quality_validation.json`. A clean-data rebuild alone validates transformations after cleaning. The original reference commit must be available locally for the fair comparison. Core versions are pinned in `requirements.txt`; `outputs/validation_environment.json` records the run environment.

Policy-validation campaigns execute in separate worker processes with deadlines and cleanup, isolating native learner state across folds. A long-lived-process rerun encountered a native XGBoost failure while standalone evaluation succeeded. Process isolation is a practical safeguard; it does not establish the underlying library bug or its cause. Preserve recorded seeds and native-thread settings when comparing numerical results.

```text
python -m pytest -q
python run_pipeline.py --from-clean
python scripts/compare_reference.py
python scripts/build_metric_diagnostics.py
python scripts/build_validation_report.py
python docs/build_walkthrough_pdf.py
```

The recorded verification run passed **130 focused tests**. Focused checks cover signed-margin factorization, exploration-only fitting, serialization, assignment alignment, feature timing, observation-scope parity, explicit coverage contracts/unknown zeros, frozen purchase thresholds, signed bias/calibration, conditional-margin decomposition, refund accounting/maturity, DR/bootstrap logic, feasible allocation, and ITT analysis. Recorded execution scope: Earlier coverage revision executed notebooks 02-10 from committed cleaned data. Metrics revision executed notebooks 05-10, rebuilt all model/policy artifacts and diagnostic tables, passed all 130 tests, and compared predictions and submission with the preceding coverage commit. Raw CSV validation not rerun. The historical scripts `validate_coverage_revision.py` and `validate_metric_revision.py` compare their original revision snapshots; run them on the corresponding historical checkout. Restoring the original baseline intentionally changes their saved probability references. Generating this report itself does not rerun validation. The walkthrough PDF explains prediction and allocation separately.

Before launch, predefine the financial horizon, randomization, power, monitoring, and analysis. Compare **all** customers randomized to the POLICY arm, including those given NO_OFFER, with the randomized control arm using `src.measurement.estimate_policy_itt`. A 5% minimum control share satisfies the project constraint but does not guarantee adequate power.
