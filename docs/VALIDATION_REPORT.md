**Correction and validation report — November 12, 2026**

The code is more reliable and its performance claims are better supported. **The retained NCM model did not improve in predictive accuracy when compared with the original model refit in the same environment. Profitable personalized targeting is still not established.**

The reference snapshot is `ea950d0e9011ff0e105f5d93758ab54944a7c75f`. The original saved results are preserved in `outputs/reference/`. All changes were developed for the user's `main` branch.

**Before and after: the same prediction task and runtime**

| version | rmse | mae | r2 |
| --- | --- | --- | --- |
| original_saved_snapshot | 63.3774 | 27.4112 | 0.0056 |
| original_refit_current_environment | 63.3694 | 27.4111 | 0.0059 |
| corrected_selected_model | 63.3694 | 27.4111 | 0.0059 |

The original saved snapshot differs slightly from the original refit in this runtime. That difference must not be attributed to the corrections: the original refit and corrected retained model have identical metrics. Primary 14-day NCM unchanged: **True**. Feature differences across all 43 predictors: **0**.

RMSE remains a valid conditional-mean prediction diagnostic; MAE and R² are supporting diagnostics. They do not measure policy quality directly.

**Model selection, without choosing a winner on CP07**

Four predeclared candidates were compared: original T-learner, a 50% shrinkage version, action means, and a pooled XGBoost model with treatment as a feature. The development folds are CP04, CP05, and CP06. Each fold uses earlier campaigns only, and XGBoost tuning uses the last prior campaign as inner validation (for example, CP04 uses CP03 to tune and CP01–02 to fit candidates). Cost-model families are likewise chosen on the last prior campaign before refitting.

Replacement requires a positive lower bound for a paired development improvement, with customer-cluster bootstrap and Bonferroni adjustment across three comparisons. Otherwise the original is retained. This conservative rule is not a claim that the original is optimal.

| candidate | paired_dr_gain_vs_original | adjusted_ci_low | adjusted_ci_high | eligible_to_replace |
| --- | --- | --- | --- | --- |
| ORIGINAL_TLEARNER | 0.0000 | 0.0000 | 0.0000 | False |
| SHRUNK_TLEARNER | 1.2928 | -1.2312 | 4.8612 | False |
| ACTION_MEAN | 4.6761 | -0.8536 | 10.7211 | False |
| POOLED_XGBOOST | 3.7689 | -1.4226 | 9.3954 | False |

Selected method: **ORIGINAL_TLEARNER**. The selection is written to `outputs/models/policy_selection.json` before CP07 evaluation. CP07 was already inspected in the preceding audit, so this is explicitly a **retrospective held-out re-analysis**, not a new prospective test. Do not repeatedly revise the method based on CP07 and continue treating it as untouched.

**Independent evaluation of the complete constrained policy**

The evaluator now applies one action per customer, a population-scaled budget, a 60% offer cap, PCT_20 at most 15% of offered customers, and a 5% reserved NO_OFFER holdout. Models, purchase rankings, costs, and historical mix use earlier campaigns. Competitors share the same constraints and reserved controls.

Observed randomized outcomes are evaluated with doubly robust (DR) and Hájek estimators. DR uses one common reward nuisance model, fitted on prior data, for all competitors. Intervals use 2,000 customer-cluster bootstrap draws, re-estimating campaign/action frequencies inside every draw. Paired comparisons use the same sampled customers. Costs receive an analogous DR check.

The selected feasible policy's CP07 incremental NCM estimate is **$3.97 per customer**, with a **95% interval [$-2.23, $9.68]**. The interval includes zero. The former +$8.73 estimate evaluated an unconstrained policy offering to 87.2% of customers; it is not an apples-to-apples before/after estimate.

| policy | dr_incremental_ncm_per_customer | ci_low | ci_high | hajek_incremental_ncm_per_customer | matched_rows | offered_share |
| --- | --- | --- | --- | --- | --- | --- |
| ORIGINAL_TLEARNER | 3.9714 | -2.2332 | 9.6825 | 3.0755 | 251 | 0.6000 |
| SHRUNK_TLEARNER | 4.8919 | -1.3245 | 11.0158 | 4.0740 | 251 | 0.6000 |
| ACTION_MEAN | 5.5713 | -2.0978 | 13.6234 | 5.9460 | 271 | 0.6000 |
| POOLED_XGBOOST | 6.2281 | -1.9944 | 15.6016 | 5.9096 | 242 | 0.6000 |
| ALL_NO_OFFER | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 257 | 0.0000 |
| PROPENSITY_PCT10 | 3.6510 | -2.4921 | 9.7598 | 3.1123 | 258 | 0.6000 |
| HISTORICAL_MIX | 4.6599 | -0.4839 | 9.7587 | 3.0303 | 237 | 0.6000 |
| RANDOM_FREE_SHIP | 12.7414 | 3.5899 | 22.6637 | 12.2317 | 246 | 0.6000 |

| reference | paired_dr_difference | ci_low | ci_high |
| --- | --- | --- | --- |
| ACTION_MEAN | -1.5999 | -9.6796 | 5.8521 |
| ALL_NO_OFFER | 3.9714 | -2.2332 | 9.6825 |
| HISTORICAL_MIX | -0.6884 | -7.1170 | 5.9735 |
| ORIGINAL_TLEARNER | 0.0000 | 0.0000 | 0.0000 |
| POOLED_XGBOOST | -2.2567 | -10.4080 | 4.6402 |
| PROPENSITY_PCT10 | 0.3204 | -6.8907 | 7.6370 |
| RANDOM_FREE_SHIP | -8.7700 | -18.3279 | -0.3639 |
| SHRUNK_TLEARNER | -0.9205 | -2.9607 | 0.3165 |

The simple seeded random-FREE_SHIP reference performs well on CP07. These individual test intervals are nominal, not adjusted for searching over test comparisons. They should motivate new validation rather than changing the chosen model after seeing CP07. Baseline randomization uses a fixed seed; the intervals condition on that allocation realization.

Actual logging probabilities are not supplied. The primary calculation assumes unconditional randomization within each campaign and uses empirical action frequencies. The table also saves a uniform-0.25 propensity sensitivity. If assignment was stratified or used different known probabilities, supply that design instead. Intervals remain conditional on fitted models and fixed batch allocations; they do not include future campaign drift, transport uncertainty, or all retraining uncertainty. Negative endpoints for DR cost intervals reflect estimator uncertainty, not negative realized costs.

**July transport and return sensitivity materially limit confidence**

July has no prior campaign exposure, no recent purchases, and seven entirely missing feature columns (including five email/redemption features). Missingness, zero rates, ranges, and unseen categories are saved in `july_feature_support.csv`. Imputation makes scoring possible; it does not validate transport.

Comparable historical subgroups are small. The selected policy's subgroup results are:

| subgroup | n | matched_rows | dr_incremental_ncm_per_customer | ci_low | ci_high |
| --- | --- | --- | --- | --- | --- |
| cold_history | 93 | 25 | -9.4716 | -26.0017 | 2.9372 |
| unseen_customer | 83 | 24 | -10.8222 | -29.5079 | 2.9828 |
| cold_and_unseen_customer | 83 | 24 | -10.8222 | -29.5079 | 2.9828 |

The negative subgroup point estimates and broad intervals are a reason for caution, not proof of either benefit or harm. Mutable customer-profile fields still require the documented static-snapshot assumption; timestamps for that history were not supplied.

The return code now reconciles early, late, applicable, and total refunds without double counting. The `all_linked` configuration passes; the default 14-day target is preserved. Multiple refund events no longer duplicate an order's feature contributions.

A full 45-day accrued-return sensitivity refits models and re-optimizes allocations on the **same mature training cohort** under both targets. It excludes CP07 from July training for this sensitivity because its 45-day labels are not mature at June 20. Future refunds are excluded. CP06 sensitivity is also evaluated after its labels mature and is labeled retrospective. The 45-day horizon is an explicit sensitivity assumption, not a newly verified business accounting rule.

| scenario | training_rows | projected_incremental_ncm | projected_spend | actions_changed_between_targets |
| --- | --- | --- | --- | --- |
| PRIMARY_14D_MATURE_COHORT | 6467 | 6236.3000 | 811.7936 | 994 |
| ACCRUED_45D_MATURE_COHORT | 6467 | 4095.1508 | 411.5514 | 994 |

The two target scenarios change **994 of 2,776 assignments**. Their projected values use different accounting targets and must not be compared as proof that one model is better. See `return_sensitivity_cp06.csv` for outcome-based retrospective sensitivity. The deployed primary target remains 14-day NCM.

**Cost models and July projection**

Notebook 08 chooses each cost model using CP06 only, comparing XGBoost with the action mean; CP07 follows selection. FREE_SHIP selects the mean. Its CP07 RMSE is about $0.890 versus about $0.897 in the original saved XGBoost output: a small improvement on that metric, while mean cost calibration remains imperfect.

| action | candidate | rmse | mae | obs_mean | pred_mean | relative_mean_bias |
| --- | --- | --- | --- | --- | --- | --- |
| PCT_10 | xgb | 20.8975 | 8.2918 | 4.9054 | 4.5943 | -0.0634 |
| PCT_20 | xgb | 31.2833 | 16.4322 | 10.8787 | 8.5411 | -0.2149 |
| FREE_SHIP | action_mean | 0.8903 | 0.2258 | 0.1351 | 0.0950 | -0.2972 |

The current July allocation contains 1453 FREE_SHIP, 212 PCT_10, 0 PCT_20, and 1111 NO_OFFER customers. All 139 controls remain NO_OFFER. Projected incremental NCM is **$6,408.91**, and projected promotion spend is **$1,306.57**. These are model projections, not observed gains or independently proven profit. Mathematical feasibility under estimated costs does not guarantee realized spending.

**Correct prospective measurement**

Use `measurement_assignment.csv` and `src.measurement.estimate_policy_itt`. Compare all customers randomly assigned to the POLICY arm, including optimizer-selected NO_OFFER customers, against the randomized holdout. Require complete outcome coverage, use the agreed accounting horizon, and perform a power calculation. The estimator separately reports the arm contrast and the campaign-level scaled effect; it does not silently extrapolate to a different rollout intensity.

**Verification and reproducibility**

- All **49 focused tests passed**: refund modes and maturity, feature merge safety, clustered inference, DR identities and logging-probability resampling, exact policy coverage, optimizer feasibility, baseline constraints, deterministic ordering, selection separation, cost selection timing, and ITT analysis.
- Notebooks **02–10 executed successfully** from the committed clean snapshot. Raw CSVs are intentionally absent; notebook 01's original raw-data validation was not rerun and is not claimed as newly verified.
- The walkthrough PDF was rebuilt, and documentation/dashboard distinguish projected values from held-out evidence.
- Core package versions are pinned in `requirements.txt`; the actual run environment is recorded in `outputs/validation_environment.json`.

Reproduce from a checkout with the pinned environment:

```text
python -m pytest -q
python run_pipeline.py --from-clean
python scripts/compare_reference.py
python scripts/build_validation_report.py
```

The reference comparison needs the original Git commit available locally. Full raw-data validation requires the 11 source CSVs and `python run_pipeline.py` without `--from-clean`.

The next substantive improvement requires stronger evidence for the target population, including a correctly analyzed randomized campaign. More favorable model projections alone would not demonstrate improvement.
