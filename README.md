# PromoPilot: targeted promotion analysis

This project recommends `NO_OFFER`, `PCT_10`, `PCT_20`, or `FREE_SHIP` for each EverTrail July customer. The objective is expected **incremental 14-day net contribution margin (NCM)** under campaign constraints.

**Read [the validation report](docs/VALIDATION_REPORT.md) for measured before/after results and limitations.** July totals in the submission and dashboard are model projections. An optimizer ranking first under the predictions it maximizes is not independent evidence of profitable targeting.

**Measured outcome of the performance revision:** two-part XGBoost reduces CP07 NCM RMSE from **63.3694 to 62.7385** (1.00%) and MAE from **27.4111 to 26.4806** (3.39%). Logistic regression's average precision increases from **0.2323 to 0.2367**, while recall falls from **60.88% to 50.10%** at the models' respective development-selected thresholds. The average-precision interval includes no improvement; the recall decline is clear in this retrospective comparison. The selected model is not uniformly better. A simple constrained `RANDOM_FREE_SHIP` policy wins development validation and replaces personalized allocation. Averaging feasible random recipient sets, its retrospective CP07 DR gain is **$7.90 per customer** versus no offer, and **$3.93** versus the original feasible policy, with paired 95% interval **[-$2.12, $10.44]**. Policy superiority remains uncertain.

The saved seed-42 allocation happens to produce a more favorable CP07 estimate: $12.74 versus no offer and a paired $8.77 improvement over the original, interval [$0.36, $18.33]. That result conditions on the chosen recipients. It must not be presented as a reliable gain across recipient randomizations or as observed July profit.

The earlier performance revision changed the fitted models and selected allocation. The subsequent transaction-coverage correction labels 19 purchase/return/product predictors as `observed_*` and adds separate coverage diagnostics. It preserves all 43 model input values, their order, and the original decision windows; it adds no predictive accuracy gain. The XGBoost improvements above come from the prior model revision. [The validation report](docs/VALIDATION_REPORT.md) separates these improvements and their limits.

The current metric extension adds average precision/recall reporting, per-offer profit bias and calibration, and separate purchase/margin diagnostics for the two-part model. These evaluate existing predictions; they are not an additional model-performance gain.

The revised pipeline separates three questions:

1. How well do the models predict observed NCM and promotion spend?
2. What is the value of the **complete constrained policy** on held-out randomized outcomes?
3. Does the July allocation satisfy the constraints, and how will it be tested prospectively?

## Run

Use **Python 3.13** in an isolated environment. The corrected run's core library versions are pinned in `requirements.txt`; ensure the notebooks' `python3` kernel uses that same environment.

For exact before/after comparisons, also preserve the recorded seed and native-thread settings in `outputs/validation_environment.json` and the selection files. The current run records `OMP_NUM_THREADS` as unset. Different numerical environments can change tied optimizer allocations; compare results in the same runtime.

Policy validation executes campaign folds in separate worker processes with time limits and cleanup. This isolates native learner state between campaigns. It addresses a failure seen during a long-lived-process rerun; the underlying native-library cause was not established.

```bash
python -m pip install -r requirements.txt
```

For a raw-data rebuild, put the 11 supplied CSVs in `data/raw/` and run:

```bash
python run_pipeline.py
```

Raw CSVs are intentionally excluded from GitHub. If using the committed cleaned tables instead, rebuild from Step 02:

```bash
python run_pipeline.py --from-clean
```

The correction was verified by executing Steps 02-10 from the committed cleaned snapshot; the original raw CSVs were unavailable. This verifies transformations after cleaning, not the raw inputs. An explicit contiguous range remains available, for example `python run_pipeline.py 06 10`. See [VALIDATION_REPORT.md](docs/VALIDATION_REPORT.md) for validation commands and then run the tests and dashboard:

```bash
python -m pytest tests
streamlit run dashboard/app.py
```

The final `policy_submission.csv` is also copied to `outputs/final/`. Its columns are exactly `customer_id,recommended_action,estimated_incremental_ncm,estimated_promo_cost`.

## Pipeline

| Step | Notebook | Responsibility |
|---|---|---|
| 01 | `01_data_validation` | Validate raw keys, links, timestamps, duplicates, and financial reconciliation; save clean tables |
| 02 | `02_build_historical_features` | Build 43 pre-send predictors plus separate transaction-coverage and historical-reference diagnostics |
| 03 | `03_construct_14_day_outcomes` | Construct purchases, NCM, promotion cost, and explicit refund-timing outcomes |
| 04 | `04_business_analysis` | Descriptive KPIs and exploration-only effects with customer-clustered uncertainty |
| 05 | `05_logistic_propensity_model` | Logistic purchase model; average precision and recall using frozen development thresholds |
| 06 | `06_tlearner_xgboost` | Two-part XGBoost, profit bias/calibration and component diagnostics, separate policy selection |
| 07 | `07_score_july_population` | Rebuild features and coverage diagnostics at each July cutoff and score each action |
| 08 | `08_estimate_promotion_cost` | Expected promotion cost per customer and action |
| 09 | `09_milp_policy_optimization` | Reserve randomized controls and solve the constrained allocation |
| 10 | `10_generate_outputs` | Submission, feasible baseline projections, policy audit, and measurement assignment |

The notebooks read saved outputs from previous steps. Shared modules in `src/` contain feature, outcome, model, inference, and allocation logic so validation can use the same definitions as July scoring. See the [updated walkthrough](docs/Walmart_Pipeline_Walkthrough.md).

## What the scores mean

An action model estimates `E[NCM | customer features, action]`. Estimated incremental NCM is its prediction minus the `NO_OFFER` prediction. Causal models use randomized `exploration` rows; targeted `business_rule` assignments are not treated as a randomized experiment.

Squared-error fitting and RMSE are appropriate diagnostics for **expected dollar NCM**. MAE and R² supply additional descriptions; R² and RMSE give equivalent rankings on the same observations. These factual metrics do not establish that the policy chooses the best action. Matching average effects on the same rows used for fitting is only an in-sample plausibility check.

The predictive search compares 14 fixed expected-NCM configurations on rolling CP04-CP06 validation, fitting prior campaigns only. Minimum pooled factual RMSE selects `HURDLE_XGB_HALF`: pooled XGBoost purchase probability times a signed purchaser-margin estimate, shrunk 50% toward each training arm's purchaser mean. It handles the roughly 88% nonbuyer zeros without discarding negative profit. The logistic search considers feature transforms, compact interactions, regularization, and nested probability calibration; selection favors average precision subject to log-loss/Brier safeguards. The project requires a purchase-propensity baseline; logistic regression is the implemented choice, not a mandated algorithm.

A separate development search compares 21 feasible allocation rules, including model optimizers, pooled/segment rules, and simple baselines. Maximum pooled development DR value selects `RANDOM_FREE_SHIP`. Prediction and allocation winners answer different questions. The previous rule that kept the original unless a multiple-comparison-adjusted interval excluded zero has been replaced by transparent validation ranking with uncertainty reported separately. All choices are saved before the CP07 retrospective comparison. See `outputs/experiments/` for the full search results.

Logged assignment probabilities are not supplied. Evaluation estimates action frequencies separately within each campaign and therefore assumes unconditional randomized offers within campaign. A uniform 25%-per-action sensitivity is also reported. The customer-cluster bootstrap retains all sampled customers' campaign observations and re-estimates campaign action frequencies in each draw. Its intervals are conditional on the fitted outcome/cost models and fixed allocations; they do not include model-refitting uncertainty or future campaign drift.

The original CP07 unconstrained policy-value diagnostic is historical context, not an evaluation of the final allocation: its offer rate exceeded 60%. Revised evaluation applies the complete constraints and compares feasible alternatives using randomized observed outcomes and uncertainty. CP07 was already examined in the original analysis and audit, so this retrospective re-analysis is not a new untouched confirmatory experiment. Prospective randomized evidence is still required.

## Evaluation metrics

The purchase-model headlines are **average precision** and **recall**. Average precision summarizes precision-recall ranking over thresholds. Recall is the fraction of actual purchasers classified positive at a particular threshold; it can reach 100% by predicting everyone positive. Report the frozen threshold, precision and predicted-positive share beside recall.

For the original and selected logistic models, the same max-F1 rule chooses each model's threshold from its own out-of-time CP04-CP06 predictions. Thresholds are saved before CP07 evaluation and held fixed in paired bootstrap draws. `purchase_headline_metrics.csv` contains the comparison; `purchase_thresholds.json` records the thresholds and provenance. The canonical field is `average_precision`; legacy `pr_auc` is retained as an identical compatibility alias, not trapezoidal PR area. Log loss/Brier remain supporting probability-quality checks and selection safeguards, rather than headline metrics.

| Purchase diagnostic on CP07 | Original refit | Selected logistic |
|---|---|---|
| Average precision | 0.232325 | 0.236708 |
| Recall | 60.88% | 50.10% |
| Precision | 20.20% | 21.49% |
| Classified-positive share | 39.74% | 30.74% |
| Frozen development threshold | 0.12903 | 0.15878 |

Both models are evaluated on the same 3,800 CP07 assignments, including 501 purchasers. The selected model has slightly higher average precision and precision, but lower recall at their respective development-selected thresholds: it classifies fewer customers positive and misses more purchasers. The recall difference is **-10.78 percentage points**, with paired 95% interval **[-13.52, -8.28] percentage points** conditional on the fitted models and frozen thresholds. This is a tradeoff, not uniform model improvement; a different threshold would change recall and precision.

The continuous expected-NCM forecast still uses RMSE, MAE and R². Additional files under `outputs/tables/` make its limitations visible:

| Output | Interpretation |
|---|---|
| `profit_bias_by_offer.csv` | Signed mean prediction error (predicted minus observed), dollar errors, and bias intervals overall and by offer |
| `profit_calibration_by_offer.csv` | Predicted versus observed average NCM in prediction bands; tied predictions stay together |
| `hurdle_purchase_metrics.csv` | Average precision and recall for the XGBoost purchase component, overall and by observed offer |
| `purchaser_margin_metrics.csv` | Error in the blended conditional-margin forecast among actual purchasers, preserving negative margins |
| `hurdle_component_predictions.csv` | CP07 purchase probabilities and conditional margins, with their product checked against the expected-NCM forecast |

The two-part purchase threshold is also frozen from earlier-only CP04-CP06 predictions, using one threshold across offers. Its CP07 evaluation uses randomized exploration rows, whereas the logistic baseline uses all CP07 assignments. The XGBoost purchase component has AP **0.234062** and recall **42.96%** on 1,060 randomized CP07 assignments with 135 purchasers. These populations differ, so its AP/recall numbers and the logistic baseline results are not a direct head-to-head ranking. Conditional-margin errors use purchasers only and cannot be compared directly with full-population expected-NCM errors. Prediction bands and bias intervals are descriptive checks conditional on fitted models; they are not new selection evidence or guaranteed population biases.

Run `python scripts/build_metric_diagnostics.py` to refresh the additional diagnostics after the required model and prediction outputs exist. The dashboard's Model performance tab shows the headline purchase metrics, supporting thresholds and precision, offer-level profit bias/calibration, and component errors. The Historical evaluation tab retains **incremental NCM per customer with uncertainty under the full constrained policy** as the primary business evaluation. More metrics do not establish profitable targeting or repair July's missing transaction coverage.

## Timing and financial assumptions

- Historical events must precede `send_ts`; July events must occur on or before `policy_as_of_ts`. Future-event perturbation tests check timestamped feature families.
- Customer tier, membership, preferences, and related profile fields lack effective-date histories. Historical availability remains an explicit assumption; timestamp filtering cannot verify it.
- Outcome orders are completed, belong to the same customer and campaign, and occur in `[send_ts, send_ts + 14 days)`.
- `NCM = net_revenue - cogs - shipping_cost - applicable_refunds`. Net revenue already includes discounts and shipping revenue; promotion cost is not subtracted twice.
- The default `in_window` return rule counts refunds inside the 14-day window. `all_linked` is an alternative accounting sensitivity. Early, late, and total refunds are reconciled separately. The end-to-end return sensitivity compares 14-day and 45-day accrued-return targets on identical mature cohorts, using as-of cutoffs; it does not silently treat every eventual refund as available in historical training. The report distinguishes retrospective CP06 sensitivity from the primary CP07 evaluation.
- FREE_SHIP promotion cost is an inferred waived fee, not a recorded field. Observed paid/free baskets bracket a threshold between $74.99 and $75.08; they do not identify an exact tariff.
- Purchase propensity predicts buying under the historical offer mixture. It does not estimate buying without an offer or causal uplift.

## July support and constraints

All 2,776 July customers have **zero observed completed purchases in their recent 30/90-day windows**. Their complete recent purchase histories are **unknown**. The latest ordinary (`organic_preperiod`) transaction supplied is dated January 9, 2025; later campaign-attributed transactions belong to historical campaign customers. July customers have no historical campaign assignments. This explains the empty recent windows in the extract without establishing that July customers made no purchases. The last observed transaction is not a verified data-coverage endpoint.

The decision cutoff remains **June 20, 2025 at 10:00**. The 90-day window starts **March 22 at 10:00**, and the 30-day window starts **May 21 at 10:00**; July scoring includes the cutoff timestamp. These calculations were already correct. Moving them backward would replace recent behavior with stale history. The 14-day outcome definition also remains unchanged.

`src/coverage.py` checks explicit source coverage against each historical and July decision window. `config/transaction_coverage.json` has an empty `coverage_intervals` list because no complete-transaction source contract was supplied. Therefore `complete_orders_30d` and `complete_orders_90d` remain missing, while `observed_orders_30d` and `observed_orders_90d` retain the available-record counts. Add verified coverage only from a source contract with its scope, interval and availability time; a maximum transaction timestamp cannot certify it.

The pipeline writes `historical_transaction_coverage.csv` and `july_transaction_coverage.csv` under `outputs/tables/`. The separate `historical_organic_reference_window.csv` and `july_organic_reference_window.csv` summarize observed organic purchases in **[October 12, 2024, January 10, 2025)**. In that stale reference interval, 256 July customers have 279 completed orders; 510 have at least one completed order anywhere in the supplied history. These audit tables are not new model predictors or replacements for recent windows, and the reference interval is not certified complete.

Nineteen purchase, return and product predictors now carry an `observed_` prefix; the other predictor names, including `prior_offer_redemption_rate`, stay unchanged. The same 43 predictor values and ordering are retained for both training and scoring. Coverage and reference metadata are excluded from that predictor list. Imputation allows prediction but cannot establish transaction completeness or transfer of historical effects to July.

The allocation uses one action per customer and enforces:

- Expected promotion spend no greater than **$62,000**.
- At most **60%** receiving a commercial offer.
- `PCT_20` no greater than **15% of commercial-offer recipients**.
- At least **5%** randomly reserved as measurement controls and assigned `NO_OFFER`.

The candidate optimizer uses SciPy/HiGHS through `src/policy.py`, without PuLP or a machine-specific CBC executable path. The production allocation follows the development-selected feasible policy; currently this is seeded random FREE_SHIP assignment under the same budget, offer cap, and controls. The improved XGBoost model still supplies action-value forecasts. Unused budget is acceptable, and feasibility under predicted spend is distinct from calibration of actual spend.

For July cost models, Step 08 selects XGBoost or an action-mean predictor **separately for each commercial action using CP06 RMSE**, with the simpler mean winning ties. The choice is fixed before CP07 evaluation and then refitted for July. Cost diagnostics include absolute and relative mean bias; a lower RMSE alone does not guarantee calibrated aggregate spend.

Historical policy folds use the same cost-family rule on their last prior campaign, then refit on earlier data. The final choices are XGBoost for PCT_10/PCT_20 and the action mean for FREE_SHIP; see the saved metadata and report for test calibration.

Required baselines are no offers, PCT_10 by propensity, and the feasible historical offer mix. A feasible randomized FREE_SHIP baseline also tests whether personalization adds value over a simple action choice. Baselines use the same constraints and reserved controls. July projections and historical outcome-based evaluation are reported separately.

## Prospective measurement

Compare **all customers randomized to the policy arm**, including those it assigns `NO_OFFER`, with the randomized control arm. Comparing only selected offer recipients with a random population holdout would confound targeting with intervention effects.

Specify the NCM/return window, analysis, stopping rule, and power before launch. The minimum 5% controls satisfies the project constraint but does not guarantee adequate statistical power. Offer-specific effects need suitable randomization within predefined target groups.

## Repository contents

- `data/raw/`: supplied raw inputs, not committed.
- `data/processed/`: cleaned tables, historical features/outcomes, and July features.
- `src/`: shared computation and evaluation helpers.
- `notebooks/`: ten pipeline steps.
- `tests/`: targeted regression tests for corrected logic.
- `outputs/tables/`: diagnostics, effects, historical evaluation, July projections, and constraints.
- `outputs/models/`: fitted artifacts and model metadata.
- `outputs/figures/`: generated charts.
- `docs/VALIDATION_REPORT.md`: authoritative revision scope, commands, before/after measurements, and remaining limitations.
- `docs/Walmart_Pipeline_Walkthrough.md`: current explanation; regenerate its PDF after editing.
- `docs/Pipeline_Guide.pdf`: historical guide retained for provenance; **superseded** by the revised walkthrough and validation report. Do not use its model/results claims as current guidance.
- `dashboard/app.py`: saved-results dashboard separating projections and held-out evaluation.

Retain dependency versions and input provenance with evaluation runs. Reproducibility claims apply to the recorded environment and inputs, not arbitrary future dependency versions.
