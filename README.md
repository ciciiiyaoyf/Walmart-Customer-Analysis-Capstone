# PromoPilot: targeted promotion analysis

This project recommends `NO_OFFER`, `PCT_10`, `PCT_20`, or `FREE_SHIP` for each EverTrail July customer. The objective is expected **incremental 14-day net contribution margin (NCM)** under campaign constraints.

**Read [the validation report](docs/VALIDATION_REPORT.md) for measured before/after results and limitations.** July totals in the submission and dashboard are model projections. An optimizer ranking first under the predictions it maximizes is not independent evidence of profitable targeting.

**Measured outcome of the correction:** the conservative development rule retained `ORIGINAL_TLEARNER`. Its same-runtime CP07 RMSE is unchanged at **63.3694**. The complete feasible policy's retrospective DR lift is **$3.97 per customer**, with a 95% interval **[-$2.23, $9.68]**; profitable personalization remains unproven. All **49 focused tests passed**. Reliability and evaluation improved; an improvement in the NCM model's predictive accuracy was not demonstrated. The report also records cost-model changes, small cold-history samples, and substantial sensitivity to return accounting.

The revised pipeline separates three questions:

1. How well do the models predict observed NCM and promotion spend?
2. What is the value of the **complete constrained policy** on held-out randomized outcomes?
3. Does the July allocation satisfy the constraints, and how will it be tested prospectively?

## Run

Use **Python 3.13** in an isolated environment. The corrected run's core library versions are pinned in `requirements.txt`; ensure the notebooks' `python3` kernel uses that same environment.

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
| 02 | `02_build_historical_features` | Build 43 pre-send features using the shared feature builder |
| 03 | `03_construct_14_day_outcomes` | Construct purchases, NCM, promotion cost, and explicit refund-timing outcomes |
| 04 | `04_business_analysis` | Descriptive KPIs and exploration-only effects with customer-clustered uncertainty |
| 05 | `05_logistic_propensity_model` | Purchase-propensity baseline |
| 06 | `06_tlearner_xgboost` | Action-specific expected NCM models and diagnostic evaluation |
| 07 | `07_score_july_population` | Rebuild features at each July cutoff and score each action |
| 08 | `08_estimate_promotion_cost` | Expected promotion cost per customer and action |
| 09 | `09_milp_policy_optimization` | Reserve randomized controls and solve the constrained allocation |
| 10 | `10_generate_outputs` | Submission, feasible baseline projections, policy audit, and measurement assignment |

The notebooks read saved outputs from previous steps. Shared modules in `src/` contain feature, outcome, model, inference, and allocation logic so validation can use the same definitions as July scoring. See the [updated walkthrough](docs/Walmart_Pipeline_Walkthrough.md).

## What the scores mean

An action model estimates `E[NCM | customer features, action]`. Estimated incremental NCM is its prediction minus the `NO_OFFER` prediction. Causal models use randomized `exploration` rows; targeted `business_rule` assignments are not treated as a randomized experiment.

Squared-error fitting and RMSE are appropriate diagnostics for **expected dollar NCM**. MAE and R² supply additional descriptions; R² and RMSE give equivalent rankings on the same observations. These factual metrics do not establish that the policy chooses the best action. Matching average effects on the same rows used for fitting is only an in-sample plausibility check.

The revised validation compares the original T-learner with simpler/shrunk and pooled alternatives. Candidate selection uses rolling CP04–CP06 validation with nested temporal hyperparameter tuning: each fold tunes on its last prior campaign and refits on earlier campaigns only (CP04 uses CP03 internally, CP05 uses CP04, CP06 uses CP05). CP07 is evaluated after the family choice is fixed, with CP06 as its inner tuning campaign. A replacement requires a positive lower confidence bound for paired DR improvement after correction for three candidate comparisons; otherwise the original is retained. The report identifies the actual choice and evidence.

Logged assignment probabilities are not supplied. Evaluation estimates action frequencies separately within each campaign and therefore assumes unconditional randomized offers within campaign. A uniform 25%-per-action sensitivity is also reported. The customer-cluster bootstrap retains all sampled customers' campaign observations and re-estimates campaign action frequencies in each draw. Its intervals are conditional on the fitted outcome/cost models and fixed allocations; they do not include model-refitting uncertainty or future campaign drift.

The original CP07 unconstrained policy-value diagnostic is historical context, not an evaluation of the final allocation: its offer rate exceeded 60%. Revised evaluation applies the complete constraints and compares feasible alternatives using randomized observed outcomes and uncertainty. CP07 was already examined in the original analysis and audit, so this retrospective re-analysis is not a new untouched confirmatory experiment. Prospective randomized evidence is still required.

## Timing and financial assumptions

- Historical events must precede `send_ts`; July events must occur on or before `policy_as_of_ts`. Future-event perturbation tests check timestamped feature families.
- Customer tier, membership, preferences, and related profile fields lack effective-date histories. Historical availability remains an explicit assumption; timestamp filtering cannot verify it.
- Outcome orders are completed, belong to the same customer and campaign, and occur in `[send_ts, send_ts + 14 days)`.
- `NCM = net_revenue - cogs - shipping_cost - applicable_refunds`. Net revenue already includes discounts and shipping revenue; promotion cost is not subtracted twice.
- The default `in_window` return rule counts refunds inside the 14-day window. `all_linked` is an alternative accounting sensitivity. Early, late, and total refunds are reconciled separately. The end-to-end return sensitivity compares 14-day and 45-day accrued-return targets on identical mature cohorts, using as-of cutoffs; it does not silently treat every eventual refund as available in historical training. The report distinguishes retrospective CP06 sensitivity from the primary CP07 evaluation.
- FREE_SHIP promotion cost is an inferred waived fee, not a recorded field. Observed paid/free baskets bracket a threshold between $74.99 and $75.08; they do not identify an exact tariff.
- Purchase propensity predicts buying under the historical offer mixture. It does not estimate buying without an offer or causal uplift.

## July support and constraints

July customers have no historical campaign assignments and no purchases in the available prior 90-day window. Several history features are completely missing. Imputation allows prediction; it does not establish that effects transfer to this population. Validation examines comparable customers and feature support, and reporting retains the limitation.

The allocation uses one action per customer and enforces:

- Expected promotion spend no greater than **$62,000**.
- At most **60%** receiving a commercial offer.
- `PCT_20` no greater than **15% of commercial-offer recipients**.
- At least **5%** randomly reserved as measurement controls and assigned `NO_OFFER`.

The allocator uses SciPy/HiGHS through `src/policy.py`, without PuLP or a machine-specific CBC executable path. Unused budget is acceptable when predicted gains or other constraints limit useful offers. Feasibility under predicted spend is distinct from calibration of actual spend.

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
