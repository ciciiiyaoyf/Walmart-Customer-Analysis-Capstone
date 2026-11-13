# PromoPilot pipeline walkthrough

Revised November 13, 2026. This document explains the ten implemented steps and how to interpret their outputs. The accompanying **VALIDATION_REPORT.md** records the actual revision commands, before/after measurements, selected model, and remaining limitations. Its measured results take precedence over historical example numbers in older reports.

The business objective is to recommend one of NO_OFFER, PCT_10, PCT_20, or FREE_SHIP for every July customer, maximizing expected incremental 14-day net contribution margin (NCM) under budget and customer-experience constraints.

**Measured performance revision:** the two-part XGBoost model reduces CP07 RMSE from $63.3694 to $62.7385 and MAE from $27.4111 to $26.4806. Logistic average precision rises from 0.2323 to 0.2367, with small probability-quality gains but uncertainty that includes zero improvement. Separately, RANDOM_FREE_SHIP wins development policy selection. Averaging feasible random recipient sets, CP07 DR gain is $7.90 per customer versus no offer and $3.93 versus the original policy, with paired 95% interval [-$2.12, $10.44]. Policy superiority remains uncertain. The saved seed-42 allocation's more favorable $12.74 estimate depends partly on recipient luck. Full comparisons are in VALIDATION_REPORT.md.

The first correction pass repaired evaluation and accounting while retaining the original model. The later performance revision changed both models and allocation using development data, then measured their retrospective CP07 behavior. The current transaction-coverage correction adds explicit observed-history names and separate coverage diagnostics while preserving all 43 model predictor values and their order. It does not add an accuracy gain; the reported 1.00% RMSE and 3.39% MAE improvements belong to the preceding model revision.

There are three distinct outputs:

1. **Prediction diagnostics:** how accurately a model predicts observed NCM and cost.
2. **Historical policy evaluation:** what a complete feasible policy would earn, estimated from randomized observed outcomes with uncertainty.
3. **July projections:** predicted customer-action values and a constraint-feasible allocation. These are forecasts, not observed profit.

## 1. Workflow and execution

| Step | Notebook | Main saved output |
|---|---|---|
| 01 | 01_data_validation | Clean copies of the eleven source tables; data-quality summaries |
| 02 | 02_build_historical_features | Historical pre-send features, schema, transaction coverage and reference-window audit |
| 03 | 03_construct_14_day_outcomes | Historical outcomes and joined feature/outcome table |
| 04 | 04_business_analysis | Descriptive KPIs and exploration-only effect estimates |
| 05 | 05_logistic_propensity_model | Purchase-propensity model and diagnostics |
| 06 | 06_tlearner_xgboost | Expected NCM models, predictions, and diagnostics |
| 07 | 07_score_july_population | July features, coverage and reference-window audit, customer-action NCM scores |
| 08 | 08_estimate_promotion_cost | Expected costs and long customer-action value table |
| 09 | 09_milp_policy_optimization | Reserved controls, allocation, and constraint audit |
| 10 | 10_generate_outputs | Submission, projected baseline comparison, and measurement assignments |

Use Python 3.13 with the core versions pinned in requirements.txt and the same environment for the notebooks' python3 kernel. Run `python run_pipeline.py` from the repository root after supplying the eleven raw CSVs in `data/raw/`. The runner executes notebooks in sequence and stops on failure. `python run_pipeline.py --from-clean` starts at Step 02 using committed clean tables; an explicit range such as `python run_pipeline.py 06 10` is also supported.

The correction run executed Steps 02-10 from the committed cleaned snapshot because the source raw CSVs were unavailable. This is a downstream rebuild, not a revalidation of raw inputs. The older Pipeline_Guide.pdf is retained as historical material and is superseded by this walkthrough and VALIDATION_REPORT.md.

See VALIDATION_REPORT.md for the additional policy-evaluation command and exact environment used. Run `python -m pytest tests` for regression checks and `streamlit run dashboard/app.py` for the dashboard.

Each step reads saved outputs from earlier steps. Re-running a step overwrites its named artifacts; later steps must be rebuilt when an upstream definition or model changes.

## 2. Step 01: validation and cleaning

The raw inputs are customers, campaigns, historical_assignments, web_sessions, email_events, orders, order_items, products, returns, offer_catalog, and policy_scoring_population.

Checks cover identifiers, foreign keys, duplicate records, timestamps, valid offer/status values, missingness, item-to-order reconciliation, net revenue, and refund consistency. Exact duplicate order-item rows are removed before aggregation. Cancelled orders are excluded from purchase outcomes. Anonymous web sessions cannot be joined to identified customers.

Raw hashes and data-quality summaries document the inputs. A successful clean-data rebuild does not prove that its absent raw inputs were valid; the distinction is recorded in the validation report.

Organic orders with no campaign link are excluded from campaign outcomes only after the supplied data are checked to establish that these orders precede the historical campaigns. A future dataset with organic purchases inside a campaign window would require revisiting that attribution rule.

## 3. Step 02: pre-decision features

The shared feature builder is used both for historical campaigns and July. Historical events must occur **strictly before send_ts**. July events may occur **on or before policy_as_of_ts**.

| Family | Examples |
|---|---|
| Profile | Tenure, membership, customer tier, region, acquisition channel |
| Purchase | Observed prior orders, 30/90-day counts, revenue, NCM, order recency; prior offer redemption rate |
| Returns | Observed return counts, refund amount, historical return rate |
| Web | Sessions, page views, cart additions, session recency, preferred browsing category |
| Email | Prior deliveries, opens/clicks, rates, and recency |
| Product | Observed units, distinct products/categories, top purchase category |

Counts and totals describe supplied eligible events and are zero when none are observed. Such zeros do not establish complete purchase inactivity. Nineteen predictors in the purchase, return and product families use an `observed_` prefix, for example `observed_orders_90d`, `observed_days_since_last_order`, and `observed_historical_ncm_90d`. `prior_offer_redemption_rate` and other families retain their names. The model still receives the same 43 values in the same order; the internal computations and timestamp rules are unchanged.

Undefined rates, recency, and category fields remain missing for model preprocessing. Refunds known at the decision time are aggregated by decision and order before joining to purchases, avoiding duplicated order totals when an order has multiple refunds.

`src/coverage.py` produces separate observed counts, requested-window boundaries, verification status and `complete_orders_30d` / `complete_orders_90d`. A complete count is available only when an explicit all-source coverage contract covers the entire requested window and was known by the decision cutoff. `config/transaction_coverage.json` currently contains no verified intervals because none were supplied. Complete counts consequently remain missing for historical training rows as well as July. Coverage metadata are not additional model predictors, and a last observed order timestamp is never treated as a coverage watermark.

The historical outputs include `outputs/tables/historical_transaction_coverage.csv` and `historical_organic_reference_window.csv`. The latter describes only a fixed old organic reference interval, not a corrected recent-feature window. Step 07 produces the corresponding July audits.

Tests inspect event eligibility and inject future events to confirm that features do not change. These tests apply to timestamped events. Profile attributes such as tier and membership lack effective-date histories; their historical availability remains an assumption requiring source provenance or sensitivity analysis. An event cutoff cannot establish that a mutable profile snapshot was available in the past.

## 4. Step 03: outcomes and returns

An eligible outcome order is completed, belongs to the assignment's customer and campaign, and occurs in the half-open interval from send_ts through send_ts + 14 days.

The binary purchase outcome records whether any eligible order exists. Financial fields aggregate eligible orders to the assignment level, including assignments with no purchases.

**NCM = net revenue - COGS - shipping cost - applicable refunds.**

Net revenue already includes merchandise discounts and shipping revenue. Subtracting a promotion-cost estimate again would double-count relevant costs; the optimizer instead uses promotion cost as a separate budget constraint.

### Refund accounting

Refunds are linked by order and separated into amounts arriving inside and after the outcome window. The default **in_window** target measures the stated 14-day financial outcome. The **all_linked** alternative considers every linked refund available under the chosen as-of rule.

The corrected reconciliation keeps early, late, and total refunds distinct. It does not add late refunds to an already all-inclusive amount. Regression tests include an order with both early and late refunds.

The original saved data contain approximately $15.4K of in-window refunds and $83.8K arriving later. That makes accounting sensitivity material. The sensitivity compares 14-day and 45-day accrued-return targets on identical mature cohorts while retaining the selected policy rule. The current random allocation changes zero actions, but its forecast increment changes from $124.30 to -$569.05. This sign reversal remains a limitation even though allocation is stable. The bounded 45-day target is distinct from every eventual linked refund.

When later refunds are used for model training or validation, labels must be mature and observable at the relevant fitting/decision cutoff. Some supplied returns occur after the July cutoff. The 45-day July sensitivity excludes CP07 because its labels have not matured by that cutoff. Retrospective policy sensitivity is evaluated on CP06 with mature training labels; it is not a new untouched test. The validation report identifies the observed changes and remaining limitations.

### Promotion cost

PCT offers use the discount on redeemed orders. NO_OFFER costs zero. FREE_SHIP uses an inferred waived shipping fee: observed baskets support a paid/free threshold between $74.99 and $75.08 and a $5.99 fee. This is an assumption about the tariff, not an independently recorded waived-fee field.

## 5. Step 04: descriptive and causal analysis

Descriptive KPIs include conversion, average order value, redemption, promotion spend, and NCM by offer, campaign, tenure/value, and behavior.

Normal business_rule assignments were targeted. Their raw differences across offers combine customer selection and offer effects; they cannot be interpreted as randomized treatment effects. Exploration rows support randomized comparisons, subject to the actual assignment design, treatment consistency, and population coverage.

The revised pooled-effect uncertainty accounts for customers appearing in multiple campaigns. In the original data, 7,527 exploration assignments represent 5,672 customers; 1,585 customers appear more than once. Treating all assignments as independent omits this dependence. Campaign composition and multiple exploratory subgroup comparisons also matter.

A positive point estimate is not a demonstrated positive effect. In the original unclustered analysis, FREE_SHIP's overall estimate was +$3.56 with a 95% interval of -$0.89 to +$8.01. The under-one-year and high-value PCT_10 intervals also crossed zero. Those figures are historical diagnostics, not current clustered estimates or proof of profitable segments. Use the regenerated effect tables for current intervals.

## 6. Step 05: purchase propensity

The project requires a purchase-propensity baseline; it does not mandate logistic regression as the algorithm. Logistic regression is the implemented baseline. Preprocessing is fitted inside each training fold. Rolling CP04-CP06 development compares raw and transformed predictors, compact interactions, regularization, and nested temporal sigmoid calibration. Selection maximizes mean average precision with log-loss and Brier safeguards. Calibration is retained only if it improves both probability scores on development data. The frozen winner is refitted on CP01-CP06 and evaluated on CP07, then refitted for deployment.

The selected model uses signed-log transformations and compact interactions with C = 0.01 and no additional calibration. CP07 ROC-AUC changes from 0.658743 to 0.658879, average precision from 0.232325 to 0.236708, log loss from 0.371117 to 0.370689, and Brier score from 0.109704 to 0.109558. These gains are small. The paired 95% interval for average-precision change is [-0.00445, 0.01321], so a reliable large improvement is not established.

The score predicts purchasing under the historical mixture of promotional actions. It does not estimate the probability of purchase under NO_OFFER, and it does not measure how much a promotion changes behavior.

ROC-AUC, precision-recall, log loss, Brier score, calibration, and lift describe this predictive baseline. They are not substitutes for profit-based evaluation of the promotion policy.

## 7. Step 06: expected NCM and model comparison

The original T-learner fits one action-specific XGBoost model on exploration rows to estimate expected NCM under each action. Incremental NCM subtracts the NO_OFFER prediction.

Squared-error fitting is appropriate for the conditional mean needed by an expected-profit objective. RMSE, MAE, R², and action-level mean calibration remain useful diagnostics. MSE, RMSE, and R² rank predictions equivalently on the same sample. Fitting absolute error instead would target a conditional median, which need not represent expected profit in a zero-heavy outcome distribution.

The fair same-runtime original refit has CP07 RMSE $63.3694, MAE $27.4111, and R² 0.0059. The new selected model achieves $62.7385, $26.4806, and 0.0256 respectively: modest improvements, with realized individual profit still difficult to predict.

Matching fitted models' average effects to averages from the same training data is an in-sample plausibility check. It does not validate individualized effects.

### Revised model selection

Fourteen fixed predictive configurations cover the original T-learner, pooled XGBoost, a shared baseline with action offsets, and two-part purchase/margin models. Every CP04-CP06 fold fits earlier campaigns only. Minimum pooled factual RMSE selects HURDLE_XGB_HALF; no CP07 result chooses the candidate.

About 88% of exploration outcomes are zero because no purchase occurred. The selected estimator multiplies pooled XGBoost purchase probability by a conditional purchaser-margin forecast, blended equally with each training arm's mean purchaser margin. This shares information across arms and reduces noise from separate small action samples. Negative purchaser margins are preserved; future realized purchase is a training label, never an input at scoring time.

Development RMSE falls from 62.9717 to 62.3674. On CP07, RMSE improves 1.00% and MAE 3.39%. A paired customer bootstrap gives old-minus-new MSE 79.57, 95% interval [17.45, 140.35], and MAE 0.931, interval [0.452, 1.423]. These intervals hold fits fixed and omit search/retraining and future-wave uncertainty. Three arm-specific RMSEs improve; NO_OFFER worsens slightly. The evidence supports a modest predictive gain, not uniform improvement everywhere.

Allocation selection is separate: 21 feasible rules include model optimizers, shrunk segment estimates, and simple baselines. Maximum pooled CP04-CP06 DR value selects RANDOM_FREE_SHIP (3.8176 per customer versus -1.9076 for the original feasible policy). Uncertainty is reported rather than used as a requirement that every replacement prove a positive adjusted lower bound. Selected development performance is subject to search optimism.

CP07 evaluation follows the frozen choices. CP07 has already been inspected in the original analysis and audit, so these calculations are retrospective evidence, not a fresh untouched confirmatory test.

### Evaluate the complete feasible policy

The original notebook's normalized-IPW evaluation used unconstrained argmax recommendations. It offered promotions to about 87% of CP07 customers and therefore did not evaluate the final 60%-capped policy.

The revised evaluation applies the allocation constraints, budget scaled to population size, and reserved controls before evaluating actions against randomized outcomes. It reports doubly robust and Hájek estimates, matched rows/effective sample size, and uncertainty in incremental value versus NO_OFFER. Candidate comparisons should use paired uncertainty because policies are evaluated on the same customers.

The source tables do not include logged assignment probabilities. Evaluation uses empirical action frequencies within campaign, assuming unconditional randomization within each campaign; a uniform 25%-per-action sensitivity is also reported. These assumptions should be checked against the actual exploration design, especially if assignment was stratified or adaptive.

The customer-cluster bootstrap retains each sampled customer's campaign records and re-estimates campaign action frequencies in every draw. Subgroup effects use the full campaign's logging population rather than estimating frequencies only inside the subgroup. Intervals are conditional on fitted outcome/cost models and fixed allocations; they do not include model-refitting uncertainty or future campaign drift. Outcome predictions are fitted without evaluated outcomes. None of these procedures eliminates distribution shift or proves that July's population is represented.

For the fixed seed-42 allocation, RANDOM_FREE_SHIP has CP07 DR incremental NCM of $12.74 per customer, compared with $3.97 for the original feasible policy. The paired difference is $8.77, 95% interval [$0.36, $18.33]. However, averaging all feasible random recipient sets with the same reserved controls lowers the estimate to $7.90, interval [$1.36, $15.35]. The averaged paired difference versus the original is $3.93, interval [-$2.12, $10.44]. The saved seed is favorable; superiority over the original remains uncertain once recipient randomization is averaged. This diagnostic does not change selection or establish future July gains. Conversion or generic uplift ranking alone cannot answer the four-action, profit-and-constraints question.

## 8. Step 07: July scoring and support

The shared feature builder reconstructs the same feature definitions at each customer's policy_as_of_ts. Cutoff eligibility, future-event perturbation, schema parity, and one-row-per-customer checks are retained. The July decision cutoff is **June 20, 2025 at 10:00**, with these unchanged decision-relative windows:

| Feature window | Start, inclusive | End, inclusive for July scoring |
|---|---|---|
| Prior 90 days | March 22, 2025 at 10:00 | June 20, 2025 at 10:00 |
| Prior 30 days | May 21, 2025 at 10:00 | June 20, 2025 at 10:00 |

These boundaries were already mathematically correct. Moving the decision cutoff to the campaign send, moving the lookback to January, or changing the 14-day target would answer different questions rather than repair missing source coverage.

July is a distinct population: none of its 2,776 customers appears in the historical campaign assignments. The supplied ordinary (`organic_preperiod`) transactions end with an observed event on January 9, 2025, while later supplied orders are campaign-attributed. As a result, all July customers have zero **observed** completed orders in the recent 30/90-day windows. The latest available July-cohort purchase is approximately 161.5 days before the decision cutoff. January 9 is an observed transaction date, not a certified coverage endpoint; the data do not establish how many ordinary transactions occurred after it. Complete recent purchase counts are unknown, not verified zero.

`outputs/tables/july_transaction_coverage.csv` records the requested windows, `observed_orders_30d` / `observed_orders_90d`, source-specific observed counts, coverage status and `complete_orders_30d` / `complete_orders_90d`. With the empty verified-interval configuration, every July complete count remains missing. The same rule applies to the historical coverage output, so campaign-attributed orders are never treated as proof of complete ordinary-purchase history.

For context, `july_organic_reference_window.csv` summarizes **[October 12, 2024, January 10, 2025)**, including the start and excluding January 10. It contains 279 observed completed organic orders across 256 July customers. Across the full supplied history, 510 July customers have at least one completed order. The fixed reference interval is stale, is not certified complete, and is an audit output only: it is neither a recent-90-day substitute nor a new predictor. The equivalent historical audit uses the same interval and exposes reference availability and age.

Imputation allows models to produce scores; it does not verify purchase-history completeness or transfer of treatment effects. The support analysis identifies customers with no campaign history and no observed recent purchases. That is a description of recorded features, not a verified inactive-customer segment. The coverage correction improves interpretation and auditability without changing the 43 model predictor values, so it cannot be credited with the preceding model revision's accuracy gains.

Outputs contain predicted purchase propensity, expected NCM under all four actions, incremental NCM, and an unconstrained best-action diagnostic. That diagnostic is not the deployed allocation.

## 9. Step 08: expected promotion spend

The cost models predict mean promotion cost per customer and commercial action on exploration data. Step 08 compares XGBoost with an action-mean predictor on CP06 for each action, selects the lowest RMSE (mean wins ties), fixes that choice before CP07 evaluation, and then refits for July. NO_OFFER cost is fixed at zero; negative cost predictions are clipped to zero. NCM is allowed to be negative.

Cost evaluation reports absolute and relative mean bias as well as individual prediction error, especially for selected recipients. A low cost RMSE can conceal systematic underprediction when nonzero costs are rare. Historical validation and July budget slack are different facts: a feasible predicted budget does not establish accurate costs.

Historical policy folds apply the same cost-family selection rule on their last prior campaign, then refit on earlier data. The final July choices are XGBoost for PCT_10/PCT_20 and the action mean for FREE_SHIP. Test calibration remains imperfect and is reported separately from the small FREE_SHIP RMSE improvement.

The customer-action table is the common input to the production allocator and the backtesting allocator. It contains predicted NCM, incremental NCM, expected cost, propensity, and identifiers.

## 10. Step 09: constrained allocation

Reserve at least 5% of customers as random controls before optimization, using a fixed seed and deterministic customer ordering. Controls are forced to NO_OFFER.

| Constraint | Rule |
|---|---|
| Assignment | Exactly one valid action per customer |
| Expected spend | At most $62,000 for the July population |
| Commercial offers | At most floor(60% × number of customers) |
| PCT_20 | At most 15% of all commercial-offer recipients |
| Controls | At least ceil(5% × number of customers), all NO_OFFER |

The model-optimizer candidate sums estimated incremental NCM. Subtracting each customer's NO_OFFER value is valid because it is constant across alternative allocations for that customer. Production follows the separately selected feasible policy: currently RANDOM_FREE_SHIP, with seeded random customer ordering, the same controls, and the same budget/offer constraints. The XGBoost predictive champion supplies forecasts without forcing the allocation to follow its noisy individualized ranking.

A shared SciPy/HiGHS allocator replaces PuLP and the machine-specific CBC path. Tests cover binding budget, the PCT_20 denominator, reserved controls, deterministic behavior, and assignment uniqueness. The saved audit independently checks the produced allocation.

The budget does not have to be exhausted. A positive predicted gain is also uncertain; the optimizer's objective is not a guarantee of realized profit.

## 11. Step 10: outputs and comparisons

The submission has exactly four fields: customer_id, recommended_action, estimated_incremental_ncm, estimated_promo_cost. Output checks enforce unique complete coverage, valid actions, finite values, nonnegative costs, and all policy constraints.

The projected baselines use the same customers, constraints, and reserved controls:

1. ALL_NO_OFFER.
2. PROPENSITY_PCT10: rank by purchase propensity and assign PCT_10 subject to budget and offer cap.
3. HISTORICAL_MIX: approximate the historical commercial mix as closely as feasible.
4. RANDOM_FREE_SHIP: a simple feasible randomized FREE_SHIP policy to challenge claims that personalization adds value.

The model optimizer's projected advantage is expected under the objective it maximizes. It is retained as a comparator, while the submission follows the development-selected policy. All July dollar values remain projections. Historical outcome-based comparisons and their intervals appear separately in heldout_policy_comparison.csv and the validation report.

The dashboard separates July projections from held-out evaluation and shows uncertainty where available. An interval crossing zero is inconclusive evidence of improvement, not proof of no effect and not proof of profit.

## 12. July measurement design

The reserved holdout is randomized at the policy-arm level. The correct comparison is **all customers assigned to the policy arm**, including those the policy assigns NO_OFFER, against the randomized NO_OFFER control arm.

Comparing only offered customers against the random holdout would mix targeting differences with intervention effects. For effects among particular target customers or for specific offers, define those groups before randomization and use a suitable randomized design within them.

Preserve the randomized arm assignment separately from the final action. Predefine the outcome/return window, estimand, analysis, confidence interval, power, and monitoring rules. Meeting the 5% minimum control requirement does not guarantee enough power for a small noisy dollar lift.

## 13. Reading the final recommendation

The validation report answers whether the correction improved predictions, measured policy value, uncertainty, or only reliability. These are separate outcomes. A lower July projection after correcting evaluation is not necessarily a worse system; it can be a more honest estimate.

Useful conclusions require:

- Feasible allocation and calibrated expected spend.
- Earlier-only fitting and clear label availability.
- Outcome-based policy comparisons with uncertainty.
- Evidence that target customers are represented.
- Stability to return accounting and model uncertainty.
- A prospective randomized test before claiming realized July gains.

Do not claim that all customers would buy anyway, that FREE_SHIP is universally profitable, or that personalization works solely because the fitted model assigns positive gains. State the observed evidence, its uncertainty, and the limits of the population and time period studied.
