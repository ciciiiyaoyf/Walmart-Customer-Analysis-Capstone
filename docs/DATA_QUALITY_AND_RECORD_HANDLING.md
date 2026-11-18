# Data quality and record handling

This document describes the implemented rules, executable checks, and remaining limits of the supplied data. The historical modeling unit is one **customer-campaign assignment**, identified by `assignment_id`; the July scoring unit is one customer. Repeated customers across different campaigns are legitimate observations, not duplicate records.

## Source, lineage, and decision times

- The original `WalmartData.zip` contains 11 CSV tables and three documentation files. Its SHA-256 is `428e7023bfae583e6f211d932def9d2a3aa96332862b5af80b1bc973e83a741f`. The November 17, 2026 source-package audit reconciled all original fields to the saved clean tables, after date parsing and equivalent missing-value handling; the only removed records were 160 exact duplicate order-item rows.
- The source `README_START_HERE.md` specifies **Pacific Time** and **U.S. dollars**. `src/common.py` parses the supplied timestamps without adding timezone information; comparisons use their supplied Pacific clock values. The pipeline does not reinterpret them as UTC.
- Notebook 01 reads `data/raw/`, records schema and quality results, and writes `data/processed/clean/`. It checks raw-file hashes before and after processing. Notebook 02 builds historical features; Notebook 03 builds outcomes and joins them by `assignment_id` with `validate="1:1"`. Notebook 07 reuses the same feature builder for July.
- Historical features use `event_ts < send_ts`. July features use `event_ts <= policy_as_of_ts`, as required by the source prompt and data dictionary. The current July cutoff is June 20, 2025 at 10:00 Pacific Time. **No seven-day shift is applied to historical cutoffs.**
- A 30- or 90-day feature window includes its lower boundary and applies the appropriate upper-cutoff rule above. Events after the decision cannot become predictors merely because they exist in the extract. A later campaign may legitimately use an earlier campaign's already-observed events.

## Orders and cancellations

1. Retain every raw order for audit, including the 102 cancelled orders. The clean table adds `is_completed`; only completed orders enter purchase and financial outcomes or purchase/product-history features. Notebook 01 checks that cancelled orders have zero financial fields.
2. Keep organic preperiod orders, whose `campaign_id` is null, for eligible purchase history. They are not attributed to a campaign outcome. Notebook 01 checks that these records precede the first campaign.
3. An outcome order must match **both customer and campaign**, be completed, and satisfy `send_ts <= order_ts < send_ts + 14 days`. Notebook 03 asserts that each eligible `order_id` occurs once. It does not assign every temporally nearby purchase to the latest campaign.
4. Sum eligible orders by `assignment_id`, then join those aggregates onto the complete assignment spine. An assignment with no eligible supplied order receives `purchase_14d=0`, zero order count and financial sums, and undefined average order value. These are labels under the supplied campaign extract, not a certification of complete transactions from every possible source.
5. Use order-level amounts for financial totals. Deduplicated item lines supply product history and reconcile merchandise gross revenue and COGS; joining item lines must not multiply order revenue.

The signed profit definition is:

```text
NCM = net revenue - COGS - shipping cost - applicable refunds
net revenue = merchandise gross revenue - discounts + shipping revenue
```

Negative NCM is a valid loss and is retained. Promotion cost is a separate budget measure; subtracting the discount or waived-fee estimate again from NCM would double-count relevant costs. Free-shipping promotion cost uses an inferred fee rule, not a directly recorded waived-fee field; see Notebook 03 and the walkthrough for that assumption.

## Returns, partial refunds, and label maturity

- Keep distinct return records and link them to orders by `order_id`. Multiple partial refunds on one order are allowed; they are not duplicate purchases. Notebook 01 checks unique `return_id`, valid order links, positive refund amounts, completed-order status, and return timestamps after the order.
- For historical features, only refunds known at that decision cutoff are eligible. `src/features.py` first sums them by **decision key and order_id**, then merges once onto the order with `validate="1:1"`. Thus two refunds change margin twice but do not duplicate revenue, order count, or product quantities. Return-event counts can exceed returned-order counts; the historical return rate uses distinct refunded orders.
- `src/outcomes.py::applicable_refunds` links refunds only to eligible outcome orders and keeps separate in-window, post-window, and total-linked amounts. A refund exactly at `send_ts + 14 days` is outside the primary half-open window. Reconciliation verifies `in_window + post_window = total_linked` and checks the selected applicable amount.
- The primary `in_window` rule subtracts refunds before the 14-day outcome end. `all_linked` subtracts all linked refunds available in the selected snapshot or optional inclusive `as_of_ts`; it is a separate accounting sensitivity. The default snapshot-wide alternative is retrospective and must not be treated as an available historical label without checking timing.
- `src/diagnostics.py::build_accrued_target` provides a **45-day return-accrual sensitivity**. Purchases still come from the original 14-day purchase window. Refunds must occur before `send_ts + 45 days` and at or before the label cutoff; only rows whose full accrual window has matured are retained. It preserves the primary `ncm_14d` column. This is neither 45 days of purchases nor unlimited eventual returns.
- An as-of filter prevents using already-recorded future refunds but cannot recover refunds not yet observed. Maturity restrictions and the accounting sensitivity address a different issue from ordinary transaction-source completeness.
- Step 01 checks both each refund and the **sum of all refunds for an order** against that order's net revenue, with a $0.05 reconciliation tolerance. This prevents several individually valid-looking partial refunds from collectively exceeding the order amount. Financial fields, item quantities/prices/costs, and refund amounts must be finite. Five supplied refunds exceed merchandise gross but remain within order net revenue; they are retained with a warning. Negative NCM after costs remains valid even when total refunds do not exceed revenue.

## Multiple campaign exposures

- Notebook 01 requires unique `assignment_id` and unique `(customer_id, campaign_id)`. A person assigned in CP01 and CP02 keeps two decision rows, with separately dated features and outcomes. The canonical feature/outcome join uses `assignment_id`, not `customer_id` alone.
- The feature builder includes only assignments preceding the row cutoff and explicitly excludes the current historical assignment. `prior_emails_delivered` counts eligible delivered assignment rows; with the enforced customer-campaign uniqueness, those rows represent distinct prior delivered campaigns. It is not a count of email-open events or every prior campaign regardless of delivery.
- For `prior_offer_redemption_rate`, eligible prior commercial assignments are joined to known redeemed orders, then reduced to one maximum redemption indicator per `prior_assignment_id`. Multiple redeemed orders in one campaign therefore do not enlarge the denominator or count that exposure repeatedly. There is no separate `prior_campaign_count` predictor in the current feature schema.
- Prior email opens/clicks are event counts filtered by event time; they are not distinct exposure counts. The current campaign's post-send engagement cannot enter its historical predictors.
- Repeated observations from a person are also relevant to uncertainty: customer-cluster bootstrap procedures retain that person's repeated rows together. Treating every assignment as an independent person would exaggerate precision.

## Exact duplicates versus conflicting records

Notebook 01 removes **exact full-row duplicates only from `order_items`**: 16,306 raw rows become 16,146 clean rows. It then checks uniqueness of `(order_id, line_number)` and reconciles item gross/COGS to completed-order amounts within $0.05. Two lines with the same key but different quantities, products, prices, or costs remain a conflict and produce a failure; the code does not silently choose one version.

Other tables are checked for exact duplicates and their relevant key uniqueness; they are not blanket-deduplicated. Distinct orders, partial-refund records, email events, and a customer's separate campaign assignments must not be removed just because they share a customer or order identifier. Anonymous web sessions are retained in clean data with `is_identified=False` but excluded from identified customer features; the supplied snapshot has 220 such sessions.

## Meaningful checks on the actual tables

Notebook 01's `check()` function records counts and `PASS`, `WARN`, or `FAIL` in `outputs/tables/data_quality_summary.csv`. An early gate stops invalid schema/keys before downstream joins; if it fails, the registry contains the checks completed up to that point. The final gate rejects any remaining `FAIL` **before clean-table writes**; accepted warnings remain visible. The November 17, 2026 original-source run has **83 checks: 80 PASS, 3 WARN, zero FAIL**. These gates cover the specified checks, not every conceivable data defect.

| Named check or check family | Error it is intended to catch |
| --- | --- |
| `pk_unique:order_id`, `pk_unique:return_id`, `unique_customer_campaign_assignment` | Duplicate entities or multiple assignments masquerading as one campaign decision. |
| `pk_unique:(order_id,line_number) after dedup` | Conflicting item lines surviving exact-duplicate removal. |
| `fk:order_item.order_id`, `fk:return.order_id`, email/assignment matching | Orphan records and links to the wrong customer or campaign. |
| `campaign orders inside [send, send+14d)` and `no overlapping historical outcome windows` | Invalid supplied attribution/timing and overlapping campaign outcome windows. |
| `cancelled orders carry zero financials`, `valid order_status` | Cancelled orders carrying apparent sales or unrecognized order states. |
| `net = gross - discount + shipping_rev`, item gross/COGS reconciliation | Inconsistent financial totals and duplicate-line inflation. |
| `return_ts after order_ts`, `returns only on completed orders` | Impossible refund timing or a refund attached to an ineligible order. |
| `finite order financials`, `finite item quantities/prices/costs`, `finite refund amounts` | Infinite or missing numeric values that comparison-only checks could overlook. |
| `total refunds per order <= net_revenue` | Several partial refunds collectively exceeding the order amount. |
| `redeemed offer == assigned offer`, discount-rate checks | Promotion cost or redemption recorded under the wrong assigned action. |

Expected row counts are snapshot checks, not a rule that future source extracts must have the same size. Any legitimate new extract needs an intentional review of those expectations.

## Executable regression tests

These tests exercise behavior using controlled fixtures, including cases where an incorrect implementation would change the answer. They complement the checks on the actual tables.

`tests/test_data_quality.py` executes the actual Step 01 validation cells and failure gates. Only the input loader and output directories are substituted. Fixtures are reconstructed from the committed clean tables; 160 synthetic exact item duplicates restore the input-shape contract. These fixtures exercise processing rules without requiring the private ZIP and are separate from verification of the original raw package. Clean-table saving is not executed by the fixture harness.

The new file contains **18 passing cases across ten test functions**, covering these distinct defects and valid cases:

| Test function in `tests/test_data_quality.py` | Behavior protected |
| --- | --- |
| `test_valid_input_retains_repeat_customers_and_deduplicates_exact_items` | Valid repeat campaign customers survive, only exact duplicate item rows collapse, and inputs remain unchanged. |
| `test_duplicate_primary_keys_are_rejected` | Duplicate order, product, or return identities stop processing. |
| `test_duplicate_customer_campaign_assignments_are_rejected` | Two assignments cannot represent the same customer-campaign decision. |
| `test_conflicting_item_rows_are_rejected` | Same item key with conflicting values is rejected rather than silently deduplicated. |
| `test_broken_relationships_are_rejected` | Missing product, order, or customer references are rejected. |
| `test_email_customer_must_match_its_assignment` | An email event cannot be attached to another customer's assignment. |
| `test_financial_corruption_is_rejected` | Revenue arithmetic, cancelled-order amounts, and item COGS must reconcile. |
| `test_nonfinite_amounts_are_rejected` | Infinite order, item, and refund values are rejected. |
| `test_cumulative_refunds_cannot_exceed_order_revenue` | Partial refunds cannot evade the cap by appearing as separate records. |
| `test_return_before_order_is_rejected` | Refund timing cannot precede the linked order. |

Existing tests also protect time-dependent features, outcomes, and measurement:

| Exact pytest node | Behavior protected |
| --- | --- |
| `tests/test_feature_observation_scope.py::test_events_after_decision_cannot_change_any_predictor` | Adding future orders, refunds, assignments, sessions, and emails leaves every predictor unchanged. |
| `tests/test_feature_observation_scope.py::test_decision_relative_window_and_cutoff_boundaries_are_preserved` | Exact lower-window boundaries and historical-exclusive versus July-inclusive cutoffs. |
| `tests/test_feature_observation_scope.py::test_stale_history_stays_observed_zero_instead_of_shifting_the_window` | Old transactions cannot silently move the required recent-history window backward. |
| `tests/test_outcomes.py::test_multiple_known_refunds_do_not_duplicate_order_features` | Multiple partial refunds reduce margin without multiplying orders, revenue, or units. |
| `tests/test_outcomes.py::test_alternative_return_rules_reconcile_without_double_counting` | Early, boundary-time, and late refunds reconcile under both return rules. |
| `tests/test_outcomes.py::test_as_of_cutoff_excludes_unavailable_refunds_including_in_sensitivity` | The alternative accounting rule still excludes refunds unavailable at the cutoff. |
| `tests/test_diagnostics.py::test_accrued_target_uses_later_returns_once_and_excludes_immature_rows` | The 45-day sensitivity excludes immature rows and refunds on cancelled/out-of-window orders. |
| `tests/test_diagnostics.py::test_refund_at_or_beyond_horizon_stays_excluded_even_when_observed` | Observing a refund later does not enlarge the defined 45-day horizon. |
| `tests/test_diagnostics.py::test_verified_inactivity_requires_complete_coverage_and_known_zero` | An observed zero without verified coverage is not labeled verified inactivity. |
| `tests/test_measurement.py::test_missing_customer_outcomes_are_not_dropped_or_imputed_zero` | Future campaign measurement cannot silently discard missing customers or invent zero outcomes. |

Run the relevant fixture-based tests from the repository root:

```bash
python -m pytest -q tests/test_data_quality.py tests/test_feature_observation_scope.py tests/test_outcomes.py tests/test_diagnostics.py tests/test_measurement.py
```

For the complete test suite, use `python -m pytest -q`. To run the actual-table validation, place the 11 supplied CSVs in `data/raw/` and execute `python run_pipeline.py 01 01`; this writes DQ reports and, only after the gate passes, clean outputs. `--from-clean` starts at Notebook 02 and therefore does not rerun raw-data validation.

## Recorded verification

The complete suite passed **148 cases**, including the 18 new data-quality cases. All Step 01 code cells were separately executed in sequence using IPython in an isolated workspace against the original ZIP. Raw-file hashes were unchanged, and all 11 rebuilt clean tables matched the preceding saved values after normalizing timestamp storage units and allowing a numeric absolute tolerance of `1e-12`. The committed clean Parquet files, model artifacts, and predictions were not replaced. The newer CSV schema report records this raw run's parser dtypes, which differ from the older snapshot's storage dtypes without changing the compared values.

[The verification record](../outputs/data_quality_validation.json) contains the archive and per-file hashes, execution method, environment, check counts, clean-table comparison rules, and regression-suite result. These results concern input validation and record handling; they do not claim a new improvement in model accuracy.

## What the July zero counts establish

The source-package audit found no raw orders of any status in the scoring customers' required recent 30/90-day windows; cleaning and ID joins did not create these zeros. Their latest supplied order is January 9, 2025 at 21:42 Pacific Time. The package does not certify complete ordinary-purchase coverage through the June cutoff or explain whether the missing recent activity was intentional in the fictional case.

Accordingly, transaction predictors retain `observed_` names and the original decision-relative windows. `src/coverage.py` keeps coverage-qualified counts separate; `complete_orders_30d`/`complete_orders_90d` remain unknown when coverage is unverified. Neither changing the cutoff nor filling unknown completeness with zero repairs missing source data. Passing the tests establishes the stated processing behavior, not complete transaction coverage or future model performance.
