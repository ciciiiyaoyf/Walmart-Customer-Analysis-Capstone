"""Build the reviewable validation report from saved pipeline results."""
import json
from pathlib import Path
import pandas as pd

ROOT=Path(__file__).resolve().parents[1]
T=ROOT/"outputs/tables"


def table(frame):
    def fmt(value):
        if isinstance(value,float): return f"{value:.4f}" if pd.notna(value) else "NA"
        return str(value).replace("|","/")
    lines=["| "+" | ".join(frame.columns)+" |", "| "+" | ".join(["---"]*len(frame.columns))+" |"]
    lines += ["| "+" | ".join(fmt(v) for v in row)+" |" for row in frame.itertuples(index=False,name=None)]
    return "\n".join(lines)


def main():
    selection=json.loads((ROOT/"outputs/models/policy_selection.json").read_text())
    selected=selection["selected_method"]
    compare=pd.read_csv(T/"before_after_model_metrics.csv")
    candidates=pd.read_csv(T/"policy_candidate_selection.csv")
    policies=pd.read_csv(T/"heldout_policy_comparison.csv")
    row=policies.set_index("policy").loc[selected]
    final=pd.read_csv(T/"optimization_summary.csv").iloc[0]
    cold=pd.read_csv(T/"heldout_cold_start_evaluation.csv").query("policy == @selected")
    returns=pd.read_csv(T/"return_sensitivity_july.csv")
    costs=pd.read_csv(T/"promo_cost_metrics.csv").query("split == 'test_CP07'")
    paired=pd.read_csv(T/"heldout_policy_paired_comparisons.csv")
    data_check=json.loads((T/"before_after_data_check.json").read_text())
    text=f'''**Correction and validation report — November 12, 2026**

The code is more reliable and its performance claims are better supported. **The retained NCM model did not improve in predictive accuracy when compared with the original model refit in the same environment. Profitable personalized targeting is still not established.**

The reference snapshot is `ea950d0e9011ff0e105f5d93758ab54944a7c75f`. The original saved results are preserved in `outputs/reference/`. All changes were developed for the user's `main` branch.

**Before and after: the same prediction task and runtime**

{table(compare)}

The original saved snapshot differs slightly from the original refit in this runtime. That difference must not be attributed to the corrections: the original refit and corrected retained model have identical metrics. Primary 14-day NCM unchanged: **{data_check['primary_ncm_unchanged']}**. Feature differences across all 43 predictors: **{len(data_check['feature_differences'])}**.

RMSE remains a valid conditional-mean prediction diagnostic; MAE and R² are supporting diagnostics. They do not measure policy quality directly.

**Model selection, without choosing a winner on CP07**

Four predeclared candidates were compared: original T-learner, a 50% shrinkage version, action means, and a pooled XGBoost model with treatment as a feature. The development folds are CP04, CP05, and CP06. Each fold uses earlier campaigns only, and XGBoost tuning uses the last prior campaign as inner validation (for example, CP04 uses CP03 to tune and CP01–02 to fit candidates). Cost-model families are likewise chosen on the last prior campaign before refitting.

Replacement requires a positive lower bound for a paired development improvement, with customer-cluster bootstrap and Bonferroni adjustment across three comparisons. Otherwise the original is retained. This conservative rule is not a claim that the original is optimal.

{table(candidates)}

Selected method: **{selected}**. The selection is written to `outputs/models/policy_selection.json` before CP07 evaluation. CP07 was already inspected in the preceding audit, so this is explicitly a **retrospective held-out re-analysis**, not a new prospective test. Do not repeatedly revise the method based on CP07 and continue treating it as untouched.

**Independent evaluation of the complete constrained policy**

The evaluator now applies one action per customer, a population-scaled budget, a 60% offer cap, PCT_20 at most 15% of offered customers, and a 5% reserved NO_OFFER holdout. Models, purchase rankings, costs, and historical mix use earlier campaigns. Competitors share the same constraints and reserved controls.

Observed randomized outcomes are evaluated with doubly robust (DR) and Hájek estimators. DR uses one common reward nuisance model, fitted on prior data, for all competitors. Intervals use 2,000 customer-cluster bootstrap draws, re-estimating campaign/action frequencies inside every draw. Paired comparisons use the same sampled customers. Costs receive an analogous DR check.

The selected feasible policy's CP07 incremental NCM estimate is **${row.dr_incremental_ncm_per_customer:.2f} per customer**, with a **95% interval [${row.ci_low:.2f}, ${row.ci_high:.2f}]**. The interval includes zero. The former +$8.73 estimate evaluated an unconstrained policy offering to 87.2% of customers; it is not an apples-to-apples before/after estimate.

{table(policies[['policy','dr_incremental_ncm_per_customer','ci_low','ci_high','hajek_incremental_ncm_per_customer','matched_rows','offered_share']])}

{table(paired[['reference','paired_dr_difference','ci_low','ci_high']])}

The simple seeded random-FREE_SHIP reference performs well on CP07. These individual test intervals are nominal, not adjusted for searching over test comparisons. They should motivate new validation rather than changing the chosen model after seeing CP07. Baseline randomization uses a fixed seed; the intervals condition on that allocation realization.

Actual logging probabilities are not supplied. The primary calculation assumes unconditional randomization within each campaign and uses empirical action frequencies. The table also saves a uniform-0.25 propensity sensitivity. If assignment was stratified or used different known probabilities, supply that design instead. Intervals remain conditional on fitted models and fixed batch allocations; they do not include future campaign drift, transport uncertainty, or all retraining uncertainty. Negative endpoints for DR cost intervals reflect estimator uncertainty, not negative realized costs.

**July transport and return sensitivity materially limit confidence**

July has no prior campaign exposure, no recent purchases, and seven entirely missing feature columns (including five email/redemption features). Missingness, zero rates, ranges, and unseen categories are saved in `july_feature_support.csv`. Imputation makes scoring possible; it does not validate transport.

Comparable historical subgroups are small. The selected policy's subgroup results are:

{table(cold[['subgroup','n','matched_rows','dr_incremental_ncm_per_customer','ci_low','ci_high']])}

The negative subgroup point estimates and broad intervals are a reason for caution, not proof of either benefit or harm. Mutable customer-profile fields still require the documented static-snapshot assumption; timestamps for that history were not supplied.

The return code now reconciles early, late, applicable, and total refunds without double counting. The `all_linked` configuration passes; the default 14-day target is preserved. Multiple refund events no longer duplicate an order's feature contributions.

A full 45-day accrued-return sensitivity refits models and re-optimizes allocations on the **same mature training cohort** under both targets. It excludes CP07 from July training for this sensitivity because its 45-day labels are not mature at June 20. Future refunds are excluded. CP06 sensitivity is also evaluated after its labels mature and is labeled retrospective. The 45-day horizon is an explicit sensitivity assumption, not a newly verified business accounting rule.

{table(returns[['scenario','training_rows','projected_incremental_ncm','projected_spend','actions_changed_between_targets']])}

The two target scenarios change **{int(returns.actions_changed_between_targets.iloc[0])} of 2,776 assignments**. Their projected values use different accounting targets and must not be compared as proof that one model is better. See `return_sensitivity_cp06.csv` for outcome-based retrospective sensitivity. The deployed primary target remains 14-day NCM.

**Cost models and July projection**

Notebook 08 chooses each cost model using CP06 only, comparing XGBoost with the action mean; CP07 follows selection. FREE_SHIP selects the mean. Its CP07 RMSE is about $0.890 versus about $0.897 in the original saved XGBoost output: a small improvement on that metric, while mean cost calibration remains imperfect.

{table(costs[['action','candidate','rmse','mae','obs_mean','pred_mean','relative_mean_bias']])}

The current July allocation contains {int(final.n_FREE_SHIP)} FREE_SHIP, {int(final.n_PCT_10)} PCT_10, {int(final.n_PCT_20)} PCT_20, and {int(final.n_NO_OFFER)} NO_OFFER customers. All {int(final.holdout_customers)} controls remain NO_OFFER. Projected incremental NCM is **${final.objective_total_expected_incremental_ncm:,.2f}**, and projected promotion spend is **${final.expected_promo_spend:,.2f}**. These are model projections, not observed gains or independently proven profit. Mathematical feasibility under estimated costs does not guarantee realized spending.

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
'''
    path=ROOT/"docs/VALIDATION_REPORT.md"
    path.write_text(text,encoding="utf-8")
    print(path)


if __name__ == "__main__":
    main()
