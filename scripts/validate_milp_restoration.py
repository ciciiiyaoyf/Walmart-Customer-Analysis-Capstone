"""Verify restored MILP against the last experiment commit and original scores."""
from __future__ import annotations

import argparse
import hashlib
import io
import json
import math
from pathlib import Path
import subprocess
import sys

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.policy import ACTIONS, ALLOCATOR_NAME, assert_feasible, optimize_policy

REFERENCE = '6af5bac2a7cccb1011d620857c268253dc7143ce'
RETIRED_PREFIXES = ('outputs/experiments/allocators/', 'outputs/reference/allocator_before_20261118/', 'docs/figures/')
RETIRED_FILES = {
    'src/allocation_annealing.py', 'src/allocation_cpsat.py',
    'tests/test_allocation_annealing.py', 'tests/test_allocation_cpsat.py',
    'requirements-optimizer-experiments.txt', 'scripts/compare_allocation_methods.py',
    'scripts/validate_allocator_revision.py', 'outputs/allocator_revision_validation.json',
    'outputs/tables/greedy_policy_detailed.csv', 'outputs/tables/optimized_policy_detailed.csv',
    'docs/ALLOCATION_METHOD_COMPARISON.md',
    'docs/Pipeline_Guide.pdf', 'docs/make_diagrams.py',
    'notebooks/09_greedy_policy_allocation.ipynb',
}


def git_bytes(*args):
    return subprocess.check_output(['git', '-c', f'safe.directory={ROOT.as_posix()}',
                                    *args], cwd=ROOT)


def old_bytes(relative):
    return git_bytes('show', f'{REFERENCE}:{relative}')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--tests-passed', type=int)
    parser.add_argument('--dashboard-passed', action='store_true')
    args = parser.parse_args()
    previous = json.loads(old_bytes('outputs/experiments/allocators/comparison.json'))
    hashes = {p: h for p, h in previous['protected_file_hashes'].items()
              if p.endswith('.joblib') or p in {'outputs/tables/july_customer_action_values.csv',
                  'data/processed/july_features.parquet', 'outputs/tables/measurement_holdout.csv'}}
    for path, expected in hashes.items():
        if hashlib.sha256((ROOT / path).read_bytes()).hexdigest() != expected:
            raise AssertionError(f'Model/forecast/control input unexpectedly changed: {path}')
    tables = ROOT / 'outputs/tables'
    values = pd.read_csv(tables / 'july_customer_action_values.csv')
    hold = pd.read_csv(tables / 'measurement_holdout.csv').customer_id.to_numpy()
    old_hold = pd.read_csv(io.BytesIO(old_bytes('outputs/tables/measurement_holdout.csv'))).customer_id.to_numpy()
    np.testing.assert_array_equal(hold, old_hold)
    old = pd.read_csv(io.BytesIO(old_bytes('outputs/tables/selected_policy_detailed.csv'))).set_index('customer_id').sort_index()
    current = pd.read_csv(tables / 'selected_policy_detailed.csv').set_index('customer_id').sort_index()
    policy = optimize_policy(values, holdout_ids=hold)
    assert policy.attrs['allocator'] == ALLOCATOR_NAME == 'scipy_highs_milp'
    pd.testing.assert_series_equal(current.recommended_action, policy.set_index('customer_id').recommended_action)
    chosen = current.reset_index()[['customer_id', 'recommended_action']].merge(values,
        left_on=['customer_id', 'recommended_action'], right_on=['customer_id', 'action'], validate='one_to_one')
    assert_feasible(chosen, holdout_ids=hold, expected_customers=values.customer_id.unique())
    for col in ['predicted_ncm', 'estimated_incremental_ncm', 'estimated_promo_cost']:
        np.testing.assert_allclose(current[col], chosen.set_index('customer_id').sort_index()[col], rtol=0, atol=1e-10)
    expected_submission = current.reset_index()[['customer_id', 'recommended_action', 'estimated_incremental_ncm', 'estimated_promo_cost']]
    # Step 10 intentionally exports both monetary columns to four decimals.
    expected_submission = expected_submission.round({'estimated_incremental_ncm': 4, 'estimated_promo_cost': 4})
    for path in ['policy_submission.csv', 'outputs/final/policy_submission.csv']:
        saved = pd.read_csv(ROOT / path).sort_values('customer_id').reset_index(drop=True)
        pd.testing.assert_frame_equal(saved, expected_submission, check_exact=False, rtol=0, atol=1e-10)
    matrix = values.pivot(index='customer_id', columns='action', values='estimated_incremental_ncm')[ACTIONS]
    cap = math.floor(.6 * len(matrix))
    relaxed = matrix.drop(index=hold).max(axis=1).clip(lower=0).sort_values(ascending=False)
    bound = math.fsum(relaxed.iloc[:cap])
    objective = math.fsum(chosen.estimated_incremental_ncm)
    before_history = pd.read_csv(io.BytesIO(old_bytes('outputs/tables/heldout_policy_comparison.csv')))
    after_history = pd.read_csv(tables / 'heldout_policy_comparison.csv')
    historical_columns = ['policy', 'dr_incremental_ncm_per_customer', 'ci_low', 'ci_high']
    chosen_model = 'HURDLE_XGB_HALF'
    selection = json.loads((ROOT / 'outputs/models/policy_selection.json').read_text())
    if selection['allocator'] != ALLOCATOR_NAME or selection['selected_policy'] != chosen_model:
        raise AssertionError('Deployment metadata is inconsistent with restored MILP')
    for path in [ROOT / 'outputs/experiments/policy/development_fold_summary.csv',
                 ROOT / 'outputs/experiments/policy/retrospective_cp07_summary.csv']:
        frame = pd.read_csv(path)
        if not set(frame.allocation_algorithm).issubset({ALLOCATOR_NAME, 'feasible_baseline_rule'}):
            raise AssertionError('Historical evaluation still contains an obsolete allocation algorithm')
    old_paths = git_bytes('ls-tree', '-r', '--name-only', REFERENCE).decode('utf-8').splitlines()
    retired = sorted(p for p in old_paths if p in RETIRED_FILES or p.startswith(RETIRED_PREFIXES))
    remaining = [p for p in retired if (ROOT / p).exists()]
    if remaining:
        raise AssertionError(f'Retired files still present: {remaining}')
    # Keep a compact experiment conclusion; code and detailed runs remain in Git history.
    comparisons = [{k: r[k] for k in ['method', 'incremental_ncm', 'changed_assignments', 'all_constraints_pass']}
                   for r in previous['results'] if r['method'].startswith(('CP_SAT', 'SIMULATED_ANNEALING'))]
    result = dict(date='2026-11-19', comparison_commit=REFERENCE, allocator=ALLOCATOR_NAME,
        selected_policy=chosen_model, validation_selected_policy=selection['validation_selected_policy'],
        forecast_models_refitted=False, unchanged_model_input_hash_count=len(hashes), unchanged_model_input_hashes=hashes,
        submission_monetary_rounding_digits=4,
        controls_unchanged=True, control_count=len(hold), all_constraints_pass=True,
        projected_incremental_ncm=objective, expected_promo_cost=math.fsum(chosen.estimated_promo_cost),
        action_counts={a: int(chosen.recommended_action.eq(a).sum()) for a in ACTIONS},
        difference_vs_previous_greedy=objective-math.fsum(old.estimated_incremental_ncm),
        action_changes_vs_previous_greedy=int(current.recommended_action.ne(old.recommended_action).sum()),
        solver=policy.attrs,
        original_dollar_upper_bound=bound, bound_gap=bound-objective,
        reaches_bound=math.isclose(bound, objective, rel_tol=0, abs_tol=1e-9),
        bound_explanation='Sum the largest positive per-customer maximum gains among noncontrols under the offer cap, relaxing budget and PCT20 limits. A feasible allocation attaining this bound is optimal for the original saved forecasts to numerical precision.',
        positive_gain_customers={a: int(matrix[a].gt(0).sum()) for a in ACTIONS},
        earlier_alternative_experiments=dict(source_commit=REFERENCE, results=comparisons,
            interpretation='CP-SAT matched the allocation; simulated annealing retained its starting greedy incumbent. Neither increased projected NCM. Full experiment code and records remain in this Git commit.'),
        historical_selected_policy_before=before_history.loc[before_history.policy.eq(chosen_model), historical_columns].to_dict('records'),
        historical_selected_policy_after=after_history.loc[after_history.policy.eq(chosen_model), historical_columns].to_dict('records'),
        execution=dict(historical_policy_search='CP04-CP06 development and retrospective CP07 rerun using MILP; 2000 bootstrap draws',
            fully_executed_notebooks=['07', '09', '10'],
            notebook_06='Presentation refreshed from rerun policy evaluation and unchanged saved profit models; unchanged component-diagnostic outputs retained',
            forecast_model_artifacts='All existing fitted profit, purchase-propensity, and promotion-cost model files retained unchanged'),
        removed_tracked_file_count=len(retired), removed_tracked_files=retired,
        evidence_scope='Frozen July projections and retrospective CP07 evaluation; no observed July gains and no claim that optimal predictions validate the model.')
    if args.tests_passed is not None:
        result['tests'] = dict(command='python -m pytest -q', passed=args.tests_passed)
    if args.dashboard_passed:
        result['dashboard'] = dict(app_test_passed=True, exceptions=0)
    target = ROOT / 'outputs/milp_restoration_validation.json'
    target.write_text(json.dumps(result, indent=2, default=lambda x: x.item() if isinstance(x, np.generic) else str(x))+'\n', encoding='utf-8')
    print(json.dumps({k: result[k] for k in ['allocator', 'projected_incremental_ncm', 'expected_promo_cost',
        'action_counts', 'bound_gap', 'action_changes_vs_previous_greedy', 'removed_tracked_file_count']}, indent=2))


if __name__ == '__main__':
    main()
