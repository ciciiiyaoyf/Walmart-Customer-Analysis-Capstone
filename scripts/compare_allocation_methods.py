"""Compare allocation algorithms on frozen July forecasts, without retraining.

Writes only outputs/experiments/allocators and docs/ALLOCATION_METHOD_COMPARISON.md.
The deployed allocator, submission, models, controls, and scores are hash checked.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import math
import platform
from pathlib import Path
import subprocess
import sys
from time import perf_counter

# Load the optional solver before other native libraries on Windows/Anaconda.
from ortools.sat.python import cp_model  # noqa: F401
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.allocation_annealing import solve_annealing_policy
from src.allocation_cpsat import solve_cpsat_policy
from src.policy import ACTIONS, assert_feasible, optimize_policy

CONSTRAINTS = dict(budget=62000, max_offer_share=.6, max_pct20_share=.15)


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def json_default(value):
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    raise TypeError(f"Cannot serialize {type(value)}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--iterations', type=int, default=30000)
    parser.add_argument('--restarts', type=int, default=4)
    parser.add_argument('--seeds', type=int, nargs='+', default=[7, 42, 2026])
    parser.add_argument('--time-limit', type=float, default=30)
    args = parser.parse_args()
    out = ROOT / 'outputs/experiments/allocators'
    out.mkdir(parents=True, exist_ok=True)
    protected = [ROOT / p for p in [
        'src/policy.py', 'config/allocation_policy.json', 'policy_submission.csv',
        'outputs/final/policy_submission.csv', 'outputs/tables/july_customer_action_values.csv',
        'outputs/tables/measurement_holdout.csv', 'outputs/tables/measurement_assignment.csv',
        'outputs/tables/selected_policy_detailed.csv', 'data/processed/july_features.parquet',
        'outputs/models/policy_selection.json',
    ]] + sorted((ROOT / 'outputs/models').glob('*.joblib'))
    hashes = {str(p.relative_to(ROOT)).replace('\\', '/'): digest(p) for p in protected}
    v = pd.read_csv(ROOT / 'outputs/tables/july_customer_action_values.csv')
    hold = pd.read_csv(ROOT / 'outputs/tables/measurement_holdout.csv').customer_id.tolist()
    current = pd.read_csv(ROOT / 'outputs/tables/selected_policy_detailed.csv')
    ids = sorted(v.customer_id.unique())
    baseline_actions = current.set_index('customer_id').recommended_action.sort_index()
    cap = math.floor(CONSTRAINTS['max_offer_share'] * len(ids))
    matrix = v.pivot(index='customer_id', columns='action', values='estimated_incremental_ncm')[ACTIONS]
    best = matrix.max(axis=1)
    eligible_best = best.drop(index=hold).clip(lower=0).sort_values(ascending=False)
    upper_bound = math.fsum(eligible_best.iloc[:cap])
    records, diagnostics = [], {}

    def record(name, policy, elapsed, save=True):
        assert_feasible(policy, **CONSTRAINTS, holdout_ids=hold, expected_customers=ids)
        # Revalue every assignment using the unchanged, original-dollar input table.
        chosen = policy[['customer_id', 'recommended_action']].merge(
            v, left_on=['customer_id', 'recommended_action'],
            right_on=['customer_id', 'action'], validate='one_to_one')
        chosen = chosen.sort_values('customer_id').reset_index(drop=True)
        assert_feasible(chosen, **CONSTRAINTS, holdout_ids=hold, expected_customers=ids)
        actions = chosen.set_index('customer_id').recommended_action
        gain = math.fsum(chosen.estimated_incremental_ncm)
        row = dict(method=name, incremental_ncm=gain,
                   promotion_cost=math.fsum(chosen.estimated_promo_cost),
                   difference_vs_deployed=gain - math.fsum(current.estimated_incremental_ncm),
                   changed_assignments=int(actions.ne(baseline_actions).sum()),
                   all_constraints_pass=True, elapsed_seconds=elapsed,
                   original_upper_bound=upper_bound, original_upper_bound_gap=upper_bound-gain,
                   reaches_original_upper_bound=math.isclose(gain, upper_bound, rel_tol=0, abs_tol=1e-9))
        row.update({a: int(chosen.action.eq(a).sum()) for a in ACTIONS})
        records.append(row)
        diagnostics[name] = dict(policy.attrs)
        if save:
            chosen[['customer_id', 'recommended_action', 'estimated_incremental_ncm',
                    'estimated_promo_cost']].to_csv(out / f'{name.lower()}_assignments.csv', index=False)
        print(json.dumps(row, default=json_default), flush=True)

    record('DEPLOYED_GREEDY', current, None, save=False)
    old = pd.read_csv(ROOT / 'outputs/reference/allocator_before_20261118/milp_policy_detailed.csv')
    record('PREVIOUS_MILP_REFERENCE', old, None, save=False)
    start = perf_counter()
    greedy = optimize_policy(v, **CONSTRAINTS, holdout_ids=hold)
    record('GREEDY_RERUN', greedy, perf_counter()-start, save=False)
    for scale in [1_000_000, 100_000_000]:
        start = perf_counter()
        p = solve_cpsat_policy(v, **CONSTRAINTS, holdout_ids=hold,
                              scale=scale, time_limit=args.time_limit, seed=42)
        record(f'CP_SAT_SCALE_{scale}', p, perf_counter()-start)
    for seed in args.seeds:
        start = perf_counter()
        p = solve_annealing_policy(v, **CONSTRAINTS, holdout_ids=hold, seed=seed,
                                  iterations=args.iterations, restarts=args.restarts)
        record(f'SIMULATED_ANNEALING_SEED_{seed}', p, perf_counter()-start)

    distribution = []
    for scope, frame in [('all_customers', v), ('eligible_noncontrols', v[~v.customer_id.isin(hold)])]:
        for action, g in frame.groupby('action').estimated_incremental_ncm:
            distribution.append(dict(scope=scope, action=action, customers=len(g),
                positive=int(g.gt(0).sum()), negative=int(g.lt(0).sum()),
                minimum=float(g.min()), median=float(g.median()), mean=float(g.mean()), maximum=float(g.max())))
    pd.DataFrame(distribution).to_csv(out / 'offer_gain_diagnostics.csv', index=False)
    pd.DataFrame(records).to_csv(out / 'method_comparison.csv', index=False)
    after = {str(p.relative_to(ROOT)).replace('\\', '/'): digest(p) for p in protected}
    if after != hashes:
        raise AssertionError('An experiment changed a protected deployment/model/input file')
    revision = subprocess.check_output(['git', '-c', f'safe.directory={ROOT.as_posix()}',
                                        'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip()
    result = dict(date='2026-11-19', starting_commit=revision,
        scope='Frozen July model projections only; no new causal performance evaluation or model fitting',
        constraints=CONSTRAINTS, customers=len(ids), fixed_controls=len(hold), offer_cap=cap,
        deployment_changed=False, protected_files_unchanged=True, protected_file_hashes=hashes,
        settings=vars(args), experiment_source_hashes={str(p.relative_to(ROOT)).replace('\\', '/'): digest(p)
            for p in [Path(__file__).resolve(), ROOT / 'src/allocation_cpsat.py', ROOT / 'src/allocation_annealing.py']},
        versions={'python': platform.python_version(),
            **{p: importlib.metadata.version(p) for p in ['numpy', 'pandas', 'ortools']}},
        upper_bound_certificate=dict(value=upper_bound,
            construction='Sum the largest positive per-customer maximum gains among noncontrols, limited by the offer cap; relax budget and PCT20-share constraints.',
            meaning='Any original feasible assignment is no better than this bound. A feasible assignment attaining it is globally optimal for these fixed forecasts and controls, within floating-point precision.',
            baseline_reaches_bound=records[0]['reaches_original_upper_bound'],
            comparison_tolerance=1e-9,
            unique_best_action_counts=matrix.idxmax(axis=1).value_counts().to_dict(),
            customers_with_tied_best_actions=int(matrix.eq(matrix.max(axis=1), axis=0).sum(axis=1).gt(1).sum()),
            last_selected_gain=float(eligible_best.iloc[cap-1]),
            first_excluded_gain=float(eligible_best.iloc[cap])),
        results=records, solver_diagnostics=diagnostics, offer_diagnostics=distribution,
        caveat='An optimum under saved predictions does not establish that the predictions, causal effects, or real July profit are correct. CP-SAT scaled-model certificates are separately labeled; the original-dollar certificate above does not rely on scaled costs.')
    (out / 'comparison.json').write_text(json.dumps(result, indent=2, default=json_default)+'\n', encoding='utf-8')
    table = ['| Method | Projected incremental NCM | PCT10 | PCT20 | FREE_SHIP | NO_OFFER | Changes vs deployed |',
             '|---|---:|---:|---:|---:|---:|---:|']
    for r in records:
        table.append(f"| {r['method']} | ${r['incremental_ncm']:,.8f} | {r['PCT_10']} | {r['PCT_20']} | {r['FREE_SHIP']} | {r['NO_OFFER']} | {r['changed_assignments']} |")
    if records[0]['reaches_original_upper_bound']:
        certificate_text = ('The deployed policy attains this bound within $0.000000001 while satisfying all original constraints. '
            'Therefore, for these exact forecasts and fixed controls, the existing allocation is globally optimal to numerical precision. '
            'This instance-specific proof does not make greedy a generally optimal algorithm. Changing the algorithm cannot produce a larger objective under the unchanged inputs.')
    else:
        certificate_text = (f"The deployed policy is ${records[0]['original_upper_bound_gap']:,.10f} below this relaxed bound. "
            'This check does not establish global optimality; a loose bound does not prove that a better feasible allocation exists.')
    universal_free = bool(matrix['FREE_SHIP'].gt(0).all() and
        matrix['FREE_SHIP'].gt(matrix[['NO_OFFER', 'PCT_10', 'PCT_20']].max(axis=1)).all())
    discounts_negative = bool(matrix[['PCT_10', 'PCT_20']].lt(0).all().all())
    if universal_free and discounts_negative:
        offer_text = ('The saved model predicts negative incremental NCM for PCT10 and PCT20 for every July customer; '
            'FREE_SHIP is positive and uniquely best for every customer. The algorithm is responding to those forecasts. '
            'This does not prove that real customers never benefit from discounts or that the offer-effect model is correct. '
            'The next substantive investigation is the offer-value model, treatment-response evidence, and accounting assumptions.')
    else:
        offer_text = ('The action-gain diagnostics below describe the current forecasts. They do not support a universal claim '
            'that both discounts are unprofitable and FREE_SHIP is best for everyone. Inspect the feasible allocation and individual comparisons together.')
    offer_table = ['| Offer | Customers with positive gain | Median predicted gain | Minimum | Maximum |',
                   '|---|---:|---:|---:|---:|']
    for r in distribution:
        if r['scope'] == 'all_customers' and r['action'] != 'NO_OFFER':
            offer_table.append(f"| {r['action']} | {r['positive']:,} | ${r['median']:.4f} | ${r['minimum']:.4f} | ${r['maximum']:.4f} |")
    report = f'''# Alternative allocation methods: November 19, 2026

This experiment compares CP-SAT and multi-start simulated annealing with the deployed greedy allocation and the saved previous MILP assignment. It uses the same {len(ids):,} customers, {len(hold)} reserved controls, original-dollar action predictions, $62,000 budget, at-most-60% offer cap ({cap:,} people), and at-most-15% PCT20 share among actual offer recipients.

No models were retrained and no controls, production allocator, or submission were changed. All {len(hashes)} protected files retain their input hashes. These are model projections, not observed July profit or a new historical policy comparison.

{chr(10).join(table)}

## What the methods actually test

- **CP-SAT:** builds one binary choice per customer/action, enforces all constraints jointly, and searches the combination space. Runs use two integer scales (one million and one hundred million). Costs are rounded upward and budget downward, so original-dollar feasibility is preserved. Solver optimality applies to the scaled problem; it is not silently treated as a certificate for the original costs. See the full solver statuses, bounds, and scaling metadata in `comparison.json`.
- **Simulated annealing:** uses feasible customer-action changes and paired changes, permits temporary worsening moves, and keeps the best feasible assignment. Runs use seeds {args.seeds}, {args.restarts} restarts per seed, and {args.iterations:,} proposal iterations per restart. Its retained candidate set includes greedy, so it cannot return worse than that incumbent. Acceptance/search statistics are saved; a matching incumbent by itself is not a proof of optimality.
- **Saved previous MILP:** reference assignment only; this experiment does not call a MILP solver.

## Independent bound on the original objective

For each noncontrol customer, take the largest of their four predicted incremental gains. Sum the largest {cap:,} positive values. This ignores budget and PCT20-share restrictions, so it is an upper bound on every feasible assignment under the original rules. The upper bound is **${upper_bound:,.10f}**.

{certificate_text}

## Offer-value diagnostics

{offer_text}

{chr(10).join(offer_table)}

## Reproduce

Install the normal project requirements, then the optional solver dependency:

```bash
python -m pip install -r requirements-optimizer-experiments.txt
python scripts/compare_allocation_methods.py
python -c "from ortools.sat.python import cp_model; import os; os.environ['PYTEST_DISABLE_PLUGIN_AUTOLOAD']='1'; import pytest; raise SystemExit(pytest.main(['-q','tests/test_allocation_cpsat.py','tests/test_allocation_annealing.py']))"
```

In this Windows/Anaconda environment, optional solver packages were installed only in the ignored `.validation-cache/allocator-deps` directory. The benchmark and optional-test command import CP-SAT before other native libraries to avoid DLL load-order conflicts. No global packages were changed. Run the normal project suite separately without the optional solver directory on `PYTHONPATH`; its optional CP-SAT tests then skip and are covered by the separate command above. The 20 CP-SAT tests include comparisons with exhaustive search and a case where conservative rounding changes the feasible set. The eight annealing tests include escaping a greedy budget trap and using a negative-gain support offer to enable a profitable PCT20 assignment.

Evidence: [comparison](../outputs/experiments/allocators/comparison.json), [method table](../outputs/experiments/allocators/method_comparison.csv), and [offer-gain diagnostics](../outputs/experiments/allocators/offer_gain_diagnostics.csv). CP-SAT integer-coefficient requirements and status meanings: [official OR-Tools documentation](https://developers.google.com/optimization/cp/cp_solver).
'''
    (ROOT / 'docs/ALLOCATION_METHOD_COMPARISON.md').write_text(report, encoding='utf-8')
    print('Protected files unchanged; saved experiment outputs and method-comparison document.', flush=True)


if __name__ == '__main__':
    main()
