# Alternative allocation methods: November 19, 2026

This experiment compares CP-SAT and multi-start simulated annealing with the deployed greedy allocation and the saved previous MILP assignment. It uses the same 2,776 customers, 139 reserved controls, original-dollar action predictions, $62,000 budget, at-most-60% offer cap (1,665 people), and at-most-15% PCT20 share among actual offer recipients.

No models were retrained and no controls, production allocator, or submission were changed. All 26 protected files retain their input hashes. These are model projections, not observed July profit or a new historical policy comparison.

| Method | Projected incremental NCM | PCT10 | PCT20 | FREE_SHIP | NO_OFFER | Changes vs deployed |
|---|---:|---:|---:|---:|---:|---:|
| DEPLOYED_GREEDY | $2,501.78337601 | 0 | 0 | 1665 | 1111 | 0 |
| PREVIOUS_MILP_REFERENCE | $2,501.78337601 | 0 | 0 | 1665 | 1111 | 0 |
| GREEDY_RERUN | $2,501.78337601 | 0 | 0 | 1665 | 1111 | 0 |
| CP_SAT_SCALE_1000000 | $2,501.78337601 | 0 | 0 | 1665 | 1111 | 0 |
| CP_SAT_SCALE_100000000 | $2,501.78337601 | 0 | 0 | 1665 | 1111 | 0 |
| SIMULATED_ANNEALING_SEED_7 | $2,501.78337601 | 0 | 0 | 1665 | 1111 | 0 |
| SIMULATED_ANNEALING_SEED_42 | $2,501.78337601 | 0 | 0 | 1665 | 1111 | 0 |
| SIMULATED_ANNEALING_SEED_2026 | $2,501.78337601 | 0 | 0 | 1665 | 1111 | 0 |

## What the methods actually test

- **CP-SAT:** builds one binary choice per customer/action, enforces all constraints jointly, and searches the combination space. Runs use two integer scales (one million and one hundred million). Costs are rounded upward and budget downward, so original-dollar feasibility is preserved. Solver optimality applies to the scaled problem; it is not silently treated as a certificate for the original costs. See the full solver statuses, bounds, and scaling metadata in `comparison.json`.
- **Simulated annealing:** uses feasible customer-action changes and paired changes, permits temporary worsening moves, and keeps the best feasible assignment. Runs use seeds [7, 42, 2026], 4 restarts per seed, and 30,000 proposal iterations per restart. Its retained candidate set includes greedy, so it cannot return worse than that incumbent. Acceptance/search statistics are saved; a matching incumbent by itself is not a proof of optimality.
- **Saved previous MILP:** reference assignment only; this experiment does not call a MILP solver.

## Independent bound on the original objective

For each noncontrol customer, take the largest of their four predicted incremental gains. Sum the largest 1,665 positive values. This ignores budget and PCT20-share restrictions, so it is an upper bound on every feasible assignment under the original rules. The upper bound is **$2,501.7833760103**.

The deployed policy attains this bound within $0.000000001 while satisfying all original constraints. Therefore, for these exact forecasts and fixed controls, the existing allocation is globally optimal to numerical precision. This instance-specific proof does not make greedy a generally optimal algorithm. Changing the algorithm cannot produce a larger objective under the unchanged inputs.

## Offer-value diagnostics

The saved model predicts negative incremental NCM for PCT10 and PCT20 for every July customer; FREE_SHIP is positive and uniquely best for every customer. The algorithm is responding to those forecasts. This does not prove that real customers never benefit from discounts or that the offer-effect model is correct. The next substantive investigation is the offer-value model, treatment-response evidence, and accounting assumptions.

| Offer | Customers with positive gain | Median predicted gain | Minimum | Maximum |
|---|---:|---:|---:|---:|
| FREE_SHIP | 2,776 | $0.9406 | $0.5026 | $2.9152 |
| PCT_10 | 0 | $-0.8083 | $-3.0839 | $-0.4257 |
| PCT_20 | 0 | $-4.6331 | $-17.4247 | $-2.4845 |

## Reproduce

Install the normal project requirements, then the optional solver dependency:

```bash
python -m pip install -r requirements-optimizer-experiments.txt
python scripts/compare_allocation_methods.py
python -c "from ortools.sat.python import cp_model; import os; os.environ['PYTEST_DISABLE_PLUGIN_AUTOLOAD']='1'; import pytest; raise SystemExit(pytest.main(['-q','tests/test_allocation_cpsat.py','tests/test_allocation_annealing.py']))"
```

In this Windows/Anaconda environment, optional solver packages were installed only in the ignored `.validation-cache/allocator-deps` directory. The benchmark and optional-test command import CP-SAT before other native libraries to avoid DLL load-order conflicts. No global packages were changed. Run the normal project suite separately without the optional solver directory on `PYTHONPATH`; its optional CP-SAT tests then skip and are covered by the separate command above. The 20 CP-SAT tests include comparisons with exhaustive search and a case where conservative rounding changes the feasible set. The eight annealing tests include escaping a greedy budget trap and using a negative-gain support offer to enable a profitable PCT20 assignment.

Evidence: [comparison](../outputs/experiments/allocators/comparison.json), [method table](../outputs/experiments/allocators/method_comparison.csv), and [offer-gain diagnostics](../outputs/experiments/allocators/offer_gain_diagnostics.csv). CP-SAT integer-coefficient requirements and status meanings: [official OR-Tools documentation](https://developers.google.com/optimization/cp/cp_solver).
