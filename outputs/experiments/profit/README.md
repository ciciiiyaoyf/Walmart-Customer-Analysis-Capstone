# Expected-profit model experiment

The selected model, `HURDLE_XGB_HALF`, improves factual net-contribution-margin prediction over the original four independent XGBoost regressors. The change is substantive: it shares information across randomized offer arms and separately models whether a customer buys and what their signed margin is if they buy.

## Design and selection

- Training uses only randomized exploration assignments and the existing 43 pre-decision features.
- Fourteen fixed candidates cover the original T-learner, pooled XGBoost, a shared model with action intercepts, and two-part models with several conditional-margin shrinkage levels.
- Rolling development fits earlier campaigns and evaluates CP04, CP05, and CP06. Selection minimizes pooled factual RMSE across those three folds, with MAE and model name as tie-breakers. Every candidate and every fold is retained in the CSV artifacts.
- `selection.json` was written before the selected candidate was evaluated on CP07. CP07 is a **retrospective test**, because the project had already inspected that campaign during earlier work. It is not a new untouched external test set.
- This experiment selects expected-outcome prediction, not a marketing policy. A lower factual RMSE alone does not establish better treatment-effect ranking or higher feasible policy value. Constrained policy evaluation is a separate analysis.

In CP01–06 exploration data, 87.85% of outcomes are zero because no purchase occurred. Fifteen purchaser outcomes are negative. The two-part identity is therefore useful, but taking a logarithm of the target or clipping negative profit would be incorrect. The selected estimator is:

`P(purchase | features, action) × [0.5 × E(NCM | purchase, features, action) + 0.5 × training-arm mean purchaser NCM]`.

Both component learners are pooled XGBoost models with action as an additional categorical predictor. The purchase learner has 150 trees, depth 2, learning rate 0.03, minimum child weight 20, and L2 regularization 20. The purchaser-margin learner has 100 trees, depth 2, learning rate 0.03, minimum child weight 10, and L2 regularization 30. Both use row and column subsampling of 0.8, seed 42, and histogram trees. Arm means and preprocessing are fitted on each training fold only. Margin predictions remain signed.

## Results

| Evaluation | Model | RMSE | MAE | R² |
|---|---|---:|---:|---:|
| CP04–06 rolling, 3,237 rows | Original T-learner | 62.9717 | 27.7017 | 0.0087 |
| CP04–06 rolling, 3,237 rows | Selected two-part XGBoost | 62.3674 | 27.1506 | 0.0276 |
| CP07 retrospective, 1,060 rows | Original T-learner | 63.3694 | 27.4111 | 0.0059 |
| CP07 retrospective, 1,060 rows | Selected two-part XGBoost | 62.7385 | 26.4806 | 0.0256 |

CP07 RMSE improves by 1.00%, and MAE by 3.39%. Three of the four action-specific RMSE values improve. NO_OFFER RMSE worsens slightly, from 40.9644 to 41.1716; improvements are not uniform across every subgroup or metric. R² remains low, so individual realized profit remains difficult to predict.

Five thousand paired customer-bootstrap draws give an old-minus-new CP07 mean-squared-error difference of 79.57 with 95% interval [17.45, 140.35], and mean-absolute-error difference of 0.931 with interval [0.452, 1.423]. Positive differences favor the new model. These intervals hold fitted models fixed and do not capture uncertainty from selecting candidates, retraining, or future campaign shifts. Development intervals are additionally optimistic because those folds selected the candidate.

RMSE is appropriate for a model whose target is expected dollar profit and whose training loss is squared error. MSE adds no independent ranking information because RMSE is its square root. MAE describes typical absolute dollar error; R² compares against a mean-prediction benchmark. None of these establishes causal uplift or policy profitability. Those require randomized logged-outcome evaluation of the actual feasible allocation.

## Reproduction and integration

1. `python scripts/experiment_profit.py` develops candidates and writes the selection.
2. `python scripts/experiment_profit.py --evaluate-test` evaluates only the original model and the frozen selected candidate on CP07.
3. `python -m pytest tests/test_profit_models.py -q` verifies exploration-only fitting, signed margins, valid hurdle factorization, all-arm support, serialization, and compatible output shape.

The current run used XGBoost 3.4.1 and scikit-learn 1.9.0. On this machine the project-local `.deps` directory must be on `PYTHONPATH`.

`fit_profit_models(data, num_features, categorical_features, 'HURDLE_XGB_HALF')` returns the existing action-to-predictor mapping, so `src.modeling.predict_actions` works without changing its API. `HURDLE_XGB_HALF_cp07_models.joblib` contains the CP01–06 fitted models used for the test. `development_predictions.csv` and `cp07_predictions.csv` contain every action's predicted NCM plus factual predictions, keyed by assignment ID, for independent policy checks.
