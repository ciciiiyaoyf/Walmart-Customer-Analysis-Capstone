"""Compare original refits with the current predictive champion, separately from allocation."""
import io
import json
import subprocess
import sys
from pathlib import Path
import numpy as np
import pandas as pd

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from src.modeling import ACTIONS, feature_lists, fit_action_models, predict_actions, reg_metrics

REFERENCE="ea950d0e9011ff0e105f5d93758ab54944a7c75f"


def factual_predictions(predictions,expected):
    """Align by decision ID and reject mismatched test populations or outcomes."""
    if not predictions.assignment_id.is_unique or not expected.assignment_id.is_unique:
        raise ValueError('Each test assignment needs exactly one prediction')
    if set(predictions.assignment_id)!=set(expected.assignment_id):
        raise ValueError('Original and revised test assignment sets differ')
    aligned=predictions.set_index('assignment_id').loc[expected.assignment_id]
    if not np.array_equal(aligned.offer_type.to_numpy(),expected.offer_type.to_numpy()):
        raise ValueError('Original and revised observed actions differ')
    if not np.allclose(aligned.ncm_14d.to_numpy(),expected.ncm_14d.to_numpy(),rtol=1e-12,atol=1e-12):
        raise ValueError('Original and revised test targets differ')
    mu=aligned[[f'predicted_ncm_{a}' for a in ACTIONS]].to_numpy(float)
    if not np.isfinite(mu).all():raise ValueError('Revised predictions must be finite')
    return mu[np.arange(len(expected)),[ACTIONS.index(a) for a in expected.offer_type]]


def main():
    git=["git","-c",f"safe.directory={ROOT.as_posix()}","-C",str(ROOT)]
    original=pd.read_parquet(io.BytesIO(subprocess.check_output(git+["show",f"{REFERENCE}:data/processed/historical_feature_outcome.parquet"])))
    current=pd.read_parquet(ROOT/"data/processed/historical_feature_outcome.parquet")
    num,cat=feature_lists(); cols=num+cat
    before=original.sort_values("assignment_id").reset_index(drop=True)
    after=current.sort_values("assignment_id").reset_index(drop=True)
    if not before.assignment_id.equals(after.assignment_id):
        raise ValueError('Original and current historical assignment sets differ')
    differences=[]
    for c in cols:
        equal=(np.isclose(before[c].to_numpy(float),after[c].to_numpy(float),rtol=1e-12,atol=1e-12,equal_nan=True)
               if c in num else before[c].fillna("__missing__").eq(after[c].fillna("__missing__")).to_numpy())
        if not equal.all(): differences.append({"feature":c,"changed_rows":int((~equal).sum())})
    exp=original[original.assignment_source.eq("exploration")]
    train=exp[exp.campaign_id.lt("CP07")]
    test=exp[exp.campaign_id.eq("CP07")].sort_values("customer_id")
    models=fit_action_models(train,num,cat,{"n_estimators":100,"max_depth":1,"learning_rate":0.03})
    mu=predict_actions(models,test[cols])
    factual=mu[np.arange(len(test)),[ACTIONS.index(a) for a in test.offer_type]]
    rerun=reg_metrics(test.ncm_14d,factual)
    tables=ROOT/"outputs/tables"
    saved=pd.read_csv(ROOT/"outputs/reference/tlearner_test_metrics.csv").query("scope == 'pooled'").iloc[0]
    policy_selection=json.loads((ROOT/"outputs/models/policy_selection.json").read_text())
    selected_policy=policy_selection["selected_method"]
    profit_selection=json.loads((ROOT/"outputs/experiments/profit/selection.json").read_text())
    champion=profit_selection['selected_model']
    predictions=pd.read_csv(tables/'xgboost_test_predictions.csv')
    revised_factual=factual_predictions(predictions,test)
    revised=reg_metrics(test.ncm_14d,revised_factual)
    recorded=pd.read_csv(tables/'xgboost_model_metrics.csv').query("scope == 'pooled'").iloc[0]
    if not all(np.isclose(revised[k],recorded[k],rtol=1e-10,atol=1e-10) for k in ['rmse','mae','r2']):
        raise ValueError('Saved champion metrics disagree with saved predictions')
    rows=[{'version':name,'role':'profit_prediction','model':model,
           **{k:float(row[k]) for k in ['rmse','mae','r2']}} for name,model,row in
          [('original_saved_snapshot','ORIGINAL_TLEARNER',saved),
           ('original_refit_current_environment','ORIGINAL_TLEARNER',rerun),
           ('corrected_predictive_champion',champion,revised)]]
    # A random or fixed-action policy has no factual regression metric. Never
    # substitute the allocation model's accuracy for the predictive champion's.
    comparison=pd.read_csv(tables/'model_comparison_cp07.csv')
    allocation=comparison.query("model == @selected_policy and scope == 'pooled'")
    if selected_policy!=champion and len(allocation):
        row=allocation.iloc[0]
        rows.append({'version':'selected_allocation_model','role':'allocation_only','model':selected_policy,
                     **{k:float(row[k]) for k in ['rmse','mae','r2']}})
    results=pd.DataFrame(rows)
    results.to_csv(tables/"before_after_model_metrics.csv",index=False)
    improvement={k:float(rerun[k]-revised[k]) for k in ['rmse','mae']}
    improvement['r2_increase']=float(revised['r2']-rerun['r2'])
    check={"reference_commit":REFERENCE,"primary_ncm_unchanged":bool(np.allclose(before.ncm_14d,after.ncm_14d)),
           "feature_differences":differences,'predictive_champion':champion,
           'selected_allocation_policy':selected_policy,
           'allocation_policy_separate_from_predictive_champion':bool(selected_policy!=champion),
           'factual_predictions_changed_rows':int((~np.isclose(factual,revised_factual,rtol=1e-10,atol=1e-10)).sum()),
           'champion_improvement_over_same_runtime_refit':improvement,
           'champion_rmse_percent_reduction':float(100*improvement['rmse']/rerun['rmse']),
           'champion_mae_percent_reduction':float(100*improvement['mae']/rerun['mae']),
           "explanation":"Original and revised predictions use the same CP07 decisions and targets. Unchanged labels or features do not imply unchanged models. Compare the predictive champion with the original refit in this runtime; the allocation policy is selected and evaluated separately. Saved snapshot differences may reflect runtime/numerical differences."}
    (tables/"before_after_data_check.json").write_text(json.dumps(check,indent=2),encoding="utf-8")
    print(results.to_string(index=False))


if __name__ == "__main__":
    main()
