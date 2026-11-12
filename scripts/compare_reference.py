"""Compare original and revised fitting in one runtime, using the original Git snapshot."""
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


def main():
    git=["git","-c",f"safe.directory={ROOT.as_posix()}","-C",str(ROOT)]
    original=pd.read_parquet(io.BytesIO(subprocess.check_output(git+["show",f"{REFERENCE}:data/processed/historical_feature_outcome.parquet"])))
    current=pd.read_parquet(ROOT/"data/processed/historical_feature_outcome.parquet")
    num,cat=feature_lists(); cols=num+cat
    before=original.sort_values("assignment_id").reset_index(drop=True)
    after=current.sort_values("assignment_id").reset_index(drop=True)
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
    selected=json.loads((ROOT/"outputs/models/policy_selection.json").read_text())["selected_method"]
    revised=pd.read_csv(tables/"model_comparison_cp07.csv").query("model == @selected and scope == 'pooled'").iloc[0]
    results=pd.DataFrame([{ "version":name,**{k:float(row[k]) for k in ["rmse","mae","r2"]}} for name,row in
        [("original_saved_snapshot",saved),("original_refit_current_environment",rerun),("corrected_selected_model",revised)]])
    results.to_csv(tables/"before_after_model_metrics.csv",index=False)
    check={"reference_commit":REFERENCE,"primary_ncm_unchanged":bool(np.allclose(before.ncm_14d,after.ncm_14d)),
           "feature_differences":differences,"explanation":"Compare original and corrected fitting in the same runtime; saved snapshot differences may reflect runtime/numerical differences."}
    (tables/"before_after_data_check.json").write_text(json.dumps(check,indent=2),encoding="utf-8")
    print(results.to_string(index=False))


if __name__ == "__main__":
    main()
