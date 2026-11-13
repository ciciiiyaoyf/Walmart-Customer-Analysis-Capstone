"""Bounded purchase-model search on CP04-06; CP07 evaluation only after a saved selection lock."""
import argparse
import json
import sys
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, average_precision_score, log_loss, brier_score_loss

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from src.modeling import feature_lists, make_propensity_pipe
from src.propensity import fit_purchase_model, select_purchase_config, PURCHASE_SELECTION_RULE


def metrics(y,p):
    return {"roc_auc":roc_auc_score(y,p),"pr_auc":average_precision_score(y,p),
            "log_loss":log_loss(y,p),"brier":brier_score_loss(y,p),
            "observed_mean":float(np.mean(y)),"predicted_mean":float(np.mean(p))}


def evaluate(config, data, num, cat, campaigns=("CP04","CP05","CP06")):
    rows=[]; predictions=[]
    for campaign in campaigns:
        tr=data[data.campaign_id < campaign]
        va=data[data.campaign_id == campaign]
        model=fit_purchase_model(tr,num,cat,config)
        p=model.predict_proba(va[num+cat])[:,1]
        rows.append({"campaign":campaign,**config,**metrics(va.purchase_14d,p)})
        predictions.append(va[["assignment_id","customer_id","campaign_id","purchase_14d"]].assign(prediction=p))
    return pd.DataFrame(rows),pd.concat(predictions,ignore_index=True)


def paired_purchase_intervals(predictions, n_bootstrap=2000, seed=42):
    """Paired customer bootstrap; descriptive uncertainty, never model selection."""
    wide=predictions.pivot(index=['assignment_id','customer_id','purchase_14d'],columns='model',values='prediction').reset_index()
    y=wide.purchase_14d.to_numpy(); old=wide.original.to_numpy(); new=wide.selected.to_numpy()
    customers=wide.customer_id.to_numpy(); ids=np.unique(customers)
    grouped={customer:np.flatnonzero(customers==customer) for customer in ids}
    unique_rows=len(ids)==len(wide)
    rng=np.random.default_rng(seed)
    keys=['roc_auc','pr_auc','log_loss','brier']
    old_score=metrics(y,old);new_score=metrics(y,new)
    draws=[]
    for _ in range(n_bootstrap):
        index=(rng.integers(0,len(wide),len(wide)) if unique_rows else
               np.concatenate([grouped[c] for c in rng.choice(ids,len(ids),replace=True)]))
        if len(np.unique(y[index]))<2:continue
        before=metrics(y[index],old[index]);after=metrics(y[index],new[index])
        draws.append([after[k]-before[k] for k in keys])
    bounds=np.quantile(draws,[.025,.975],axis=0)
    return pd.DataFrame([{'metric':k,'original':old_score[k],'selected':new_score[k],
                          'difference_selected_minus_original':new_score[k]-old_score[k],
                          'ci_low':bounds[0,j],'ci_high':bounds[1,j],
                          'better_direction':'higher' if k in ['roc_auc','pr_auc'] else 'lower',
                          'bootstrap_draws':len(draws),'bootstrap_unit':'customer'} for j,k in enumerate(keys)])


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--test",action="store_true",help="Evaluate locked choice on CP07, after development run")
    parser.add_argument("--bootstrap-only",action="store_true",help="Compute intervals from saved CP07 predictions without fitting")
    args=parser.parse_args()
    data=pd.read_parquet(ROOT/"data/processed/historical_feature_outcome.parquet")
    num,cat=feature_lists()
    out=ROOT/"outputs/experiments/propensity";out.mkdir(parents=True,exist_ok=True)
    if args.bootstrap_only:
        intervals=paired_purchase_intervals(pd.read_csv(out/'cp07_predictions.csv'))
        intervals.to_csv(out/'cp07_paired_bootstrap.csv',index=False)
        print(intervals.to_string(index=False),flush=True)
        return
    if not args.test:
        config,folds,summary=select_purchase_config(data,num,cat,verbose=True)
        folds.to_csv(out/'development_folds.csv',index=False)
        summary.to_csv(out/'development_summary.csv',index=False)
        selection={'config':config,'development_campaigns':['CP04','CP05','CP06'],
            'rule':PURCHASE_SELECTION_RULE,
            'test_status':'CP07 not used in this search; already inspected in previous audit so retrospective confirmation only'}
        (out/'selection.json').write_text(json.dumps(selection,indent=2),encoding='utf-8')
        print('LOCKED:',selection,flush=True)
        return
    selection=json.loads((out/'selection.json').read_text())
    rows=[]; predictions=[]
    for name,cfg in [('original',{'mode':'raw','C':.01,'calibration':'none'}),('selected',selection['config'])]:
        result,pred=evaluate(cfg,data,num,cat,campaigns=['CP07'])
        result['model']=name;pred['model']=name
        rows.append(result);predictions.append(pred)
    comparison=pd.concat(rows,ignore_index=True)
    comparison.to_csv(out/'cp07_comparison.csv',index=False)
    predictions=pd.concat(predictions,ignore_index=True)
    predictions.to_csv(out/'cp07_predictions.csv',index=False)
    paired_purchase_intervals(predictions).to_csv(out/'cp07_paired_bootstrap.csv',index=False)
    print(comparison.to_string(index=False),flush=True)


if __name__=='__main__':main()
