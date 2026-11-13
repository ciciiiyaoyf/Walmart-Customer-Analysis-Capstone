"""Bounded purchase-model search on CP04-06; CP07 evaluation only after a saved selection lock."""
import argparse
import json
import sys
from pathlib import Path
import numpy as np
import pandas as pd

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from src.modeling import feature_lists, make_propensity_pipe
from src.propensity import (fit_purchase_model, select_purchase_config, PURCHASE_SELECTION_RULE,
                            purchase_metrics, choose_purchase_threshold)


def metrics(y, p, threshold=None):
    return purchase_metrics(y, p, threshold=threshold)


def development_purchase_thresholds(data, num, cat, selected_config):
    """Generate prior-only CP04-06 predictions and lock a threshold per model."""
    development = data[data.campaign_id.lt("CP07")]
    predictions = []
    for campaign in ["CP04", "CP05", "CP06"]:
        train = development[development.campaign_id.lt(campaign)]
        held = development[development.campaign_id.eq(campaign)]
        if train.empty or held.empty:
            raise ValueError("Threshold selection requires CP04-CP06 and earlier fitting campaigns")
        if 'send_ts' in development and train.send_ts.max()+pd.Timedelta(days=14) > held.send_ts.min():
            raise ValueError("Threshold training labels must mature before validation decisions")
        models = {
            "original": make_propensity_pipe(num, cat, C=.01).fit(train[num+cat], train.purchase_14d),
            "selected": fit_purchase_model(train, num, cat, selected_config),
        }
        for name, model in models.items():
            predictions.append(held[["assignment_id", "customer_id", "campaign_id", "purchase_14d"]].assign(
                model=name, prediction=model.predict_proba(held[num+cat])[:, 1]))
    predictions = pd.concat(predictions, ignore_index=True)
    thresholds = {name: choose_purchase_threshold(group.purchase_14d, group.prediction)
                  for name, group in predictions.groupby("model")}
    metadata = {**thresholds, "campaigns": ["CP04", "CP05", "CP06"],
                "rule": "maximum F1 on pooled out-of-time development predictions; lowest threshold wins ties",
                "selected_config": selected_config, "frozen_before_cp07": True,
                "interpretation": "Per-model illustrative classification thresholds; not promotion allocation rules"}
    return metadata, predictions


def evaluate(config, data, num, cat, campaigns=("CP04","CP05","CP06"), threshold=None):
    rows=[]; predictions=[]
    for campaign in campaigns:
        tr=data[data.campaign_id < campaign]
        va=data[data.campaign_id == campaign]
        model=fit_purchase_model(tr,num,cat,config)
        p=model.predict_proba(va[num+cat])[:,1]
        rows.append({"campaign":campaign,**config,**metrics(va.purchase_14d,p,threshold=threshold)})
        predictions.append(va[["assignment_id","customer_id","campaign_id","purchase_14d"]].assign(prediction=p))
    return pd.DataFrame(rows),pd.concat(predictions,ignore_index=True)


def paired_purchase_intervals(predictions, n_bootstrap=2000, seed=42, thresholds=None):
    """Paired bootstrap at frozen development thresholds; never optimize a draw.

    With no thresholds, recall is omitted. Probability scores remain available.
    Both models must use exactly the same evaluation decisions and outcomes.
    """
    if n_bootstrap < 2 or set(predictions.model) != {"original", "selected"}:
        raise ValueError("At least two draws and original/selected predictions are required")
    if predictions.duplicated(["assignment_id", "model"]).any():
        raise ValueError("Each assignment/model requires exactly one prediction")
    if thresholds is not None and not {"original", "selected"}.issubset(thresholds):
        raise ValueError("Supply frozen thresholds for both original and selected models")
    old_threshold = None if thresholds is None else thresholds['original']
    new_threshold = None if thresholds is None else thresholds['selected']
    wide=predictions.pivot(index=['assignment_id','customer_id','purchase_14d'],columns='model',values='prediction').reset_index()
    y=wide.purchase_14d.to_numpy(); old=wide.original.to_numpy(); new=wide.selected.to_numpy()
    customers=wide.customer_id.to_numpy(); ids=np.unique(customers)
    grouped={customer:np.flatnonzero(customers==customer) for customer in ids}
    unique_rows=len(ids)==len(wide)
    rng=np.random.default_rng(seed)
    keys=['average_precision','roc_auc','pr_auc','log_loss','brier']
    if thresholds is not None: keys.insert(1,'recall')
    old_score=metrics(y,old,old_threshold);new_score=metrics(y,new,new_threshold)
    draws=[]
    for _ in range(n_bootstrap):
        index=(rng.integers(0,len(wide),len(wide)) if unique_rows else
               np.concatenate([grouped[c] for c in rng.choice(ids,len(ids),replace=True)]))
        if len(np.unique(y[index]))<2:continue
        before=metrics(y[index],old[index],old_threshold);after=metrics(y[index],new[index],new_threshold)
        draws.append([after[k]-before[k] for k in keys])
    if not draws: raise ValueError('No bootstrap samples contain both outcome classes')
    bounds=np.quantile(draws,[.025,.975],axis=0)
    return pd.DataFrame([{'metric':k,'original':old_score[k],'selected':new_score[k],
                          'difference_selected_minus_original':new_score[k]-old_score[k],
                          'ci_low':bounds[0,j],'ci_high':bounds[1,j],
                          'better_direction':'higher' if k in ['average_precision','recall','roc_auc','pr_auc'] else 'lower',
                          'bootstrap_draws':len(draws),'bootstrap_unit':'customer',
                          'original_threshold':old_threshold if k=='recall' else np.nan,
                          'selected_threshold':new_threshold if k=='recall' else np.nan,
                          'thresholds_fixed_in_bootstrap':thresholds is not None} for j,k in enumerate(keys)])


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--test",action="store_true",help="Evaluate locked choice on CP07, after development run")
    parser.add_argument("--bootstrap-only",action="store_true",help="Compute intervals from saved CP07 predictions without fitting")
    args=parser.parse_args()
    data=pd.read_parquet(ROOT/"data/processed/historical_feature_outcome.parquet")
    num,cat=feature_lists()
    out=ROOT/"outputs/experiments/propensity";out.mkdir(parents=True,exist_ok=True)
    if args.bootstrap_only:
        thresholds=json.loads((out/'thresholds.json').read_text())
        intervals=paired_purchase_intervals(pd.read_csv(out/'cp07_predictions.csv'),thresholds=thresholds)
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
        thresholds, threshold_predictions = development_purchase_thresholds(data,num,cat,config)
        (out/'thresholds.json').write_text(json.dumps(thresholds,indent=2)+'\n',encoding='utf-8')
        threshold_predictions.to_csv(out/'threshold_development_predictions.csv',index=False)
        print('LOCKED:',selection,flush=True)
        return
    selection=json.loads((out/'selection.json').read_text())
    thresholds=json.loads((out/'thresholds.json').read_text())
    if thresholds['selected_config'] != selection['config']:
        raise ValueError('Threshold configuration differs from the locked model choice; rerun development')
    rows=[]; predictions=[]
    for name,cfg in [('original',{'mode':'raw','C':.01,'calibration':'none'}),('selected',selection['config'])]:
        result,pred=evaluate(cfg,data,num,cat,campaigns=['CP07'],threshold=thresholds[name])
        result['model']=name;pred['model']=name
        rows.append(result);predictions.append(pred)
    comparison=pd.concat(rows,ignore_index=True)
    comparison.to_csv(out/'cp07_comparison.csv',index=False)
    predictions=pd.concat(predictions,ignore_index=True)
    predictions.to_csv(out/'cp07_predictions.csv',index=False)
    paired_purchase_intervals(predictions,thresholds=thresholds).to_csv(out/'cp07_paired_bootstrap.csv',index=False)
    print(comparison.to_string(index=False),flush=True)


if __name__=='__main__':main()
