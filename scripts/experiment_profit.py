"""Develop profit models on CP04-06, lock choice, optionally evaluate CP07 once."""
import argparse
import json
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import joblib

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from src.modeling import ACTIONS,feature_lists,predict_actions,reg_metrics
from src.profit_models import PROFIT_CANDIDATES,fit_profit_models
from src.evaluation import cluster_mean_interval

OUT=ROOT/'outputs/experiments/profit'


def require_mature_split(train, test):
    """Observed training labels must finish before the validation decisions."""
    if train.empty or test.empty or train.send_ts.max()+pd.Timedelta(days=14) > test.send_ts.min():
        raise ValueError('Training outcomes must mature before evaluation decisions')


def paired_loss_comparison(predictions,selected):
    """Conditional customer-bootstrap uncertainty; positive change is better."""
    columns=['assignment_id','customer_id','ncm_14d','factual_prediction']
    old=predictions.loc[predictions.model.eq('ORIGINAL_TLEARNER'),columns]
    new=predictions.loc[predictions.model.eq(selected),['assignment_id','factual_prediction']]
    paired=old.merge(new,on='assignment_id',validate='one_to_one',suffixes=('_old','_new'))
    if len(paired)!=len(old):raise ValueError('Paired evaluation rows differ')
    old_error=paired.ncm_14d-paired.factual_prediction_old
    new_error=paired.ncm_14d-paired.factual_prediction_new
    records=[]
    for metric,change in [('mse',old_error**2-new_error**2),('mae',abs(old_error)-abs(new_error))]:
        point,lower,upper=cluster_mean_interval(change,paired.customer_id,draws=5000)
        records.append({'comparison':f'{selected} versus ORIGINAL_TLEARNER','loss':metric,
                        'old_minus_new':point,'ci95_lower':lower,'ci95_upper':upper,
                        'n_rows':len(paired),'n_customers':paired.customer_id.nunique(),
                        'interpretation':'positive favors selected; fixed fitted models, not model-selection uncertainty'})
    return pd.DataFrame(records)


def run_profit_search(data,num,cat,out=OUT):
    """Fit rolling development candidates and serialize the frozen winner."""
    out=Path(out);out.mkdir(parents=True,exist_ok=True)
    data=data[data.assignment_source.eq('exploration')]
    rows=[]; predictions=[]
    for campaign in ['CP04','CP05','CP06']:
        train=data[data.campaign_id.lt(campaign)]
        test=data[data.campaign_id.eq(campaign)].sort_values('customer_id')
        require_mature_split(train,test)
        for name,config in PROFIT_CANDIDATES.items():
            models=fit_profit_models(train,num,cat,config)
            mu=predict_actions(models,test[num+cat])
            factual=mu[np.arange(len(test)),[ACTIONS.index(a) for a in test.offer_type]]
            result={'campaign':campaign,'model':name,**reg_metrics(test.ncm_14d,factual)}
            rows.append(result)
            pred=test[['assignment_id','customer_id','campaign_id','offer_type','ncm_14d','purchase_14d']].copy()
            pred['model']=name
            for j,a in enumerate(ACTIONS):pred[f'predicted_ncm_{a}']=mu[:,j]
            pred['factual_prediction']=factual
            predictions.append(pred)
            print(campaign,name,round(result['rmse'],4),flush=True)
    scores=pd.DataFrame(rows); allpred=pd.concat(predictions,ignore_index=True)
    scores.to_csv(out/'development_fold_metrics.csv',index=False)
    allpred.to_csv(out/'development_predictions.csv',index=False)
    pooled=[]
    for name,group in allpred.groupby('model'):
        pooled.append({'model':name,**reg_metrics(group.ncm_14d,group.factual_prediction)})
    pooled=pd.DataFrame(pooled).sort_values(['rmse','mae','model'])
    pooled.to_csv(out/'development_model_comparison.csv',index=False)
    selected=pooled.iloc[0].model
    selection={'selected_model':selected,'config':PROFIT_CANDIDATES[selected],
               'development_campaigns':['CP04','CP05','CP06'],
               'criterion':'minimum pooled out-of-time factual RMSE; MAE and stable name tie-break',
               'candidate_registry':PROFIT_CANDIDATES,
               'test_status':'CP07 retrospective; never used by this model-selection script'}
    (out/'selection.json').write_text(json.dumps(selection,indent=2),encoding='utf-8')
    paired_loss_comparison(allpred,selected).to_csv(out/'development_paired_loss.csv',index=False)
    print(pooled.to_string(index=False),flush=True)
    return selection


def evaluate_profit_selection(data,num,cat,selection,out=OUT):
    """Evaluate original and frozen winner on CP07; return metrics, predictions."""
    out=Path(out);out.mkdir(parents=True,exist_ok=True)
    data=data[data.assignment_source.eq('exploration')]
    names=['ORIGINAL_TLEARNER',selection['selected_model']]
    train=data[data.campaign_id.lt('CP07')];test=data[data.campaign_id.eq('CP07')].sort_values('customer_id')
    require_mature_split(train,test)
    rows=[];predictions=[]
    for name in dict.fromkeys(names):
        config=selection['config'] if name==selection['selected_model'] else PROFIT_CANDIDATES[name]
        models=fit_profit_models(train,num,cat,config)
        mu=predict_actions(models,test[num+cat])
        factual=mu[np.arange(len(test)),[ACTIONS.index(a) for a in test.offer_type]]
        rows.append({'model':name,'scope':'pooled',**reg_metrics(test.ncm_14d,factual)})
        for a in ACTIONS:
            keep=test.offer_type.eq(a)
            rows.append({'model':name,'scope':a,**reg_metrics(test.loc[keep,'ncm_14d'],factual[keep])})
        pred=test[['assignment_id','customer_id','campaign_id','offer_type','ncm_14d','purchase_14d']].copy()
        pred['model']=name
        for j,a in enumerate(ACTIONS):pred[f'predicted_ncm_{a}']=mu[:,j]
        pred['factual_prediction']=factual
        predictions.append(pred)
        joblib.dump(models,out/f'{name}_cp07_models.joblib')
    metrics=pd.DataFrame(rows)
    metrics.to_csv(out/'cp07_model_metrics.csv',index=False)
    predictions=pd.concat(predictions,ignore_index=True)
    predictions.to_csv(out/'cp07_predictions.csv',index=False)
    paired_loss_comparison(predictions,selection['selected_model']).to_csv(out/'cp07_paired_loss.csv',index=False)
    print(metrics.to_string(index=False))
    return metrics,predictions


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--evaluate-test',action='store_true')
    args=parser.parse_args()
    data=pd.read_parquet(ROOT/'data/processed/historical_feature_outcome.parquet')
    num,cat=feature_lists()
    if args.evaluate_test:
        selection=json.loads((OUT/'selection.json').read_text())
        evaluate_profit_selection(data,num,cat,selection)
    else:
        run_profit_search(data,num,cat)


if __name__=='__main__':main()
