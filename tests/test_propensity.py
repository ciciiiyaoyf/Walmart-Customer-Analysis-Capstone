import joblib
import numpy as np
import pandas as pd
import pytest

import src.propensity as purchase


NUM=['is_member','days_since_last_order','days_since_last_session','sessions_30d','gross_revenue_90d']
CAT=['customer_tier','preferred_category','top_category_viewed_30d','top_purchase_category_90d']


def example():
    rng=np.random.default_rng(12)
    data=pd.DataFrame({'campaign_id':np.repeat(['CP01','CP02','CP03','CP04'],30),
                       'purchase_14d':np.tile([0,0,1],40),'is_member':rng.integers(0,2,120),
                       'days_since_last_order':rng.uniform(0,60,120),
                       'days_since_last_session':rng.uniform(0,20,120),
                       'sessions_30d':rng.integers(0,20,120),'gross_revenue_90d':rng.uniform(-30,500,120)})
    data['customer_tier']=np.tile(['Gold','Standard','Silver'],40)
    for c in CAT[1:]:data[c]=np.tile(['Food','Clothing','Electronics'],40)
    return data


def test_signed_transform_preserves_negative_values_and_drops_unregistered_labels():
    data=example();data.loc[0,'gross_revenue_90d']=-9.;data.loc[1,'sessions_30d']=np.nan
    data['offer_type']='PCT_20';data['ncm_14d']=1.e9
    before=data.copy(deep=True)
    transformed=purchase.PurchaseFeatures(NUM,CAT,'log_interactions').fit_transform(data)
    assert transformed.loc[0,'gross_revenue_90d']==-np.log(10)
    assert np.isnan(transformed.loc[1,'sessions_30d'])
    assert not {'purchase_14d','offer_type','ncm_14d','campaign_id'} & set(transformed)
    pd.testing.assert_frame_equal(data,before)


def test_unseen_categories_and_missing_values_are_scoreable_and_serializable(tmp_path):
    data=example();model=purchase.fit_purchase_model(data,NUM,CAT,{'mode':'log_interactions','C':.01})
    future=data.iloc[:4].copy();future['customer_tier']='Unknown';future.loc[0,NUM[1:]]=np.nan
    p=model.predict_proba(future)
    assert np.isfinite(p).all() and (p>0).all() and (p<1).all()
    np.testing.assert_allclose(p.sum(axis=1),1.)
    path=tmp_path/'purchase.joblib';joblib.dump(model,path)
    np.testing.assert_allclose(joblib.load(path).predict_proba(future),p)


def test_temporal_calibration_fits_each_prediction_block_using_only_earlier_labels(monkeypatch):
    data=example();fitted=[];scored=[]
    original=purchase.make_purchase_pipeline
    def tracking(num,cat,config):
        pipe=original(num,cat,config);fit=pipe.fit;predict=pipe.predict_proba
        def wrapped_fit(X,y):
            fitted.append(set(data.loc[X.index,'campaign_id']))
            return fit(X,y)
        def wrapped_predict(X):
            scored.append(set(data.loc[X.index,'campaign_id']))
            assert max(fitted[-1]) < min(scored[-1])
            return predict(X)
        pipe.fit=wrapped_fit;pipe.predict_proba=wrapped_predict
        return pipe
    monkeypatch.setattr(purchase,'make_purchase_pipeline',tracking)
    model=purchase.fit_purchase_model(data,NUM,CAT,{'mode':'raw','C':.01,'calibration':'sigmoid'})
    assert model.calibration_campaigns==['CP02','CP03','CP04']
    assert scored==[{'CP02'},{'CP03'},{'CP04'}]
    assert fitted[-1]=={'CP01','CP02','CP03','CP04'}


def test_selection_does_not_consume_cp07_or_current_validation_labels(monkeypatch):
    data=example();cp07=data.iloc[:30].assign(campaign_id='CP07',purchase_14d=999)
    data=pd.concat([data,cp07],ignore_index=True);seen=[]
    original=purchase.fit_purchase_model
    def tracking(frame,num,cat,config):
        assert frame.campaign_id.max()<'CP07'
        seen.append(frame.campaign_id.max())
        return original(frame,num,cat,config)
    monkeypatch.setattr(purchase,'fit_purchase_model',tracking)
    monkeypatch.setattr(purchase,'purchase_candidates',lambda:[{'mode':'raw','C':.01,'calibration':'none'}])
    config,folds,_=purchase.select_purchase_config(data,NUM,CAT)
    assert config['mode']=='raw'
    assert set(folds.campaign)=={'CP03','CP04'}
    assert seen==['CP02','CP03']
    with pytest.raises(ValueError,match='before CP07'):
        purchase.select_purchase_config(data,NUM,CAT,['CP07'])


def test_too_little_history_is_explicit():
    data=example().query("campaign_id == 'CP01'")
    config,folds,summary=purchase.select_purchase_config(data,NUM,CAT)
    assert config=={'mode':'raw','C':.01,'calibration':'none'} and folds.empty and summary.empty
    with pytest.raises(ValueError,match='at least three'):
        purchase.fit_purchase_model(data,NUM,CAT,dict(config,calibration='sigmoid'))
