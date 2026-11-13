import joblib
import numpy as np
import pandas as pd
import pytest

import src.propensity as purchase
from src.features import OBSERVED_FEATURE_RENAMES


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


@pytest.mark.parametrize('mode', purchase.MODES)
def test_observed_schema_rename_preserves_numeric_design_and_predictions(mode):
    legacy = example()
    legacy['historical_return_rate'] = np.tile([0., .5, np.nan], 40)
    legacy['avg_order_value_90d'] = legacy.gross_revenue_90d / 3
    legacy['customer_tenure_days'] = np.arange(len(legacy)) + 10.
    legacy['prior_emails_delivered'] = np.arange(len(legacy)) % 5
    legacy['historical_open_rate'] = np.tile([0., .25, .5], 40)
    num = NUM + ['historical_return_rate', 'avg_order_value_90d', 'customer_tenure_days',
                 'prior_emails_delivered', 'historical_open_rate']
    current = legacy.rename(columns=OBSERVED_FEATURE_RENAMES)
    current_num = [OBSERVED_FEATURE_RENAMES.get(c, c) for c in num]
    current_cat = [OBSERVED_FEATURE_RENAMES.get(c, c) for c in CAT]
    config = {'mode': mode, 'C': .01}
    old_model = purchase.fit_purchase_model(legacy, num, CAT, config)
    new_model = purchase.fit_purchase_model(current, current_num, current_cat, config)
    old_features = old_model.named_steps['features'].transform(legacy)
    new_features = new_model.named_steps['features'].transform(current)
    pd.testing.assert_frame_equal(old_features.rename(columns=OBSERVED_FEATURE_RENAMES), new_features)
    old_design = old_model.named_steps['pre'].transform(old_features)
    new_design = new_model.named_steps['pre'].transform(new_features)
    np.testing.assert_array_equal(old_design, new_design)
    np.testing.assert_allclose(old_model.predict_proba(legacy), new_model.predict_proba(current), rtol=0., atol=0.)


def test_interactions_prefer_observed_columns_and_preserve_unknowns():
    data = example().rename(columns=OBSERVED_FEATURE_RENAMES)
    data['days_since_last_order'] = 0.
    data['top_purchase_category_90d'] = data.top_category_viewed_30d
    data['gross_revenue_90d'] = 1.e9
    data.loc[0, 'observed_days_since_last_order'] = 60.
    data.loc[0, 'observed_top_purchase_category_90d'] = 'Other'
    data.loc[0, 'observed_gross_revenue_90d'] = np.nan
    data.loc[0, 'is_member'] = 1
    num = [OBSERVED_FEATURE_RENAMES.get(c, c) for c in NUM]
    cat = [OBSERVED_FEATURE_RENAMES.get(c, c) for c in CAT]
    out = purchase.PurchaseFeatures(num, cat, 'log_interactions').fit_transform(data)
    assert out.loc[0, 'recent_purchase'] == 0.
    assert out.loc[0, 'purchase_matches_browse'] == 0.
    assert pd.isna(out.loc[0, 'member_spend'])


def test_model_schema_excludes_numeric_coverage_context(monkeypatch, tmp_path):
    import src.modeling as modeling
    from src.features import FAMILIES, FEATURES, CATEGORICAL
    schema = pd.DataFrame([
        {'feature': feature, 'family': family,
         'type': 'categorical' if feature in CATEGORICAL else 'numeric'}
        for family, features in FAMILIES.items() for feature in features])
    context = pd.DataFrame([
        {'feature': 'complete_orders_90d', 'family': 'context', 'type': 'numeric'},
        {'feature': 'orders_90d_coverage_status', 'family': 'context', 'type': 'categorical'},
        {'feature': 'observed_organic_reference_orders', 'family': 'context', 'type': 'numeric'}])
    pd.concat([context, schema], ignore_index=True).to_csv(tmp_path/'feature_schema.csv', index=False)
    monkeypatch.setattr(modeling, 'TABLES', tmp_path)
    num, cat = modeling.feature_lists()
    assert num == [feature for feature in FEATURES if feature not in CATEGORICAL]
    assert cat == [feature for feature in FEATURES if feature in CATEGORICAL]
    assert len(num + cat) == 43
    assert not set(context.feature) & set(num + cat)
