import joblib
import numpy as np
import pandas as pd
import pytest
from src.modeling import ACTIONS,predict_actions
from src.profit_models import HurdleActionModel,fit_profit_models,signed_log1p


class FixedProbability:
    def predict_proba(self,X):
        return np.tile([.75,.25],(len(X),1))


class FixedMargin:
    def predict(self,X):return np.full(len(X),-20.)


def rows():
    return pd.DataFrame({'x':np.tile(np.arange(12),4),
                         'offer_type':np.repeat(ACTIONS,12),
                         'assignment_source':'exploration',
                         'purchase_14d':np.tile([0,1],24),
                         'ncm_14d':np.tile([0.,-12.],24)})


def test_hurdle_preserves_negative_conditional_margin():
    model=HurdleActionModel(FixedProbability(),FixedMargin(),'NO_OFFER',-4.,.5)
    np.testing.assert_allclose(model.predict(pd.DataFrame({'x':[1,2]})),[-3.,-3.])


def test_hurdle_zero_weight_is_action_mean_and_does_not_change_input():
    x=pd.DataFrame({'x':[1,2]}); before=x.copy(deep=True)
    model=HurdleActionModel(FixedProbability(),None,'PCT_10',80.,0.)
    np.testing.assert_allclose(model.predict(x),[20.,20.])
    pd.testing.assert_frame_equal(x,before)


def test_signed_transform_does_not_discard_sign():
    np.testing.assert_allclose(signed_log1p([-9,0,9]),[-np.log(10),0,np.log(10)])


def test_rejects_nonrandomized_rows():
    data=rows();data.loc[0,'assignment_source']='business_rule'
    with pytest.raises(ValueError,match='randomized'):
        fit_profit_models(data,['x'],[],'POOLED_D2_100')


def test_rejects_missing_arm():
    with pytest.raises(ValueError,match='four actions'):
        fit_profit_models(rows().query("offer_type != 'PCT_20'"),['x'],[],'POOLED_D2_100')


def test_hurdle_rejects_nonzero_no_purchase_margin():
    data=rows();data.loc[0,'ncm_14d']=1.
    with pytest.raises(ValueError,match='zero NCM'):
        fit_profit_models(data,['x'],[],'HURDLE_XGB_MEAN')


def test_hurdle_fit_signed_target_and_serialization(tmp_path):
    data=rows()
    models=fit_profit_models(data,['x'],[],{'family':'hurdle','purchase':'xgb','purchase_params':{'n_estimators':5},'margin_weight':0.})
    predicted=predict_actions(models,data[['x']])
    assert predicted.shape==(48,4)
    assert np.isfinite(predicted).all() and (predicted<0).all()
    path=tmp_path/'models.joblib';joblib.dump(models,path)
    np.testing.assert_allclose(predict_actions(joblib.load(path),data[['x']]),predicted)


def test_pooled_supports_explicit_alternate_signed_target():
    data=rows();data['alternate']=data.ncm_14d*2
    models=fit_profit_models(data,['x'],[],{'family':'pooled','params':{'n_estimators':5}},target='alternate')
    result=predict_actions(models,data[['x']])
    assert result.shape==(48,4) and np.isfinite(result).all()


def test_invalid_margin_shrinkage_rejected():
    with pytest.raises(ValueError,match='shrinkage'):
        fit_profit_models(rows(),['x'],[],{'family':'hurdle','purchase_params':{'n_estimators':2},'margin_weight':1.1})


def test_reference_comparison_aligns_assignment_ids_before_scoring():
    from scripts.compare_reference import factual_predictions
    expected=pd.DataFrame({'assignment_id':['a','b'],'offer_type':['NO_OFFER','PCT_20'],'ncm_14d':[1.,2.]})
    predictions=expected.copy()
    for j,a in enumerate(ACTIONS):predictions[f'predicted_ncm_{a}']=[10.+j,20.+j]
    np.testing.assert_allclose(factual_predictions(predictions.iloc[::-1],expected),[10.,22.])


def test_reference_comparison_rejects_different_test_population():
    from scripts.compare_reference import factual_predictions
    expected=pd.DataFrame({'assignment_id':['a','b'],'offer_type':['NO_OFFER','PCT_20'],'ncm_14d':[1.,2.]})
    with pytest.raises(ValueError,match='assignment sets'):
        factual_predictions(expected.iloc[:1],expected)


def test_temporal_evaluation_rejects_unmatured_labels():
    from scripts.experiment_profit import require_mature_split
    train=pd.DataFrame({'send_ts':pd.to_datetime(['2025-01-01'])})
    overlap=pd.DataFrame({'send_ts':pd.to_datetime(['2025-01-14'])})
    with pytest.raises(ValueError,match='mature'):
        require_mature_split(train,overlap)
    require_mature_split(train,pd.DataFrame({'send_ts':pd.to_datetime(['2025-01-15'])}))


def test_allocation_rejects_misspelled_rule_instead_of_silently_optimizing():
    from src.policy import allocate_selected_policy
    with pytest.raises(ValueError,match='Unknown allocation'):
        allocate_selected_policy(pd.DataFrame(), 'RANDOM_FREE_SHIPPING')
