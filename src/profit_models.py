"""Partially pooled XGBoost and two-part models for expected signed NCM.

Every learner is fitted on randomized exploration rows. Hurdle models estimate
P(purchase | X, action) * E(NCM | purchase, X, action); conditional margins may
be negative and are never log-transformed or clipped to zero.
"""
from dataclasses import dataclass
import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler, FunctionTransformer
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier, XGBRegressor
from src.modeling import ACTIONS, fit_action_models


def signed_log1p(values):
    values=np.asarray(values,float)
    return np.sign(values)*np.log1p(np.abs(values))


def _preprocess(num,cat,logistic=False):
    numeric=[('impute',SimpleImputer(strategy='median',add_indicator=True,keep_empty_features=True))]
    if logistic:
        numeric += [('log',FunctionTransformer(signed_log1p)),('scale',StandardScaler())]
    return ColumnTransformer([
        ('num',Pipeline(numeric),num),
        ('cat',Pipeline([('impute',SimpleImputer(strategy='constant',fill_value='Missing',keep_empty_features=True)),
                         ('encode',OneHotEncoder(handle_unknown='ignore'))]),cat)])


def _xgb(num,cat,params,classifier=False):
    cfg=dict(n_estimators=150,max_depth=2,learning_rate=.03,min_child_weight=20,
             reg_lambda=20.,subsample=.8,colsample_bytree=.8,tree_method='hist',random_state=42,n_jobs=4)
    cfg.update(params)
    learner=(XGBClassifier(objective='binary:logistic',eval_metric='logloss',**cfg) if classifier else
             XGBRegressor(objective='reg:squarederror',**cfg))
    return Pipeline([('pre',_preprocess(num,cat)),('model',learner)])


@dataclass
class PooledActionModel:
    model: object
    action: str
    def predict(self,X):
        return np.asarray(self.model.predict(X.assign(_action=self.action)),float)


@dataclass
class HurdleActionModel:
    purchase_model: object
    margin_model: object
    action: str
    mean_margin: float
    margin_weight: float
    def predict(self,X):
        action_X=X.assign(_action=self.action)
        prob=self.purchase_model.predict_proba(action_X)[:,1]
        margin=np.full(len(X),self.mean_margin)
        if self.margin_model is not None:
            margin=self.margin_weight*self.margin_model.predict(action_X)+(1-self.margin_weight)*margin
        return np.asarray(prob*margin,float)


@dataclass
class ResidualActionModel:
    model: object
    offset: float
    def predict(self,X):
        return np.asarray(self.model.predict(X)+self.offset,float)


PROFIT_CANDIDATES={
    'ORIGINAL_TLEARNER':{'family':'tlearner'},
    'POOLED_STUMPS':{'family':'pooled','params':{'n_estimators':200,'max_depth':1,'min_child_weight':20,'reg_lambda':20}},
    'POOLED_D2_100':{'family':'pooled','params':{'n_estimators':100,'max_depth':2,'min_child_weight':20,'reg_lambda':20}},
    'POOLED_D2_200':{'family':'pooled','params':{'n_estimators':200,'max_depth':2,'min_child_weight':20,'reg_lambda':20}},
    'POOLED_D2_STRONG':{'family':'pooled','params':{'n_estimators':300,'max_depth':2,'min_child_weight':50,'reg_lambda':50}},
    'POOLED_D3_STRONG':{'family':'pooled','params':{'n_estimators':200,'max_depth':3,'min_child_weight':50,'reg_lambda':50}},
    'SHARED_BASE_EFFECTS':{'family':'shared','params':{'n_estimators':200,'max_depth':2,'min_child_weight':20,'reg_lambda':20}},
    'HURDLE_XGB_MEAN':{'family':'hurdle','purchase':'xgb','margin_weight':0.},
    'HURDLE_XGB_QUARTER':{'family':'hurdle','purchase':'xgb','margin_weight':.25},
    'HURDLE_XGB_HALF':{'family':'hurdle','purchase':'xgb','margin_weight':.5},
    'HURDLE_XGB_FULL':{'family':'hurdle','purchase':'xgb','margin_weight':1.},
    'HURDLE_LOGIT_MEAN':{'family':'hurdle','purchase':'logit','margin_weight':0.},
    'HURDLE_LOGIT_HALF':{'family':'hurdle','purchase':'logit','margin_weight':.5},
    'HURDLE_LOGIT_FULL':{'family':'hurdle','purchase':'logit','margin_weight':1.},
}


def fit_profit_models(data,num,cat,config,target='ncm_14d'):
    """Return four action predictors; config is frozen before final evaluation."""
    if isinstance(config,str):config=PROFIT_CANDIDATES[config]
    if not len(data) or set(data.offer_type)!=set(ACTIONS):
        raise ValueError('All four actions need training support')
    if 'assignment_source' in data and not data.assignment_source.eq('exploration').all():
        raise ValueError('Causal outcome models require randomized exploration training rows')
    if not np.isfinite(data[target]).all():raise ValueError('Targets must be finite')
    if config['family']=='tlearner':
        return fit_action_models(data,num,cat,dict(n_estimators=100,max_depth=1,learning_rate=.03),target=target)
    features=num+cat
    if config['family']=='shared':
        model=_xgb(num,cat,config.get('params',{})).fit(data[features],data[target])
        residual=data[target].to_numpy()-model.predict(data[features])
        return {a:ResidualActionModel(model,float(residual[data.offer_type.eq(a)].mean())) for a in ACTIONS}
    X=data[features].assign(_action=data.offer_type.to_numpy())
    if config['family']=='pooled':
        model=_xgb(num,cat+['_action'],config.get('params',{})).fit(X,data[target])
        return {a:PooledActionModel(model,a) for a in ACTIONS}
    if config['family']!='hurdle':raise ValueError('Unknown model family')
    if not data.purchase_14d.isin([0,1]).all() or data.purchase_14d.nunique()!=2:
        raise ValueError('Hurdle model needs both purchase outcomes')
    if not np.allclose(data.loc[data.purchase_14d.eq(0),target],0):
        raise ValueError('Hurdle factorization requires zero NCM for non-purchasers')
    if config.get('purchase','xgb')=='logit':
        purchase=Pipeline([('pre',_preprocess(num,cat+['_action'],True)),
                           ('model',LogisticRegression(C=.1,solver='lbfgs',max_iter=2000,random_state=42))])
    else:
        purchase=_xgb(num,cat+['_action'],config.get('purchase_params',{}),True)
    purchase.fit(X,data.purchase_14d)
    bought=data.purchase_14d.eq(1)
    margins=data.loc[bought].groupby('offer_type')[target].mean()
    if set(margins.index)!=set(ACTIONS):raise ValueError('Every action needs at least one purchaser')
    weight=float(config.get('margin_weight',.5))
    if not 0<=weight<=1:raise ValueError('Margin shrinkage weight must be in [0,1]')
    margin=None
    if weight:
        margin=_xgb(num,cat+['_action'],config.get('margin_params',dict(n_estimators=100,max_depth=2,min_child_weight=10,reg_lambda=30)))
        margin.fit(X.loc[bought],data.loc[bought,target])
    return {a:HurdleActionModel(purchase,margin,a,float(margins[a]),weight) for a in ACTIONS}
