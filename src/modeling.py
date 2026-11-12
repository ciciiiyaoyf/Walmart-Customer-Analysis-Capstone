"""Shared model plumbing for Steps 06 and 08: feature lists from the Step 02 schema and the XGBoost pipeline factory."""
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from xgboost import XGBRegressor
from src.common import TABLES

ACTIONS = ["NO_OFFER", "PCT_10", "PCT_20", "FREE_SHIP"]
COMMERCIAL = ACTIONS[1:]
XGB_FIXED = dict(objective="reg:squarederror", min_child_weight=10, subsample=0.8, colsample_bytree=0.8, reg_lambda=5.0,
                 tree_method="hist", random_state=42, n_jobs=4)
XGB_GRID = [dict(n_estimators=n, max_depth=d, learning_rate=lr) for n in (200, 400) for d in (2, 3) for lr in (0.03, 0.05)]
XGB_GRID += [dict(n_estimators=100, max_depth=1, learning_rate=0.03), dict(n_estimators=200, max_depth=1, learning_rate=0.03),
             dict(n_estimators=100, max_depth=2, learning_rate=0.03)]  # more-regularized corner: CP06 favoured the smallest models


def feature_lists():
    """Numeric / categorical predictors registered by Step 02 (context columns excluded)."""
    fs = pd.read_csv(TABLES / "feature_schema.csv").query("family != 'context'")
    return fs[fs.type == "numeric"].feature.tolist(), fs[fs.type == "categorical"].feature.tolist()


def make_xgb_pipe(num, cat, params):
    pre = ColumnTransformer([
        ("num", SimpleImputer(strategy="median", add_indicator=True), num),
        ("cat", Pipeline([("imp", SimpleImputer(strategy="constant", fill_value="Missing")),
                          ("ohe", OneHotEncoder(handle_unknown="ignore"))]), cat)])
    return Pipeline([("pre", pre), ("xgb", XGBRegressor(**XGB_FIXED, **params))])


def reg_metrics(y, p):
    y, p = np.asarray(y, float), np.asarray(p, float)
    if not len(y) or len(y) != len(p) or not np.isfinite(y).all() or not np.isfinite(p).all():
        raise ValueError("Regression metrics require equally sized finite nonempty arrays")
    sst = ((y - y.mean()) ** 2).sum()
    return {"n": len(y), "rmse": mean_squared_error(y, p) ** 0.5, "mae": mean_absolute_error(y, p),
            "r2": float(1 - ((y - p) ** 2).sum() / sst) if sst > 0 else np.nan,
            "obs_mean": y.mean(), "pred_mean": p.mean()}


class ConstantPredictor:
    """Action-mean reference; prediction interface matches the action pipelines."""
    def __init__(self, mean):
        self.mean = float(mean)

    def predict(self, X):
        return np.full(len(X), self.mean)


class ShrunkPredictor:
    def __init__(self, model, mean, weight=0.5):
        self.model, self.mean, self.weight = model, float(mean), float(weight)

    def predict(self, X):
        return self.weight * self.model.predict(X) + (1 - self.weight) * self.mean


class ActionPredictor:
    """A pooled outcome model evaluated with one fixed treatment action."""
    def __init__(self, model, action):
        self.model, self.action = model, action

    def predict(self, X):
        return self.model.predict(X.assign(_action=self.action))


CANDIDATES = ["ORIGINAL_TLEARNER", "SHRUNK_TLEARNER", "ACTION_MEAN", "POOLED_XGBOOST"]


def fit_action_models(data, num, cat, params, method="ORIGINAL_TLEARNER", target="ncm_14d", original=None):
    """Only passed training rows are used; caller owns time split and label maturity."""
    if method not in CANDIDATES:
        raise ValueError(f"Unknown method {method}")
    if set(data.offer_type.unique()) != set(ACTIONS):
        raise ValueError("Every action needs training support")
    if not np.isfinite(data[target]).all():
        raise ValueError("Nonfinite training targets")
    features = num + cat
    means = data.groupby("offer_type")[target].mean()
    if method == "ACTION_MEAN":
        return {a: ConstantPredictor(means[a]) for a in ACTIONS}
    if method == "POOLED_XGBOOST":
        cfg = dict(n_estimators=200, max_depth=2, learning_rate=0.03)
        model = make_xgb_pipe(num, cat + ["_action"], cfg).fit(
            data[features].assign(_action=data.offer_type.values), data[target])
        return {a: ActionPredictor(model, a) for a in ACTIONS}
    models = original or {a: make_xgb_pipe(num, cat, params).fit(
        data.loc[data.offer_type == a, features], data.loc[data.offer_type == a, target]) for a in ACTIONS}
    if method == "SHRUNK_TLEARNER":
        return {a: ShrunkPredictor(models[a], means[a], 0.5) for a in ACTIONS}
    return models


def predict_actions(models, X):
    result = np.column_stack([models[a].predict(X) for a in ACTIONS]).astype(float)
    if result.shape != (len(X), len(ACTIONS)) or not np.isfinite(result).all():
        raise ValueError("Invalid potential-outcome predictions")
    return result


def make_propensity_pipe(num, cat, C=0.01):
    """Purchase probability, not treatment propensity; fitted separately per time fold."""
    pre = ColumnTransformer([
        ("num", Pipeline([("imp", SimpleImputer(strategy="median", add_indicator=True)),
                          ("sc", StandardScaler())]), num),
        ("cat", Pipeline([("imp", SimpleImputer(strategy="constant", fill_value="Missing")),
                          ("ohe", OneHotEncoder(handle_unknown="ignore"))]), cat)])
    return Pipeline([("pre", pre), ("lr", LogisticRegression(C=C, solver="lbfgs", max_iter=2000, random_state=42))])
