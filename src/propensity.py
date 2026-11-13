"""Pre-decision feature engineering and temporal calibration for purchase propensity.

This predicts purchase under the historical treatment mixture, not treatment
assignment probabilities or the probability of buying without an offer.
"""
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler, SplineTransformer
from sklearn.metrics import roc_auc_score, average_precision_score, log_loss, brier_score_loss

RATE_COLUMNS = {"is_member", "marketing_consent", "prior_offer_redemption_rate",
                "historical_return_rate", "historical_open_rate", "historical_click_rate", "cart_add_rate_30d"}
SPLINE_COLUMNS = ["customer_tenure_days", "days_since_last_order", "gross_revenue_90d",
                  "sessions_30d", "days_since_last_session", "prior_emails_delivered",
                  "historical_open_rate", "avg_order_value_90d"]
MODES = ("raw", "log", "log_interactions", "log_splines")
PURCHASE_SELECTION_RULE = ("maximize mean out-of-time PR-AUC with log_loss <= original + .001 and "
                           "Brier <= original + .0005; ROC-AUC tie-break. Select nested temporal sigmoid "
                           "only if both log-loss and Brier improve.")


def purchase_candidates():
    return [{"mode":mode,"C":C,"calibration":"none"} for mode in MODES for C in [.001,.01,.1,1.]]


def purchase_metrics(y,p):
    return {"roc_auc":roc_auc_score(y,p),"pr_auc":average_precision_score(y,p),
            "log_loss":log_loss(y,p),"brier":brier_score_loss(y,p),
            "observed_mean":float(np.mean(y)),"predicted_mean":float(np.mean(p))}


class PurchaseFeatures(BaseEstimator, TransformerMixin):
    """Stateless transforms use only the supplied pre-decision predictors."""
    def __init__(self, numeric_features, categorical_features, mode="log"):
        self.numeric_features = numeric_features
        self.categorical_features = categorical_features
        self.mode = mode

    def fit(self, X, y=None):
        if self.mode not in MODES:
            raise ValueError(f"Unsupported feature mode: {self.mode}")
        missing = set(self.numeric_features + self.categorical_features) - set(X)
        if missing:
            raise ValueError(f"Missing registered pre-decision features: {sorted(missing)}")
        return self

    def transform(self, X):
        out = X[self.numeric_features + self.categorical_features].copy()
        if self.mode != "raw":
            for c in self.numeric_features:
                if c not in RATE_COLUMNS:
                    v = pd.to_numeric(out[c], errors="raise")
                    out[c] = np.sign(v) * np.log1p(np.abs(v))
        if self.mode == "log_interactions":
            tier = X.customer_tier.fillna("Missing").astype(str)
            member = X.is_member.fillna(0).astype(int)
            out["tier_membership"] = tier + "_" + member.astype(str)
            out["preferred_matches_browse"] = (X.preferred_category.notna() & X.top_category_viewed_30d.notna()
                                                & X.preferred_category.eq(X.top_category_viewed_30d)).astype(float)
            out["purchase_matches_browse"] = (X.top_purchase_category_90d.notna() & X.top_category_viewed_30d.notna()
                                               & X.top_purchase_category_90d.eq(X.top_category_viewed_30d)).astype(float)
            out["recent_purchase"] = X.days_since_last_order.le(30).astype(float)
            out["recent_browse"] = X.days_since_last_session.le(7).astype(float)
            out["session_recency_intensity"] = out.sessions_30d / (1 + out.days_since_last_session)
            out["member_sessions"] = member * out.sessions_30d
            out["member_spend"] = member * out.gross_revenue_90d
            for value in ("Gold", "Silver"):
                out[f"{value.lower()}_sessions"] = tier.eq(value) * out.sessions_30d
                out[f"{value.lower()}_spend"] = tier.eq(value) * out.gross_revenue_90d
        return out


def engineered_columns(num, cat, mode):
    num, cat = list(num), list(cat)
    if mode == "log_interactions":
        num += ["preferred_matches_browse", "purchase_matches_browse", "recent_purchase", "recent_browse",
                "session_recency_intensity", "member_sessions", "member_spend", "gold_sessions",
                "gold_spend", "silver_sessions", "silver_spend"]
        cat += ["tier_membership"]
    return num, cat


def make_purchase_pipeline(num, cat, config):
    mode = config.get("mode", "raw")
    en_num, en_cat = engineered_columns(num, cat, mode)
    transformations = [
        ("num", Pipeline([("imp", SimpleImputer(strategy="median", add_indicator=True, keep_empty_features=True)),
                          ("scale", StandardScaler())]), en_num),
        ("cat", Pipeline([("imp", SimpleImputer(strategy="constant", fill_value="Missing")),
                          ("ohe", OneHotEncoder(handle_unknown="ignore"))]), en_cat)]
    if mode == "log_splines":
        cols = [c for c in SPLINE_COLUMNS if c in en_num]
        transformations.append(("splines", Pipeline([
            ("imp", SimpleImputer(strategy="median", keep_empty_features=True)),
            ("spline", SplineTransformer(n_knots=4, degree=2, knots="uniform", extrapolation="constant", include_bias=False)),
            ("scale", StandardScaler())]), cols))
    return Pipeline([
        ("features", PurchaseFeatures(list(num), list(cat), mode)),
        ("pre", ColumnTransformer(transformations)),
        ("lr", LogisticRegression(C=float(config.get("C", .01)), max_iter=2000, solver="lbfgs", random_state=42))])


class CalibratedPurchaseModel:
    """Platt map fitted from prior out-of-time predictions; final base model uses all passed rows."""
    def __init__(self, base, calibrator, calibration_campaigns):
        self.base = base
        self.calibrator = calibrator
        self.calibration_campaigns = list(calibration_campaigns)
        self.classes_ = np.asarray([0, 1])

    def predict_proba(self, X):
        raw = np.clip(self.base.predict_proba(X)[:, 1], 1e-8, 1-1e-8)
        logit = np.log(raw / (1-raw)).reshape(-1, 1)
        return self.calibrator.predict_proba(logit)

    def predict(self, X):
        return self.predict_proba(X).argmax(axis=1)


def fit_purchase_model(data, num, cat, config):
    """Fit on passed rows only; temporal calibration never uses future campaign labels."""
    if "purchase_14d" not in data:
        raise ValueError("Missing purchase target")
    base = make_purchase_pipeline(num, cat, config)
    if config.get("calibration", "none") == "none":
        return base.fit(data[num+cat], data.purchase_14d)
    if config["calibration"] != "sigmoid":
        raise ValueError("calibration must be none or sigmoid")
    campaigns = sorted(data.campaign_id.unique())
    if len(campaigns) < 3:
        raise ValueError("Temporal calibration needs at least three training campaigns")
    probabilities, labels, used = [], [], []
    for campaign in campaigns[1:]:
        tr = data[data.campaign_id < campaign]
        va = data[data.campaign_id == campaign]
        if 'send_ts' in data and tr.send_ts.max()+pd.Timedelta(days=14) > va.send_ts.min():
            raise ValueError('Training outcomes must mature before calibration decisions')
        model = make_purchase_pipeline(num, cat, config).fit(tr[num+cat], tr.purchase_14d)
        probabilities.extend(model.predict_proba(va[num+cat])[:, 1])
        labels.extend(va.purchase_14d)
        used.append(campaign)
    p = np.clip(np.asarray(probabilities), 1e-8, 1-1e-8)
    calibrator = LogisticRegression(C=1., max_iter=1000).fit(np.log(p/(1-p)).reshape(-1, 1), labels)
    base.fit(data[num+cat], data.purchase_14d)
    return CalibratedPurchaseModel(base, calibrator, used)


def select_purchase_config(history, num, cat, validation_campaigns=None, verbose=False):
    """Select using only historical development campaigns before CP07.

    A caller may pass a shorter history for nested policy evaluation. By default
    the latest three available campaigns after the first two are validation folds.
    CP07 is never used, even if the caller supplies the full historical table.
    Returns (config, fold_metrics, candidate_summary).
    """
    development=history[history.campaign_id.lt("CP07")]
    campaigns=sorted(development.campaign_id.unique())
    validation_campaigns=(campaigns[2:][-3:] if validation_campaigns is None else list(validation_campaigns))
    if any(c >= "CP07" or c not in campaigns for c in validation_campaigns):
        raise ValueError("Purchase selection permits only available development campaigns before CP07")
    if not validation_campaigns:
        return {"mode":"raw","C":.01,"calibration":"none"},pd.DataFrame(),pd.DataFrame()
    rows=[]
    def evaluate(config):
        results=[]
        for campaign in validation_campaigns:
            tr=development[development.campaign_id.lt(campaign)]
            va=development[development.campaign_id.eq(campaign)]
            if not len(tr):raise ValueError("Every validation campaign needs earlier fitting observations")
            if 'send_ts' in development and tr.send_ts.max()+pd.Timedelta(days=14) > va.send_ts.min():
                raise ValueError('Training outcomes must mature before validation decisions')
            fitted=fit_purchase_model(tr,num,cat,config)
            p=fitted.predict_proba(va[num+cat])[:,1]
            results.append({"campaign":campaign,**config,**purchase_metrics(va.purchase_14d,p)})
        return results
    keys=['mode','C','calibration']; score_cols=['roc_auc','pr_auc','log_loss','brier']
    for config in purchase_candidates():
        result=evaluate(config);rows.extend(result)
        if verbose:print(config,pd.DataFrame(result)[score_cols].mean().to_dict(),flush=True)
    folds=pd.DataFrame(rows)
    summary=folds.groupby(keys,as_index=False)[score_cols].mean()
    original=summary.query("mode == 'raw' and C == .01").iloc[0]
    eligible=summary[(summary.log_loss <= original.log_loss+.001) & (summary.brier <= original.brier+.0005)]
    winner=eligible.sort_values(['pr_auc','roc_auc','log_loss'],ascending=[False,False,True]).iloc[0]
    config={'mode':winner['mode'],'C':float(winner.C),'calibration':'none'}
    # A calibrator needs at least two prior out-of-time prediction blocks.
    if all(sum(c < v for c in campaigns) >= 3 for v in validation_campaigns):
        calibrated=dict(config,calibration='sigmoid')
        cal=pd.DataFrame(evaluate(calibrated))
        cm=cal[score_cols].mean()
        if cm.log_loss < winner.log_loss and cm.brier <= winner.brier:
            config=calibrated
        folds=pd.concat([folds,cal],ignore_index=True)
    summary=folds.groupby(keys,as_index=False)[score_cols].mean().sort_values('pr_auc',ascending=False)
    return config,folds,summary
