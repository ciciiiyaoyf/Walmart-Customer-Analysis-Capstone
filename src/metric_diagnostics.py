"""Diagnostics for signed profit and the separately interpretable hurdle parts.

All scores concern factual outcomes under the recorded action. Neither a profit
calibration curve nor purchase recall is a treatment-effect or policy-value test.
"""
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, brier_score_loss, log_loss, roc_auc_score

from src.evaluation import cluster_mean_interval
from src.modeling import ACTIONS, reg_metrics


def regression_diagnostic(y, prediction, customers, bootstrap_draws=2000):
    """Signed mean error plus dollar losses; empty groups stay explicitly unknown."""
    y, prediction, customers = np.asarray(y, float), np.asarray(prediction, float), np.asarray(customers)
    if y.ndim != 1 or prediction.ndim != 1 or customers.ndim != 1 or not (len(y) == len(prediction) == len(customers)):
        raise ValueError("Matching one-dimensional outcomes, predictions, and customer IDs required")
    if not np.isfinite(y).all() or not np.isfinite(prediction).all() or pd.isna(customers).any():
        raise ValueError("Finite outcomes/predictions and non-null customer IDs required")
    if not len(y):
        return {"n": 0, **{key: np.nan for key in ["rmse", "mae", "r2", "obs_mean", "pred_mean",
                "mean_prediction_bias", "bias_ci_low", "bias_ci_high"]}}
    error = prediction-y
    mean, low, high = cluster_mean_interval(error, customers, draws=bootstrap_draws)
    return {**reg_metrics(y, prediction), "mean_prediction_bias": mean, "bias_ci_low": low, "bias_ci_high": high}


def _scopes(frame):
    yield "pooled", frame
    for action in ACTIONS:
        yield action, frame.loc[frame.offer_type.eq(action)]


def _validate_predictions(frame, prediction="factual_prediction"):
    required = {"assignment_id", "customer_id", "model", "offer_type", "ncm_14d", prediction}
    if not required.issubset(frame):
        raise ValueError(f"Missing diagnostic columns: {sorted(required-set(frame))}")
    if frame.empty or frame[list(required)].isna().any().any():
        raise ValueError("Nonempty diagnostic rows and non-null required fields needed")
    if frame.duplicated(["model", "assignment_id"]).any():
        raise ValueError("One prediction per model and assignment is required")
    if not frame.offer_type.isin(ACTIONS).all():
        raise ValueError("Unknown factual offer")
    if not np.isfinite(frame[["ncm_14d", prediction]].to_numpy(float)).all():
        raise ValueError("Finite diagnostic targets/predictions required")


def profit_bias_table(predictions, bootstrap_draws=2000):
    _validate_predictions(predictions)
    rows = []
    for model, group in predictions.groupby("model", sort=True):
        for scope, part in _scopes(group):
            rows.append({"model": model, "scope": scope,
                **regression_diagnostic(part.ncm_14d, part.factual_prediction, part.customer_id, bootstrap_draws)})
    return pd.DataFrame(rows)


def calibration_table(predictions, bins=5, bootstrap_draws=2000):
    """Approximate equal-frequency bands; identical predictions never split.

    Quantile edges are determined only by predictions. Outcomes do not assign
    bands. Duplicate edges/empty bands are removed, so ties can yield fewer bins.
    Bias intervals condition on fitted models and fixed prediction-derived bands.
    """
    _validate_predictions(predictions)
    if not isinstance(bins, (int, np.integer)) or isinstance(bins, bool) or bins < 1:
        raise ValueError("A positive integer number of calibration bins is required")
    rows = []
    for model, group in predictions.groupby("model", sort=True):
        for scope, part in _scopes(group):
            if part.empty:
                continue
            p = part.factual_prediction
            if p.nunique() <= 1 or bins == 1:
                band = pd.Series(0, index=part.index)
            else:
                edges = np.unique(np.quantile(p, np.linspace(0, 1, bins+1)))
                # Outer endpoints include every prediction without splitting ties.
                edges[0], edges[-1] = -np.inf, np.inf
                band = pd.cut(p, bins=edges, labels=False, include_lowest=True)
            for number, (_, block) in enumerate(part.groupby(band, observed=True, sort=True), start=1):
                diagnostic = regression_diagnostic(block.ncm_14d, block.factual_prediction,
                                                   block.customer_id, bootstrap_draws)
                rows.append({"model": model, "scope": scope, "bin": number, "n": len(block),
                    "prediction_min": float(block.factual_prediction.min()),
                    "prediction_max": float(block.factual_prediction.max()),
                    **{key: diagnostic[key] for key in ["obs_mean", "pred_mean", "mean_prediction_bias",
                                                        "bias_ci_low", "bias_ci_high"]}})
    return pd.DataFrame(rows)


def hurdle_components(models, data, features):
    """Extract probability and signed conditional margin for each observed action.

    Only the blended margin is a conditional purchaser prediction. Multiplying
    it by purchase probability produces unconditional expected NCM instead.
    """
    required = ["assignment_id", "customer_id", "campaign_id", "offer_type", "ncm_14d", "purchase_14d"]
    if data.empty or data[required].isna().any().any() or not data.assignment_id.is_unique:
        raise ValueError("Unique complete decision rows required for hurdle components")
    if not data.offer_type.isin(ACTIONS).all() or not data.purchase_14d.isin([0, 1]).all():
        raise ValueError("Known factual offers and binary purchase labels required")
    # Assignment IDs, not an incidental dataframe index, identify the decisions.
    data = data.reset_index(drop=True)
    out = data[required].copy()
    out["purchase_probability"] = np.nan
    out["conditional_margin_prediction"] = np.nan
    out["factual_prediction"] = np.nan
    for action, part in data.groupby("offer_type", observed=True, sort=False):
        model = models[action]
        if not all(hasattr(model, field) for field in ["purchase_model", "margin_model", "mean_margin", "margin_weight", "action"]):
            raise ValueError("Component diagnostics require a two-part hurdle model")
        if model.action != action:
            raise ValueError("Hurdle model action does not match its dictionary key")
        x = part[features].assign(_action=action)
        probability = np.asarray(model.purchase_model.predict_proba(x), float)[:, 1]
        margin = np.full(len(part), model.mean_margin, dtype=float)
        if model.margin_model is not None:
            margin = model.margin_weight*np.asarray(model.margin_model.predict(x), float)+(1-model.margin_weight)*margin
        expected = probability*margin
        np.testing.assert_allclose(expected, model.predict(part[features]), rtol=1e-10, atol=1e-10)
        out.loc[part.index, "purchase_probability"] = probability
        out.loc[part.index, "conditional_margin_prediction"] = margin
        out.loc[part.index, "factual_prediction"] = expected
    if not np.isfinite(out[["purchase_probability", "conditional_margin_prediction", "factual_prediction"]]).all().all():
        raise ValueError("Finite hurdle component forecasts required")
    if not out.purchase_probability.between(0, 1).all():
        raise ValueError("Purchase probabilities must lie in [0,1]")
    return out


def hurdle_purchase_table(components, threshold):
    """Purchase diagnostics at a previously frozen global threshold.

    AP/recall are undefined for a group with no observed positive outcomes;
    precision is undefined when no customers cross the threshold. Undefined
    values remain NaN rather than falsely recording performance of zero.
    """
    _validate_predictions(components)
    if not np.isfinite(threshold) or not 0 <= threshold <= 1:
        raise ValueError("A fixed probability threshold in [0,1] is required")
    rows = []
    for model, group in components.groupby("model", sort=True):
        for scope, part in _scopes(group):
            labels, p = part.purchase_14d.to_numpy(float), part.purchase_probability.to_numpy(float)
            if not np.isin(labels, [0, 1]).all() or not np.isfinite(p).all() or ((p < 0) | (p > 1)).any():
                raise ValueError("Binary outcomes and probabilities in [0,1] required")
            y = labels.astype(int)
            flagged = p >= threshold
            tp, fp = int(((y == 1) & flagged).sum()), int(((y == 0) & flagged).sum())
            fn, tn = int(((y == 1) & ~flagged).sum()), int(((y == 0) & ~flagged).sum())
            buyers = int(y.sum())
            row = {"model": model, "scope": scope, "n": len(y), "buyers": buyers,
                "average_precision": float(average_precision_score(y, p)) if buyers else np.nan,
                "recall": tp/buyers if buyers else np.nan,
                "precision": tp/(tp+fp) if tp+fp else np.nan,
                "f1": 2*tp/(2*tp+fp+fn) if 2*tp+fp+fn else np.nan,
                "threshold": float(threshold), "predicted_positive_share": float(flagged.mean()) if len(y) else np.nan,
                "tp": tp, "fp": fp, "tn": tn, "fn": fn,
                "roc_auc": float(roc_auc_score(y, p)) if len(np.unique(y)) == 2 else np.nan,
                "log_loss": float(log_loss(y, p, labels=[0, 1])) if len(y) else np.nan,
                "brier": float(brier_score_loss(y, p)) if len(y) else np.nan}
            rows.append(row)
    return pd.DataFrame(rows)


def purchaser_margin_table(components, bootstrap_draws=2000):
    """Score the conditional margin forecast only on actual purchasers."""
    _validate_predictions(components, prediction="conditional_margin_prediction")
    if not components.purchase_14d.isin([0, 1]).all():
        raise ValueError("Binary purchase labels required for purchaser-only metrics")
    rows = []
    for model, group in components.groupby("model", sort=True):
        for scope, part in _scopes(group):
            part = part.loc[part.purchase_14d.eq(1)]
            rows.append({"model": model, "scope": scope,
                **regression_diagnostic(part.ncm_14d, part.conditional_margin_prediction,
                                        part.customer_id, bootstrap_draws)})
    return pd.DataFrame(rows)
