"""Build factual-profit diagnostics and leakage-safe two-part component scores."""
import json
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.experiment_profit import require_mature_split
from src.metric_diagnostics import (profit_bias_table, calibration_table, hurdle_components,
                                    hurdle_purchase_table, purchaser_margin_table)
from src.modeling import feature_lists
from src.profit_models import fit_profit_models


def rolling_hurdle_components(history, num, cat, config, campaigns=("CP04", "CP05", "CP06")):
    """Earlier-only fits for threshold development, never CP07 outcomes."""
    if not campaigns or any(campaign not in ("CP04", "CP05", "CP06") for campaign in campaigns):
        raise ValueError("Threshold development is restricted to CP04-CP06")
    exploration = history.loc[history.assignment_source.eq("exploration")]
    records = []
    for campaign in campaigns:
        train = exploration.loc[exploration.campaign_id.lt(campaign)]
        validation = exploration.loc[exploration.campaign_id.eq(campaign)].sort_values("customer_id")
        require_mature_split(train, validation)
        models = fit_profit_models(train, num, cat, config)
        records.append(hurdle_components(models, validation, num+cat))
    return pd.concat(records, ignore_index=True)


def align_saved_predictions(saved, expected):
    """Align forecasts by decision ID and require matching factual labels/actions."""
    if not saved.assignment_id.is_unique or not expected.assignment_id.is_unique:
        raise ValueError("Unique assignment IDs required for saved prediction comparison")
    if set(saved.assignment_id) != set(expected.assignment_id):
        raise ValueError("Saved and diagnostic assignment sets differ")
    saved = saved.set_index("assignment_id").loc[expected.assignment_id]
    for column in ["customer_id", "campaign_id", "offer_type", "purchase_14d"]:
        if not np.array_equal(saved[column].to_numpy(), expected[column].to_numpy()):
            raise ValueError(f"Saved {column} differs from the diagnostic population")
    np.testing.assert_allclose(saved.ncm_14d, expected.ncm_14d, rtol=1e-12, atol=1e-12)
    return saved


def build_metric_diagnostics(history=None, tables=None, profit_out=None, bootstrap_draws=2000):
    """Add diagnostics without changing any fitted production models or policy.

    Development chooses a reporting threshold only. Factual profit forecasts are
    checked against saved results, and all downstream policy forecasts remain
    continuous expected-dollar predictions, independent of this threshold.
    """
    from src.propensity import choose_purchase_threshold
    tables = Path(tables) if tables is not None else ROOT / "outputs/tables"
    profit_out = Path(profit_out) if profit_out is not None else ROOT / "outputs/experiments/profit"
    tables.mkdir(parents=True, exist_ok=True)
    if history is None:
        history = pd.read_parquet(ROOT / "data/processed/historical_feature_outcome.parquet")
    selection = json.loads((profit_out / "selection.json").read_text(encoding="utf-8"))
    name, config = selection["selected_model"], selection["config"]
    if config.get("family") != "hurdle":
        raise ValueError("Selected model is not a hurdle model; component diagnostics require an explicit two-part candidate")
    num, cat = feature_lists()
    development = rolling_hurdle_components(history, num, cat, config)
    threshold = choose_purchase_threshold(development.purchase_14d, development.purchase_probability)
    development.assign(model=name).to_csv(tables / "hurdle_development_component_predictions.csv", index=False)
    threshold_record = {
        "model": name, "threshold": float(threshold), "rule": "maximize pooled out-of-time F1; lowest threshold wins ties",
        "comparison": "purchase_probability >= threshold", "selection_campaigns": ["CP04", "CP05", "CP06"],
        "training_rule": "each validation campaign is predicted using only earlier randomized exploration rows with mature outcomes",
        "development_rows": len(development), "development_buyers": int(development.purchase_14d.sum()),
        "cp07_used_for_threshold_selection": False, "same_threshold_for_every_action": True,
        "purpose": "report purchase recall and precision; not a promotion-allocation threshold",
        "uncertainty": "Model configuration was also selected on development data. Development diagnostics are selection-biased; CP07 is previously inspected retrospective evaluation, not a new confirmatory test.",
    }
    # Freeze before accessing saved CP07 models, probabilities, or outcomes.
    (tables / "hurdle_threshold_selection.json").write_text(json.dumps(threshold_record, indent=2)+"\n", encoding="utf-8")
    test = history.loc[history.campaign_id.eq("CP07") & history.assignment_source.eq("exploration")].sort_values("customer_id")
    train = history.loc[history.campaign_id.lt("CP07") & history.assignment_source.eq("exploration")]
    require_mature_split(train, test)
    models = joblib.load(profit_out / f"{name}_cp07_models.joblib")
    components = hurdle_components(models, test, num+cat).assign(model=name)
    predictions = pd.read_csv(profit_out / "cp07_predictions.csv")
    for model, saved in predictions.groupby("model", sort=True):
        aligned = align_saved_predictions(saved, test)
        if model == name:
            np.testing.assert_allclose(aligned.factual_prediction, components.factual_prediction, rtol=1e-10, atol=1e-10)
    if name not in set(predictions.model):
        raise ValueError("Saved CP07 predictions do not include the selected hurdle model")
    outputs = {
        "profit_bias_by_offer.csv": profit_bias_table(predictions, bootstrap_draws),
        "profit_calibration_by_offer.csv": calibration_table(predictions, bins=5, bootstrap_draws=bootstrap_draws),
        "hurdle_purchase_metrics.csv": hurdle_purchase_table(components, threshold),
        "purchaser_margin_metrics.csv": purchaser_margin_table(components, bootstrap_draws),
        "hurdle_component_predictions.csv": components,
    }
    for file, frame in outputs.items():
        frame.to_csv(tables / file, index=False)
    metadata = {
        "evaluation_campaign": "CP07", "evaluation_status": "retrospective; previously inspected",
        "profit_prediction_reference": "saved original and selected models trained on randomized CP01-CP06",
        "mean_prediction_bias_definition": "mean(predicted NCM - observed NCM); positive means overprediction",
        "calibration": "five approximate equal-frequency prediction bands per model and offer; ties never split and duplicate edges can reduce bin count",
        "bias_intervals": f"95% customer-cluster bootstrap with {bootstrap_draws} draws, conditional on fixed fits and fixed prediction bands; no multiplicity, selection, retraining or future-campaign adjustment",
        "component_definition": "expected NCM = purchase probability * signed blended conditional purchaser margin",
        "margin_population": "actual purchasers only, compared with the conditional margin prediction (not expected NCM)",
        "purchase_headline_metrics": ["average_precision", "recall"], "recall_threshold": float(threshold),
        "purchase_supporting_metrics": ["precision", "predicted_positive_share", "roc_auc", "log_loss", "brier"],
        "undefined_metrics": "AP and recall are missing when no buyers occur; precision is missing when no positives are predicted; R2 is missing for constant/empty targets",
        "saved_factual_profit_predictions_match_components": True,
        "policy_interpretation": "Factual prediction/component metrics do not establish causal uplift. Use the separately saved constrained-policy incremental NCM estimates and confidence intervals.",
        "files": list(outputs) + ["hurdle_threshold_selection.json", "hurdle_development_component_predictions.csv"],
    }
    (tables / "metric_diagnostics_metadata.json").write_text(json.dumps(metadata, indent=2)+"\n", encoding="utf-8")
    print(outputs["hurdle_purchase_metrics.csv"][["scope", "n", "buyers", "average_precision", "recall", "precision", "threshold"]].to_string(index=False))
    print(outputs["purchaser_margin_metrics.csv"].to_string(index=False))
    return metadata


if __name__ == "__main__":
    build_metric_diagnostics()
