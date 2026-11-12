"""As-of-safe return sensitivity and marginal support diagnostics.

The accrual horizon is an explicit sensitivity assumption, not an inferred
business rule. Marginal support checks do not establish joint causal overlap
or transportability to a new population.
"""
import numpy as np
import pandas as pd


def build_accrued_target(history, orders, returns, horizon_days=45, *, as_of_ts):
    """Return mature history rows with ``ncm_accrued_<horizon>d`` labels.

    Purchases remain restricted to completed same-customer/same-campaign orders
    in [send_ts, send_ts + 14d). Refunds on those orders count only before
    send_ts + horizon_days and at/before as_of_ts. A row is retained only when
    its entire accrual window has matured by as_of_ts, avoiding the optimistic
    labels that result from treating not-yet-observed returns as zero.

    Base revenue/COGS/shipping come from the existing 14-day outcome table.
    The primary ``ncm_14d`` column is preserved. The added target is a fixed-
    horizon sensitivity, not eventual lifetime profitability.
    """
    if not isinstance(horizon_days, (int, np.integer)) or isinstance(horizon_days, bool) or horizon_days < 14:
        raise ValueError("horizon_days must be an integer of at least 14 days")
    cutoff = pd.Timestamp(as_of_ts)
    if pd.isna(cutoff):
        raise ValueError("A valid as_of_ts is required")
    required = ["assignment_id", "customer_id", "campaign_id", "send_ts",
                "net_revenue_14d", "cogs_14d", "shipping_cost_14d"]
    if history[required].isna().any().any() or not history.assignment_id.is_unique:
        raise ValueError("Historical assignments require unique IDs and non-null outcome inputs")
    if not orders.order_id.is_unique:
        raise ValueError("Orders must have unique order_id")
    if "return_id" in returns and not returns.return_id.is_unique:
        raise ValueError("Returns must have unique return_id; deduplicate source records first")
    out = history.copy()
    out["accrual_end_ts"] = pd.to_datetime(out.send_ts) + pd.Timedelta(days=int(horizon_days))
    out = out.loc[out.accrual_end_ts <= cutoff].copy()
    if not np.isfinite(out[["net_revenue_14d", "cogs_14d", "shipping_cost_14d"]].to_numpy(float)).all():
        raise ValueError("Historical financial outcomes must be finite")
    if "is_completed" not in orders:
        raise ValueError("Orders must include the cleaned is_completed flag")
    eligible = out[["assignment_id", "customer_id", "campaign_id", "send_ts", "accrual_end_ts"]].merge(
        orders.loc[orders.is_completed.eq(True), ["order_id", "customer_id", "campaign_id", "order_ts"]],
        on=["customer_id", "campaign_id"], how="inner", validate="1:m"
    )
    eligible = eligible[(eligible.order_ts >= eligible.send_ts) &
                        (eligible.order_ts < eligible.send_ts + pd.Timedelta(days=14))]
    if not eligible.order_id.is_unique:
        raise ValueError("An order cannot contribute to more than one assignment")
    linked = eligible[["assignment_id", "order_id", "order_ts", "accrual_end_ts"]].merge(
        returns[["order_id", "return_ts", "refund_amount"]], on="order_id", how="inner", validate="1:m"
    )
    if linked[["return_ts", "refund_amount"]].isna().any().any():
        raise ValueError("Linked returns require timestamps and amounts")
    if not np.isfinite(linked.refund_amount.to_numpy(float)).all() or (linked.refund_amount < 0).any():
        raise ValueError("Refund amounts must be finite and nonnegative")
    if (linked.return_ts < linked.order_ts).any():
        raise ValueError("A refund cannot precede its order")
    linked = linked[(linked.return_ts < linked.accrual_end_ts) & (linked.return_ts <= cutoff)]
    refunds = linked.groupby("assignment_id").refund_amount.sum()
    refund_column = f"refund_amount_accrued_{horizon_days}d"
    target_column = f"ncm_accrued_{horizon_days}d"
    out[refund_column] = out.assignment_id.map(refunds).fillna(0.)
    out[target_column] = out.net_revenue_14d - out.cogs_14d - out.shipping_cost_14d - out[refund_column]
    out["label_as_of_ts"] = cutoff
    out["return_horizon_days"] = int(horizon_days)
    if not out.accrual_end_ts.le(out.label_as_of_ts).all():
        raise AssertionError("Accrued targets must be mature at their label cutoff")
    return out


def cold_history_mask(data):
    """Comparable first-exposure/inactive subgroup; unknown counts are not zero."""
    return data.prior_emails_delivered.eq(0) & data.orders_90d.eq(0)


def support_report(train, scoring, numeric_features, categorical_features):
    """One row per feature, contrasting exploration training and scoring data.

    Missing/zero/out-of-range/unseen rates all use the full corresponding row
    count as denominator. Numeric ranges use nonmissing finite training values.
    Out-of-range is undefined when training has no observed values; the separate
    no-training-support rate flags observed scoring values in that case.
    """
    features = list(numeric_features) + list(categorical_features)
    if len(features) != len(set(features)):
        raise ValueError("Numeric and categorical feature lists must be disjoint and unique")
    if len(train) == 0 or len(scoring) == 0:
        raise ValueError("Training and scoring populations must be non-empty")
    missing = set(features) - set(train.columns) | (set(features) - set(scoring.columns))
    if missing:
        raise ValueError(f"Missing support-report features: {sorted(missing)}")
    rows = []
    for feature in features:
        tr, sc = train[feature], scoring[feature]
        numeric = feature in numeric_features
        row = {"feature": feature, "type": "numeric" if numeric else "categorical",
               "train_rows": len(tr), "scoring_rows": len(sc),
               "train_missing_rate": float(tr.isna().mean()), "scoring_missing_rate": float(sc.isna().mean()),
               "train_distinct_observed": int(tr.nunique()), "scoring_distinct_observed": int(sc.nunique()),
               "train_all_missing": bool(tr.isna().all())}
        row["missing_rate_change"] = row["scoring_missing_rate"] - row["train_missing_rate"]
        row["scoring_no_training_support_rate"] = float(sc.notna().mean()) if tr.isna().all() else 0.
        if numeric:
            tr, sc = pd.to_numeric(tr), pd.to_numeric(sc)
            if not np.isfinite(tr.dropna()).all() or not np.isfinite(sc.dropna()).all():
                raise ValueError(f"Numeric support feature {feature!r} contains infinite values")
            lower, upper = tr.min(), tr.max()
            row.update({"train_zero_rate": float(tr.eq(0).mean()), "scoring_zero_rate": float(sc.eq(0).mean()),
                        "train_min": lower, "train_max": upper, "scoring_min": sc.min(), "scoring_max": sc.max(),
                        "scoring_out_of_range_rate": float(((sc < lower) | (sc > upper)).mean()) if tr.notna().any() else np.nan,
                        "scoring_unseen_category_rate": np.nan})
        else:
            row.update({"train_zero_rate": np.nan, "scoring_zero_rate": np.nan,
                        "train_min": np.nan, "train_max": np.nan, "scoring_min": np.nan, "scoring_max": np.nan,
                        "scoring_out_of_range_rate": np.nan,
                        "scoring_unseen_category_rate": float((sc.notna() & ~sc.isin(tr.dropna().unique())).mean())})
        rows.append(row)
    return pd.DataFrame(rows)
