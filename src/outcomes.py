"""Refund accounting for a fixed set of eligible outcome orders.

``all_linked`` is an accounting sensitivity, not permission to use future labels.
Supply ``as_of_ts`` when constructing labels available at a historical decision.
Even then, not-yet-observed returns imply censoring; use mature cohorts for training.
"""
import numpy as np
import pandas as pd


RETURN_RULES = {"in_window", "all_linked"}
REFUND_COLUMNS = [
    "refund_amount_applicable", "in_window_refund_amount",
    "post_window_refund_amount", "total_linked_refund_amount",
    "in_window_return_count", "post_window_return_count",
]


def inferred_waived_shipping_fee(gross, *, free_shipping_cutoff, standard_fee):
    """Forgone customer fee for an otherwise identical FREE_SHIP basket.

    The cutoff is an explicit assumption on merchandise before discounts.
    A basket exactly at the cutoff already qualifies for ordinary free shipping.
    Callers apply this only to eligible redeemed FREE_SHIP orders.
    """
    gross = np.asarray(gross, dtype=float)
    if not np.isfinite(free_shipping_cutoff) or free_shipping_cutoff <= 0:
        raise ValueError("Free-shipping cutoff must be finite and positive")
    if not np.isfinite(standard_fee) or standard_fee < 0:
        raise ValueError("Standard shipping fee must be finite and nonnegative")
    if not np.isfinite(gross).all() or (gross < 0).any():
        raise ValueError("Merchandise gross must be finite and nonnegative")
    return np.where(gross < free_shipping_cutoff, standard_fee, 0.)


def applicable_refunds(outcome_orders, returns, rule="in_window", as_of_ts=None):
    """Aggregate refunds once per assignment, retaining independent timing totals.

    ``outcome_orders`` must have unique order_id and assignment_id/outcome_end_ts.
    Multiple refund records per order are supported. A refund exactly at the
    outcome end is outside the half-open 14-day window. ``as_of_ts`` is inclusive.
    Without an as-of cutoff, the result describes all returns in the source
    snapshot and must not be assumed available at an earlier model-fit date.
    """
    if rule not in RETURN_RULES:
        raise ValueError(f"Unknown return rule {rule!r}; choose {sorted(RETURN_RULES)}")
    if not outcome_orders.order_id.is_unique:
        raise ValueError("Eligible outcome orders must have unique order_id")
    r = outcome_orders[["assignment_id", "order_id", "outcome_end_ts"]].merge(
        returns[["order_id", "return_ts", "refund_amount"]], on="order_id", validate="1:m"
    )
    if r[["return_ts", "outcome_end_ts", "refund_amount"]].isna().any().any():
        raise ValueError("Linked refunds require non-null timestamps and amounts")
    if not np.isfinite(r.refund_amount).all() or (r.refund_amount < 0).any():
        raise ValueError("Refund amounts must be finite and nonnegative")
    if as_of_ts is not None:
        r = r[r.return_ts <= pd.Timestamp(as_of_ts)]
    if r.empty:
        return pd.DataFrame(columns=REFUND_COLUMNS, index=pd.Index([], name="assignment_id"), dtype=float)
    r = r.assign(in_window=r.return_ts < r.outcome_end_ts)
    early = r[r.in_window].groupby("assignment_id").refund_amount
    late = r[~r.in_window].groupby("assignment_id").refund_amount
    result = pd.DataFrame({
        "in_window_refund_amount": early.sum(),
        "post_window_refund_amount": late.sum(),
        "total_linked_refund_amount": r.groupby("assignment_id").refund_amount.sum(),
        "in_window_return_count": early.size(),
        "post_window_return_count": late.size(),
    }).fillna(0)
    result["refund_amount_applicable"] = result[
        "in_window_refund_amount" if rule == "in_window" else "total_linked_refund_amount"
    ]
    return result[REFUND_COLUMNS]


def refund_reconciliation(refunds, linked_refund_total, rule="in_window", atol=0.05):
    """Return accounting checks that remain valid under either return rule."""
    if rule not in RETURN_RULES:
        raise ValueError(f"Unknown return rule {rule!r}")
    expected = refunds["in_window_refund_amount" if rule == "in_window" else "total_linked_refund_amount"]
    return {
        "refund timing partitions total": bool(np.allclose(
            refunds.in_window_refund_amount + refunds.post_window_refund_amount,
            refunds.total_linked_refund_amount, rtol=0, atol=atol)),
        "refund totals reconcile to linked returns": bool(np.isclose(
            refunds.total_linked_refund_amount.sum(), linked_refund_total, rtol=0, atol=atol)),
        "applicable refunds match selected rule": bool(np.allclose(
            refunds.refund_amount_applicable, expected, rtol=0, atol=atol)),
    }
