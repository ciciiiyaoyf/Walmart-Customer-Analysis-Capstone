"""Distinguish observed transactions from verified complete purchase history.

The last observed transaction is a diagnostic, never a data-coverage watermark.
Recent counts remain counts of supplied completed orders. Their complete-history
counterparts are unknown unless an explicit, already-available source contract
certifies all transaction sources for the entire requested window.
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd


DEFAULT_MANIFEST = Path(__file__).resolve().parents[1] / "config" / "transaction_coverage.json"
WINDOWS = (30, 90)
ORGANIC_SOURCE = "organic_preperiod"
CAMPAIGN_SOURCE = "campaign_attributed"


def _timestamp(value, label):
    result = pd.Timestamp(value)
    if pd.isna(result):
        raise ValueError(f"{label} requires an explicit valid timestamp")
    return result


def _spine(spine, key, cut):
    columns = list(dict.fromkeys([key, "customer_id", cut]))
    if not set(columns).issubset(spine):
        raise ValueError("Decision spine requires a key, customer_id, and cutoff")
    result = spine[columns].copy()
    if result[columns].isna().any().any() or not result[key].is_unique:
        raise ValueError("Decision keys must be unique and identifiers/cutoffs non-null")
    result[cut] = pd.to_datetime(result[cut], errors="raise")
    return result


def _orders(orders):
    required = {"order_id", "customer_id", "order_ts", "order_source", "is_completed"}
    if not required.issubset(orders):
        raise ValueError(f"Missing cleaned-order columns: {sorted(required-set(orders))}")
    if orders[list(required)].isna().any().any() or not orders.order_id.is_unique:
        raise ValueError("Orders require unique IDs and non-null transaction inputs")
    if not orders.is_completed.isin([True, False]).all():
        raise ValueError("is_completed must contain boolean or binary flags")
    result = orders.copy()
    result["order_ts"] = pd.to_datetime(result.order_ts, errors="raise")
    return result


def _intervals(manifest):
    if manifest is None:
        manifest = DEFAULT_MANIFEST
    if isinstance(manifest, (str, Path)):
        manifest = json.loads(Path(manifest).read_text(encoding="utf-8"))
    if not isinstance(manifest, dict) or not isinstance(manifest.get("coverage_intervals", []), list):
        raise ValueError("Coverage manifest must contain a coverage_intervals list")
    intervals = []
    for entry in manifest.get("coverage_intervals", []):
        if not isinstance(entry, dict):
            raise ValueError("Each coverage interval must be an object")
        required = {"scope", "start", "end", "available_as_of"}
        if not required.issubset(entry):
            raise ValueError(f"Missing coverage interval fields: {sorted(required-set(entry))}")
        if entry["scope"] != "all_sources":
            raise ValueError("Only explicit all_sources coverage can certify complete purchase history")
        start = _timestamp(entry["start"], "coverage start")
        end = _timestamp(entry["end"], "coverage end")
        available = _timestamp(entry["available_as_of"], "coverage available_as_of")
        if end <= start:
            raise ValueError("Coverage end must be later than coverage start")
        end_inclusive = entry.get("end_inclusive", False)
        if not isinstance(end_inclusive, bool):
            raise ValueError("Coverage end_inclusive must be boolean")
        ids = entry.get("customer_ids")
        if ids is not None and (not isinstance(ids, list) or not ids or any(pd.isna(cid) for cid in ids)):
            raise ValueError("customer_ids must be a nonempty ID list or null for global coverage")
        intervals.append(dict(start=start, end=end, end_inclusive=end_inclusive,
                              available_as_of=available, customer_ids=None if ids is None else set(ids)))
    return intervals


def _window_is_covered(intervals, customer_id, start, end, end_inclusive):
    """Cover the entire window with the union of eligible explicit intervals.

    Starts are inclusive. End inclusivity matters when a contract ends exactly
    at a decision cutoff. Adjacent intervals may jointly certify one window.
    A later-issued certification cannot alter historical feature availability.
    """
    eligible = [interval for interval in intervals
                if interval["available_as_of"] <= end and
                (interval["customer_ids"] is None or customer_id in interval["customer_ids"])]
    cursor = start
    covered_endpoint = False
    started = False
    for interval in sorted(eligible, key=lambda row: row["start"]):
        left, right = interval["start"], interval["end"]
        if right < start or (right == start and not interval["end_inclusive"]):
            continue
        if left > cursor:
            break
        started = True
        if right > cursor:
            cursor, covered_endpoint = right, interval["end_inclusive"]
        elif right == cursor:
            covered_endpoint = covered_endpoint or interval["end_inclusive"]
        if cursor > end or (cursor == end and (not end_inclusive or covered_endpoint)):
            return True
    return bool(started and (cursor > end or (cursor == end and (not end_inclusive or covered_endpoint))))


def build_transaction_coverage(spine, orders, key="assignment_id", cut="send_ts", inclusive=False,
                               manifest=None):
    """Return one diagnostic row per decision, without changing model features.

    Requested windows begin at cutoff minus 30/90 days (inclusive). Historical
    windows exclude the decision instant; scoring can explicitly include it.
    ``observed_*`` counts refer only to supplied completed orders. The
    ``complete_orders_*`` fields are NaN unless eligible all-source coverage
    intervals cover the entire window. ``latest_observed_organic_order_ts`` is
    customer-specific; ``source_latest_observed_organic_order_ts`` describes the
    whole supplied feed as known by that decision. Neither certifies coverage.
    """
    if not isinstance(inclusive, (bool, np.bool_)):
        raise ValueError("inclusive must be boolean")
    decisions, transactions = _spine(spine, key, cut), _orders(orders)
    intervals = _intervals(manifest)
    result = decisions.set_index(key).copy()
    result["window_end_inclusive"] = bool(inclusive)
    joined = decisions.merge(transactions[["customer_id", "order_ts", "order_source", "is_completed"]],
                             on="customer_id", how="inner", validate="m:m")
    prior = joined.loc[(joined.order_ts <= joined[cut]) if inclusive else (joined.order_ts < joined[cut])].copy()
    completed = prior.loc[prior.is_completed.eq(True)]
    for days in WINDOWS:
        result[f"window_{days}d_start"] = result[cut] - pd.Timedelta(days=days)
        result[f"window_{days}d_end"] = result[cut]
        window = completed.loc[completed.order_ts >= completed[cut]-pd.Timedelta(days=days)]
        counts = window.groupby(key).size().reindex(result.index, fill_value=0).astype("int64")
        result[f"observed_orders_{days}d"] = counts
        for source, label in [(ORGANIC_SOURCE, "organic"), (CAMPAIGN_SOURCE, "campaign")]:
            result[f"observed_{label}_orders_{days}d"] = window.loc[window.order_source.eq(source)].groupby(
                key).size().reindex(result.index, fill_value=0).astype("int64")
        verified = np.array([_window_is_covered(intervals, customer_id,
                            cutoff-pd.Timedelta(days=days), cutoff, bool(inclusive))
                             for customer_id, cutoff in zip(decisions.customer_id, decisions[cut])], dtype=bool)
        result[f"orders_{days}d_coverage_status"] = np.where(verified, "verified_complete", "unverified")
        result[f"complete_orders_{days}d"] = np.where(verified, counts.to_numpy(float), np.nan)
    organic = prior.loc[prior.order_source.eq(ORGANIC_SOURCE)]
    result["latest_observed_organic_order_ts"] = organic.groupby(key).order_ts.max()
    all_organic = transactions.loc[transactions.order_source.eq(ORGANIC_SOURCE), "order_ts"]
    latest = {}
    for cutoff in decisions[cut].unique():
        timestamp = pd.Timestamp(cutoff)
        available = all_organic[all_organic <= timestamp] if inclusive else all_organic[all_organic < timestamp]
        latest[timestamp] = available.max()
    result["source_latest_observed_organic_order_ts"] = result[cut].map(latest)
    any_verified = result[[f"orders_{days}d_coverage_status" for days in WINDOWS]].eq("verified_complete").any(axis=1)
    result["coverage_interpretation"] = np.where(
        any_verified,
        "complete counts are certified only for windows marked verified_complete; consult each window's status",
        "observed records are not verified complete transaction history",
    )
    return result.reset_index()


def observed_organic_reference_window(spine, orders, customers=None, key="assignment_id", cut="send_ts",
                                      reference_end="2025-01-10", lookback_days=90):
    """Summarize a fixed historical organic window, explicitly observed-only.

    The default interval is [2024-10-12, 2025-01-10), shared by training and July.
    It is not a replacement for a decision-relative recent-90-day feature. A
    decision before the reference end receives unavailable/NaN values, never
    records that would be future observations. Optional customer signup dates
    describe time since signup within this reference interval, not feed quality.
    """
    if not isinstance(lookback_days, (int, np.integer)) or isinstance(lookback_days, bool) or lookback_days <= 0:
        raise ValueError("lookback_days must be a positive integer")
    end = _timestamp(reference_end, "reference_end")
    start = end - pd.Timedelta(days=int(lookback_days))
    decisions, transactions = _spine(spine, key, cut), _orders(orders)
    result = decisions.copy()
    available = result[cut].ge(end)
    result["reference_window_start"] = start
    result["reference_window_end"] = end
    result["reference_end_inclusive"] = False
    result["reference_lookback_days"] = int(lookback_days)
    result["reference_available"] = available
    result["reference_status"] = np.where(available, "observed_only", "not_yet_available")
    result["reference_age_days"] = (result[cut]-end).dt.total_seconds().div(86400).where(available)
    events = transactions.loc[transactions.is_completed.eq(True) & transactions.order_source.eq(ORGANIC_SOURCE) &
                              transactions.order_ts.ge(start) & transactions.order_ts.lt(end)].copy()
    count = events.groupby("customer_id").size()
    result["observed_organic_reference_orders"] = result.customer_id.map(count).fillna(0.).where(available)
    latest = events.groupby("customer_id").order_ts.max()
    result["latest_observed_organic_order_ts_in_reference"] = result.customer_id.map(latest).where(available)
    if "net_revenue" in events:
        if not np.isfinite(events.net_revenue.to_numpy(float)).all():
            raise ValueError("Observed reference net revenue must be finite")
        revenue = events.groupby("customer_id").net_revenue.sum()
        result["observed_organic_reference_net_revenue"] = result.customer_id.map(revenue).fillna(0.).where(available)
    else:
        result["observed_organic_reference_net_revenue"] = np.nan
    if customers is not None:
        if not {"customer_id", "signup_date"}.issubset(customers) or not customers.customer_id.is_unique:
            raise ValueError("Customer metadata requires unique customer_id and signup_date")
        signup = result.customer_id.map(customers.set_index("customer_id").signup_date)
        signup = pd.to_datetime(signup, errors="raise")
        result["reference_days_since_signup_within_window"] = (
            (end-signup).dt.total_seconds().div(86400).clip(lower=0, upper=int(lookback_days)).where(available))
    result["reference_interpretation"] = "observed historical baseline; neither recent history nor certified complete coverage"
    return result
