"""Exercise Step 01's actual validation cells with valid and corrupted inputs.

The committed clean snapshot provides portable fixtures, so these tests need no
private raw ZIP. Reconstructed item duplicates are intentionally synthetic.
Only the loader and output paths are substituted; production check expressions
and the notebook's pre-save FAIL gate execute unchanged. Clean-table writes are
not executed. The separate recorded raw run validates the supplied CSV package.
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src import common

ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK = ROOT / "notebooks/01_data_validation.ipynb"


@pytest.fixture(scope="module")
def source_fixture():
    tables = {name: pd.read_parquet(ROOT / "data/processed/clean" / f"{name}.parquet")
              for name in common.TABLE_NAMES}
    tables["orders"] = tables["orders"].drop(columns="is_completed")
    tables["web_sessions"] = tables["web_sessions"].drop(columns="is_identified")
    tables["order_items"] = pd.concat(
        [tables["order_items"], tables["order_items"].iloc[:160]], ignore_index=True)
    return tables


@pytest.fixture
def tables(source_fixture):
    return {name: frame.copy(deep=True) for name, frame in source_fixture.items()}


def validate(tables, monkeypatch, tmp_path):
    raw = tmp_path / "raw"
    outputs = tmp_path / "tables"
    raw.mkdir()
    outputs.mkdir()
    monkeypatch.setattr(common, "RAW", raw)
    monkeypatch.setattr(common, "TABLES", outputs)
    monkeypatch.setattr(common, "PROC", tmp_path / "processed")
    monkeypatch.setattr(common, "load_all_raw", lambda: tables)
    notebook = json.loads(NOTEBOOK.read_text(encoding="utf-8"))
    state = {"__name__": "__data_quality_test__"}
    gate_ran = False
    with pd.option_context("display.width", 200, "display.max_columns", 20):
        for cell in notebook["cells"]:
            if cell["cell_type"] != "code":
                continue
            source = "".join(cell["source"])
            if "clean = dict(t)" in source:
                break
            try:
                exec(compile(source, str(NOTEBOOK), "exec"), state)
            except AssertionError as exc:
                if "dqs" in state:
                    failures = state["dqs"].query("status == 'FAIL'")
                    raise AssertionError(failures.to_string(index=False)) from exc
                raise
            gate_ran |= 'assert (dqs.status != "FAIL").all()' in source
    assert gate_ran, "The production notebook must retain its pre-save FAIL gate"
    assert not (tmp_path / "processed").exists()
    return state


def assert_rejected(tables, expected, monkeypatch, tmp_path):
    with pytest.raises(AssertionError) as caught:
        validate(tables, monkeypatch, tmp_path)
    assert expected in str(caught.value)


def test_valid_input_retains_repeat_customers_and_deduplicates_exact_items(tables, monkeypatch, tmp_path):
    before = tables["order_items"].copy(deep=True)
    state = validate(tables, monkeypatch, tmp_path)
    assert state["dqs"].status.ne("FAIL").all()
    assert len(before) - len(state["oi"]) == 160
    assert not state["oi"].duplicated(["order_id", "line_number"]).any()
    pd.testing.assert_frame_equal(before, tables["order_items"])
    # A person may occur in multiple campaigns; the customer-campaign pair is unique.
    assignments = state["ha"]
    assert assignments.customer_id.duplicated().any()
    assert not assignments.duplicated(["customer_id", "campaign_id"]).any()


@pytest.mark.parametrize("table,key,expected", [
    ("orders", "order_id", "pk_unique:order_id"),
    ("products", "product_id", "pk_unique:product_id"),
    ("returns", "return_id", "pk_unique:return_id"),
])
def test_duplicate_primary_keys_are_rejected(tables, monkeypatch, tmp_path, table, key, expected):
    tables[table].loc[1, key] = tables[table].loc[0, key]
    assert_rejected(tables, expected, monkeypatch, tmp_path)


def test_duplicate_customer_campaign_assignments_are_rejected(tables, monkeypatch, tmp_path):
    assignments = tables["historical_assignments"]
    assignments.loc[1, ["customer_id", "campaign_id"]] = assignments.loc[0, ["customer_id", "campaign_id"]]
    assert_rejected(tables, "unique_customer_campaign_assignment", monkeypatch, tmp_path)


def test_conflicting_item_rows_are_rejected(tables, monkeypatch, tmp_path):
    # One of the exact duplicates now disagrees; dropping identical rows cannot resolve it.
    tables["order_items"].loc[len(tables["order_items"]) - 1, "quantity"] += 1
    assert_rejected(tables, "pk_unique:(order_id,line_number) after dedup", monkeypatch, tmp_path)


@pytest.mark.parametrize("table,column,reference,expected", [
    ("order_items", "product_id", "products", "fk:order_item.product_id"),
    ("returns", "order_id", "orders", "fk:return.order_id"),
    ("orders", "customer_id", "customers", "fk:order.customer_id"),
])
def test_broken_relationships_are_rejected(tables, monkeypatch, tmp_path, table, column, reference, expected):
    identifiers = tables[reference][column]
    unknown = int(identifiers.max()) + 1 if pd.api.types.is_numeric_dtype(identifiers) else "unknown_identifier"
    if table == "order_items":
        items = tables[table]
        same_line = items.order_id.eq(items.loc[0, "order_id"]) & items.line_number.eq(items.loc[0, "line_number"])
        items.loc[same_line, column] = unknown
    else:
        tables[table].loc[0, column] = unknown
    assert_rejected(tables, expected, monkeypatch, tmp_path)


def test_email_customer_must_match_its_assignment(tables, monkeypatch, tmp_path):
    email = tables["email_events"]
    other = tables["customers"].loc[tables["customers"].customer_id.ne(email.loc[0, "customer_id"]), "customer_id"].iloc[0]
    email.loc[0, "customer_id"] = other
    assert_rejected(tables, "fk:email.customer/campaign match assignment", monkeypatch, tmp_path)


@pytest.mark.parametrize("kind,expected", [
    ("net_revenue", "net = gross - discount + shipping_rev"),
    ("cancelled", "cancelled orders carry zero financials"),
    ("item_cost", "item cogs reconciles to order cogs (after dedup)"),
])
def test_financial_corruption_is_rejected(tables, monkeypatch, tmp_path, kind, expected):
    orders = tables["orders"]
    if kind == "item_cost":
        completed = orders.loc[orders.order_status.eq("completed"), "order_id"]
        items = tables["order_items"]
        idx = items.index[items.order_id.isin(completed)][0]
        same_line = items.order_id.eq(items.loc[idx, "order_id"]) & items.line_number.eq(items.loc[idx, "line_number"])
        items.loc[same_line, "unit_cost"] += 10
    else:
        status = "cancelled" if kind == "cancelled" else "completed"
        idx = orders.index[orders.order_status.eq(status)][0]
        orders.loc[idx, "net_revenue"] += 10
    assert_rejected(tables, expected, monkeypatch, tmp_path)


@pytest.mark.parametrize("table,column,expected", [
    ("orders", "net_revenue", "finite order financials"),
    ("order_items", "unit_cost", "finite item quantities/prices/costs"),
    ("returns", "refund_amount", "finite refund amounts"),
])
def test_nonfinite_amounts_are_rejected(tables, monkeypatch, tmp_path, table, column, expected):
    if table == "order_items":
        items = tables[table]
        same_line = items.order_id.eq(items.loc[0, "order_id"]) & items.line_number.eq(items.loc[0, "line_number"])
        items.loc[same_line, column] = np.inf
    else:
        tables[table].loc[0, column] = np.inf
    assert_rejected(tables, expected, monkeypatch, tmp_path)


def test_cumulative_refunds_cannot_exceed_order_revenue(tables, monkeypatch, tmp_path):
    orders = tables["orders"].set_index("order_id")
    returns = tables["returns"]
    order_id = returns.loc[0, "order_id"]
    revenue = orders.loc[order_id, "net_revenue"]
    # Two separately valid partial refunds; their combined amount is invalid.
    returns.loc[0, "refund_amount"] = .75 * revenue
    returns.loc[1, ["order_id", "return_ts", "refund_amount"]] = [
        order_id, returns.loc[0, "return_ts"] + pd.Timedelta(hours=1), .75 * revenue]
    assert returns.loc[[0, 1], "refund_amount"].le(revenue).all()
    assert_rejected(tables, "total refunds per order <= net_revenue", monkeypatch, tmp_path)


def test_return_before_order_is_rejected(tables, monkeypatch, tmp_path):
    returns = tables["returns"]
    order_ts = tables["orders"].set_index("order_id").loc[returns.loc[0, "order_id"], "order_ts"]
    returns.loc[0, "return_ts"] = order_ts - pd.Timedelta(seconds=1)
    assert_rejected(tables, "return_ts after order_ts", monkeypatch, tmp_path)
