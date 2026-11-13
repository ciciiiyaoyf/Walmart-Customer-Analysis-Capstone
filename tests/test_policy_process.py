"""Process failure handling must leave no campaign worker running."""
import multiprocessing
import pandas as pd
import pytest

from scripts.experiment_policy import isolated_evaluate_campaign


def test_worker_python_failure_reaches_parent_with_traceback():
    before = {p.pid for p in multiprocessing.active_children()}
    with pytest.raises(RuntimeError, match="failed in its worker") as error:
        isolated_evaluate_campaign(pd.DataFrame(), "CP07", [], [], 2, timeout=30.)
    assert "Traceback" in str(error.value)
    assert {p.pid for p in multiprocessing.active_children()} == before


def test_timeout_terminates_only_its_worker():
    before = {p.pid for p in multiprocessing.active_children()}
    with pytest.raises(TimeoutError, match="CP07 exceeded"):
        isolated_evaluate_campaign(pd.DataFrame(), "CP07", [], [], 2, timeout=.001)
    assert {p.pid for p in multiprocessing.active_children()} == before
