import pandas as pd

from src.joins import join_tables
from src.operations import case_when, group_aggregate
from src.unions import union_tables
from src.windows import apply_window


def test_inner_join():
    left = pd.DataFrame({"id": [1, 2], "x": [10, 20]})
    right = pd.DataFrame({"id": [2, 3], "y": [30, 40]})
    result = join_tables(
        left, right, "left_table", "right_table",
        "INNER", ["id"], ["id"]
    )
    assert len(result["result"]) == 1


def test_group_aggregate():
    data = pd.DataFrame({"group": ["a", "a", "b"], "value": [1, 2, 3]})
    result = group_aggregate(
        data, "data", ["group"], "value", "SUM"
    )
    assert result["result"].loc[0, "sum_value"] == 3


def test_case_when():
    data = pd.DataFrame({"value": [1, 10]})
    result = case_when(
        data, "data", "value", ">", 5,
        "High", "Low", "band"
    )
    assert result["result"]["band"].tolist() == ["Low", "High"]


def test_union():
    a = pd.DataFrame({"id": [1, 2]})
    b = pd.DataFrame({"id": [2, 3]})
    result = union_tables(a, b, "a", "b", union_all=False)
    assert result["result"]["id"].tolist() == [1, 2, 3]


def test_rank():
    data = pd.DataFrame({"group": ["a", "a"], "value": [2, 1]})
    result = apply_window(
        data, "data", "DENSE_RANK", "value",
        ["group"], "value", False
    )
    assert result["result"]["window_value"].tolist() == [1.0, 2.0]
