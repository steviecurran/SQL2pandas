from __future__ import annotations

import pandas as pd


JOIN_MAP = {
    "INNER": "inner",
    "LEFT": "left",
    "RIGHT": "right",
    "FULL OUTER": "outer",
}


def join_tables(
    left: pd.DataFrame,
    right: pd.DataFrame,
    left_name: str,
    right_name: str,
    join_type: str,
    left_keys: list[str],
    right_keys: list[str],
    suffixes: tuple[str, str] = ("_left", "_right"),
) -> dict:
    if join_type == "CROSS":
        result = left.merge(
            right,
            how="cross",
            suffixes=suffixes,
        )
        sql = (
            f"SELECT *\nFROM {left_name}\n"
            f"CROSS JOIN {right_name};"
        )
        pandas_code = (
            f"result = {left_name}.merge(\n"
            f"    {right_name},\n"
            f"    how='cross',\n"
            f"    suffixes={suffixes!r},\n"
            f")"
        )
        diagnostics = {
            "left_rows": len(left),
            "right_rows": len(right),
            "output_rows": len(result),
            "matched_rows": len(result),
            "left_only_rows": 0,
            "right_only_rows": 0,
        }
        return {
            "sql": sql,
            "pandas": pandas_code,
            "result": result,
            "diagnostics": diagnostics,
            "explanation": "A cross join returns every possible pair of rows.",
        }

    if not left_keys or not right_keys:
        raise ValueError("Select at least one join key for each table.")
    if len(left_keys) != len(right_keys):
        raise ValueError("The same number of left and right join keys is required.")

    how = JOIN_MAP[join_type]

    result_with_indicator = left.merge(
        right,
        how=how,
        left_on=left_keys,
        right_on=right_keys,
        suffixes=suffixes,
        indicator=True,
    )

    counts = result_with_indicator["_merge"].value_counts()
    diagnostics = {
        "left_rows": len(left),
        "right_rows": len(right),
        "output_rows": len(result_with_indicator),
        "matched_rows": int(counts.get("both", 0)),
        "left_only_rows": int(counts.get("left_only", 0)),
        "right_only_rows": int(counts.get("right_only", 0)),
    }

    result = result_with_indicator.drop(columns="_merge")

    conditions = [
        f"{left_name}.{lk} = {right_name}.{rk}"
        for lk, rk in zip(left_keys, right_keys)
    ]
    sql = (
        f"SELECT *\nFROM {left_name}\n"
        f"{join_type} JOIN {right_name}\n"
        f"ON {' AND '.join(conditions)};"
    )

    same_keys = left_keys == right_keys
    if same_keys:
        key_code = f"on={left_keys!r}"
    else:
        key_code = (
            f"left_on={left_keys!r},\n"
            f"    right_on={right_keys!r}"
        )

    pandas_code = (
        f"result = {left_name}.merge(\n"
        f"    {right_name},\n"
        f"    how={how!r},\n"
        f"    {key_code},\n"
        f"    suffixes={suffixes!r},\n"
        f")"
    )

    return {
        "sql": sql,
        "pandas": pandas_code,
        "result": result,
        "diagnostics": diagnostics,
        "explanation": (
            "pandas.merge() implements SQL-style joins. "
            "The merge indicator is used internally to diagnose matched and unmatched rows."
        ),
    }


def join_key_compatibility(
    left: pd.DataFrame,
    right: pd.DataFrame,
) -> pd.DataFrame:
    rows = []
    for left_col in left.columns:
        for right_col in right.columns:
            left_dtype = str(left[left_col].dtype)
            right_dtype = str(right[right_col].dtype)
            compatible = (
                pd.api.types.is_numeric_dtype(left[left_col])
                and pd.api.types.is_numeric_dtype(right[right_col])
            ) or (
                pd.api.types.is_string_dtype(left[left_col])
                and pd.api.types.is_string_dtype(right[right_col])
            ) or left_dtype == right_dtype

            if left_col == right_col or compatible:
                rows.append({
                    "left_column": left_col,
                    "left_dtype": left_dtype,
                    "right_column": right_col,
                    "right_dtype": right_dtype,
                    "same_name": left_col == right_col,
                })
    return pd.DataFrame(rows)
