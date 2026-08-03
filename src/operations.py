from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd


def sql_literal(value: Any) -> str:
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return "NULL"
    if isinstance(value, (int, float, np.integer, np.floating)):
        return str(value)
    if isinstance(value, (pd.Timestamp, np.datetime64)):
        return f"'{pd.Timestamp(value).strftime('%Y-%m-%d')}'"
    escaped = str(value).replace("'", "''")
    return f"'{escaped}'"


def coerce_value(series: pd.Series, value: Any) -> Any:
    if pd.api.types.is_numeric_dtype(series):
        return float(value)
    if pd.api.types.is_datetime64_any_dtype(series):
        return pd.Timestamp(value)
    return str(value)


def comparison_mask(
    dataframe: pd.DataFrame,
    column: str,
    operator: str,
    value: Any,
) -> pd.Series:
    series = dataframe[column]
    converted = coerce_value(series, value)

    operators = {
        "=": series == converted,
        "!=": series != converted,
        ">": series > converted,
        ">=": series >= converted,
        "<": series < converted,
        "<=": series <= converted,
    }
    return operators[operator]


def select_columns(
    dataframe: pd.DataFrame,
    table_name: str,
    columns: list[str],
) -> dict:
    if not columns:
        raise ValueError("Select at least one column.")

    return {
        "sql": f"SELECT {', '.join(columns)}\nFROM {table_name};",
        "pandas": f"result = {table_name}[{columns!r}].copy()",
        "result": dataframe[columns].copy(),
        "explanation": "Both forms project the table onto the selected columns.",
    }


def where_filter(
    dataframe: pd.DataFrame,
    table_name: str,
    column: str,
    operator: str,
    value: Any,
) -> dict:
    converted = coerce_value(dataframe[column], value)
    py_operator = "==" if operator == "=" else operator
    mask = comparison_mask(dataframe, column, operator, value)

    return {
        "sql": (
            f"SELECT *\nFROM {table_name}\n"
            f"WHERE {column} {operator} {sql_literal(converted)};"
        ),
        "pandas": (
            f"result = {table_name}[\n"
            f"    {table_name}[{column!r}] {py_operator} {converted!r}\n"
            f"].copy()"
        ),
        "result": dataframe[mask].copy(),
        "explanation": "A SQL WHERE predicate maps to a pandas Boolean mask.",
    }


def membership_filter(
    dataframe: pd.DataFrame,
    table_name: str,
    column: str,
    values: list[Any],
    negate: bool = False,
) -> dict:
    if not values:
        raise ValueError("Select at least one value.")

    original_values = (
        dataframe[column]
        .dropna()
        .loc[lambda s: s.astype(str).isin([str(v) for v in values])]
        .unique()
        .tolist()
    )
    mask = dataframe[column].isin(original_values)
    if negate:
        mask = ~mask

    keyword = "NOT IN" if negate else "IN"
    tilde = "~" if negate else ""

    return {
        "sql": (
            f"SELECT *\nFROM {table_name}\n"
            f"WHERE {column} {keyword} "
            f"({', '.join(sql_literal(v) for v in original_values)});"
        ),
        "pandas": (
            f"result = {table_name}[\n"
            f"    {tilde}{table_name}[{column!r}].isin({original_values!r})\n"
            f"].copy()"
        ),
        "result": dataframe[mask].copy(),
        "explanation": "SQL membership tests map to Series.isin(); NOT IN adds negation.",
    }


def group_aggregate(
    dataframe: pd.DataFrame,
    table_name: str,
    group_columns: list[str],
    value_column: str,
    function: str,
    having_operator: str | None = None,
    having_value: float | None = None,
) -> dict:
    if not group_columns:
        raise ValueError("Select at least one grouping column.")

    pandas_function = {
        "SUM": "sum",
        "AVG": "mean",
        "MIN": "min",
        "MAX": "max",
        "COUNT": "count",
        "MEDIAN": "median",
    }[function]

    result_name = f"{function.lower()}_{value_column}"
    result = (
        dataframe.groupby(group_columns, dropna=False)[value_column]
        .agg(pandas_function)
        .reset_index(name=result_name)
    )

    having_sql = ""
    having_pandas = ""
    if having_operator is not None and having_value is not None:
        result = result[
            comparison_mask(
                result,
                result_name,
                having_operator,
                having_value,
            )
        ].copy()
        having_sql = (
            f"\nHAVING {function}({value_column}) "
            f"{having_operator} {having_value}"
        )
        py_operator = "==" if having_operator == "=" else having_operator
        having_pandas = (
            f"\n    .loc[lambda x: "
            f"x[{result_name!r}] {py_operator} {having_value}]"
        )

    result = result.sort_values(result_name, ascending=False).reset_index(drop=True)

    return {
        "sql": (
            f"SELECT {', '.join(group_columns)}, "
            f"{function}({value_column}) AS {result_name}\n"
            f"FROM {table_name}\n"
            f"GROUP BY {', '.join(group_columns)}"
            f"{having_sql}\n"
            f"ORDER BY {result_name} DESC;"
        ),
        "pandas": (
            f"result = (\n"
            f"    {table_name}\n"
            f"    .groupby({group_columns!r}, dropna=False)"
            f"[{value_column!r}]\n"
            f"    .{pandas_function}()\n"
            f"    .reset_index(name={result_name!r})"
            f"{having_pandas}\n"
            f"    .sort_values({result_name!r}, ascending=False)\n"
            f"    .reset_index(drop=True)\n"
            f")"
        ),
        "result": result,
        "explanation": "groupby() creates groups, applies an aggregation, and HAVING filters the grouped result.",
    }


def case_when(
    dataframe: pd.DataFrame,
    table_name: str,
    column: str,
    operator: str,
    threshold: Any,
    true_label: str,
    false_label: str,
    output_column: str,
) -> dict:
    mask = comparison_mask(dataframe, column, operator, threshold)
    result = dataframe.copy()
    result[output_column] = np.where(mask, true_label, false_label)
    converted = coerce_value(dataframe[column], threshold)

    return {
        "sql": (
            f"SELECT *,\n"
            f"       CASE\n"
            f"           WHEN {column} {operator} {sql_literal(converted)} "
            f"THEN {sql_literal(true_label)}\n"
            f"           ELSE {sql_literal(false_label)}\n"
            f"       END AS {output_column}\n"
            f"FROM {table_name};"
        ),
        "pandas": (
            f"result = {table_name}.copy()\n"
            f"result[{output_column!r}] = np.where(\n"
            f"    result[{column!r}] "
            f"{'==' if operator == '=' else operator} {converted!r},\n"
            f"    {true_label!r},\n"
            f"    {false_label!r},\n"
            f")"
        ),
        "result": result,
        "explanation": "A two-branch SQL CASE expression maps naturally to numpy.where().",
    }
