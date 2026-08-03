from __future__ import annotations

import pandas as pd


WINDOW_FUNCTIONS = [
    "ROW_NUMBER",
    "RANK",
    "DENSE_RANK",
    "LAG",
    "LEAD",
    "CUMSUM",
    "ROLLING MEAN",
]


def apply_window(
    dataframe: pd.DataFrame,
    table_name: str,
    function: str,
    value_column: str,
    partition_columns: list[str],
    order_column: str,
    ascending: bool,
    offset: int = 1,
    window_size: int = 3,
    output_column: str = "window_value",
) -> dict:
    result = dataframe.copy()

    sort_columns = partition_columns + [order_column]
    sort_ascending = [True] * len(partition_columns) + [ascending]
    result = result.sort_values(
        sort_columns,
        ascending=sort_ascending,
    ).reset_index(drop=True)

    partition_sql = (
        f"PARTITION BY {', '.join(partition_columns)} "
        if partition_columns
        else ""
    )
    order_sql = (
        f"ORDER BY {order_column} "
        f"{'ASC' if ascending else 'DESC'}"
    )

    if function == "ROW_NUMBER":
        if partition_columns:
            result[output_column] = (
                result.groupby(partition_columns, dropna=False)
                .cumcount()
                .add(1)
            )
        else:
            result[output_column] = range(1, len(result) + 1)

        sql_expr = f"ROW_NUMBER() OVER ({partition_sql}{order_sql})"
        pandas_expr = (
            f"result[{output_column!r}] = "
            + (
                f"result.groupby({partition_columns!r}, dropna=False)"
                f".cumcount().add(1)"
                if partition_columns
                else f"range(1, len(result) + 1)"
            )
        )

    elif function in {"RANK", "DENSE_RANK"}:
        method = "min" if function == "RANK" else "dense"
        if partition_columns:
            result[output_column] = (
                result.groupby(partition_columns, dropna=False)[value_column]
                .rank(method=method, ascending=ascending)
            )
            pandas_expr = (
                f"result[{output_column!r}] = (\n"
                f"    result.groupby({partition_columns!r}, dropna=False)"
                f"[{value_column!r}]\n"
                f"    .rank(method={method!r}, ascending={ascending})\n"
                f")"
            )
        else:
            result[output_column] = result[value_column].rank(
                method=method,
                ascending=ascending,
            )
            pandas_expr = (
                f"result[{output_column!r}] = "
                f"result[{value_column!r}].rank("
                f"method={method!r}, ascending={ascending})"
            )

        sql_expr = f"{function}() OVER ({partition_sql}{order_sql})"

    elif function in {"LAG", "LEAD"}:
        periods = offset if function == "LAG" else -offset
        if partition_columns:
            result[output_column] = (
                result.groupby(partition_columns, dropna=False)[value_column]
                .shift(periods)
            )
            pandas_expr = (
                f"result[{output_column!r}] = (\n"
                f"    result.groupby({partition_columns!r}, dropna=False)"
                f"[{value_column!r}]\n"
                f"    .shift({periods})\n"
                f")"
            )
        else:
            result[output_column] = result[value_column].shift(periods)
            pandas_expr = (
                f"result[{output_column!r}] = "
                f"result[{value_column!r}].shift({periods})"
            )

        sql_expr = (
            f"{function}({value_column}, {offset}) "
            f"OVER ({partition_sql}{order_sql})"
        )

    elif function == "CUMSUM":
        if partition_columns:
            result[output_column] = (
                result.groupby(partition_columns, dropna=False)[value_column]
                .cumsum()
            )
            pandas_expr = (
                f"result[{output_column!r}] = (\n"
                f"    result.groupby({partition_columns!r}, dropna=False)"
                f"[{value_column!r}]\n"
                f"    .cumsum()\n"
                f")"
            )
        else:
            result[output_column] = result[value_column].cumsum()
            pandas_expr = (
                f"result[{output_column!r}] = "
                f"result[{value_column!r}].cumsum()"
            )

        sql_expr = (
            f"SUM({value_column}) OVER ("
            f"{partition_sql}{order_sql} "
            f"ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW)"
        )

    else:
        if partition_columns:
            result[output_column] = (
                result.groupby(partition_columns, dropna=False)[value_column]
                .transform(lambda s: s.rolling(window_size, min_periods=1).mean())
            )
            pandas_expr = (
                f"result[{output_column!r}] = (\n"
                f"    result.groupby({partition_columns!r}, dropna=False)"
                f"[{value_column!r}]\n"
                f"    .transform(lambda s: "
                f"s.rolling({window_size}, min_periods=1).mean())\n"
                f")"
            )
        else:
            result[output_column] = (
                result[value_column]
                .rolling(window_size, min_periods=1)
                .mean()
            )
            pandas_expr = (
                f"result[{output_column!r}] = (\n"
                f"    result[{value_column!r}]\n"
                f"    .rolling({window_size}, min_periods=1)\n"
                f"    .mean()\n"
                f")"
            )

        sql_expr = (
            f"AVG({value_column}) OVER ("
            f"{partition_sql}{order_sql} "
            f"ROWS BETWEEN {window_size - 1} PRECEDING AND CURRENT ROW)"
        )

    sql = (
        f"SELECT *,\n"
        f"       {sql_expr} AS {output_column}\n"
        f"FROM {table_name};"
    )

    pandas_code = (
        f"result = {table_name}.sort_values(\n"
        f"    {sort_columns!r},\n"
        f"    ascending={sort_ascending!r},\n"
        f").reset_index(drop=True)\n"
        f"{pandas_expr}"
    )

    return {
        "sql": sql,
        "pandas": pandas_code,
        "result": result,
        "explanation": (
            "Window functions preserve row-level detail while calculating values "
            "across an ordered set of related rows."
        ),
    }
