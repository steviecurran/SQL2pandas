from __future__ import annotations

import pandas as pd


def union_tables(
    first: pd.DataFrame,
    second: pd.DataFrame,
    first_name: str,
    second_name: str,
    union_all: bool,
) -> dict:
    if list(first.columns) != list(second.columns):
        raise ValueError(
            "UNION requires the two tables to have the same columns in the same order."
        )

    result = pd.concat([first, second], ignore_index=True)
    if not union_all:
        result = result.drop_duplicates().reset_index(drop=True)

    keyword = "UNION ALL" if union_all else "UNION"
    drop_code = "" if union_all else "\n    .drop_duplicates()"

    return {
        "sql": (
            f"SELECT * FROM {first_name}\n"
            f"{keyword}\n"
            f"SELECT * FROM {second_name};"
        ),
        "pandas": (
            f"result = (\n"
            f"    pd.concat([{first_name}, {second_name}], ignore_index=True)"
            f"{drop_code}\n"
            f"    .reset_index(drop=True)\n"
            f")"
        ),
        "result": result,
        "explanation": (
            "pd.concat() corresponds to UNION ALL. "
            "Removing duplicates produces SQL UNION behaviour."
        ),
    }
