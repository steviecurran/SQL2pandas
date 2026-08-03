from __future__ import annotations

from pathlib import Path
from typing import BinaryIO, TextIO

import pandas as pd


SUPPORTED_EXTENSIONS = {".csv", ".txt", ".dat", ".tsv"}


def read_delimited(source) -> pd.DataFrame:
    """Read a delimited text source using automatic separator detection."""
    return pd.read_csv(
        source,
        sep=None,
        engine="python",
        comment="#",
    )


def load_builtin_tables(data_dir: Path) -> dict[str, pd.DataFrame]:
    """Load all supported repository datasets."""
    tables: dict[str, pd.DataFrame] = {}

    if not data_dir.exists():
        return tables

    for path in sorted(data_dir.iterdir()):
        if path.is_file() and path.suffix.lower() in SUPPORTED_EXTENSIONS:
            tables[path.stem] = read_delimited(path)

    return tables


def normalise_table_name(name: str) -> str:
    """Create a SQL-friendly identifier."""
    cleaned = "".join(
        char if char.isalnum() or char == "_" else "_"
        for char in name.strip()
    )
    cleaned = cleaned.strip("_") or "data"
    if cleaned[0].isdigit():
        cleaned = f"table_{cleaned}"
    return cleaned.lower()
