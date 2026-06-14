"""Robust CSV loading for user uploads."""

from __future__ import annotations

import io
from typing import BinaryIO, Union

import pandas as pd

REQUIRED = {"timestamp", "temperature"}
COLUMN_ALIASES = {
    "time": "timestamp", "date": "timestamp", "datetime": "timestamp",
    "temp": "temperature", "temperature_c": "temperature", "temp_c": "temperature",
}


def _normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = [str(c).strip().lower().lstrip("\ufeff") for c in df.columns]
    return df.rename(columns={k: v for k, v in COLUMN_ALIASES.items() if k in df.columns})


def _split_single_column(df: pd.DataFrame) -> pd.DataFrame:
    if len(df.columns) != 1:
        return df
    series = df.iloc[:, 0].astype(str)
    rows = series.str.split(",", expand=True)
    if rows.shape[1] >= 2:
        rows = rows.iloc[:, :2]
        rows.columns = ["timestamp", "temperature"]
        if str(rows.iloc[0, 0]).strip().lower() in ("timestamp", "time", "date"):
            rows = rows.iloc[1:]
        return rows.reset_index(drop=True)
    return df


def load_user_csv(source: Union[str, BinaryIO]) -> pd.DataFrame:
    raw = source.read() if hasattr(source, "read") else None
    if raw is not None and hasattr(source, "seek"):
        source.seek(0)

    attempts = []
    if raw is not None:
        for encoding in ("utf-8-sig", "utf-8", "latin-1"):
            for sep in (",", ";", "\t", None):
                try:
                    attempts.append(pd.read_csv(
                        io.BytesIO(raw), sep=sep,
                        engine="python" if sep is None else "c", encoding=encoding
                    ))
                except Exception:
                    continue

    best = None
    for df in attempts:
        df = _normalize_columns(_split_single_column(df))
        if REQUIRED.issubset(set(df.columns)):
            best = df
            break

    if best is None:
        raise ValueError(
            "CSV must have columns: timestamp, temperature. "
            "Save as comma-separated UTF-8 in Notepad."
        )

    best["timestamp"] = best["timestamp"].astype(str).str.strip()
    best["temperature"] = pd.to_numeric(best["temperature"], errors="coerce")
    best = best.dropna(subset=["timestamp", "temperature"])
    if len(best) == 0:
        raise ValueError("No valid data rows found.")
    return best[["timestamp", "temperature"]]
