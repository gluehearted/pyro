from __future__ import annotations

from dataclasses import dataclass
from datetime import timedelta
from typing import Dict, Iterable, Tuple

import pandas as pd


@dataclass(frozen=True)
class SplitRange:
    name: str
    start: pd.Timestamp
    end: pd.Timestamp


def _to_timestamp(value) -> pd.Timestamp:
    return pd.Timestamp(value).normalize()


def get_split_ranges(cfg: dict | None = None) -> Dict[str, SplitRange]:
    """
    Chronological split untuk mencegah data leakage.

    Default:
    Train : 2015-01-01 s.d. 2021-12-31
    Val   : 2022-01-01 s.d. 2023-12-31
    Test  : 2024-01-01 s.d. 2024-12-24 / sesuai config
    """

    cfg = cfg or {}
    split_cfg = cfg.get("split", {})

    train_start = _to_timestamp(split_cfg.get("train_start", "2015-01-01"))
    train_end = _to_timestamp(split_cfg.get("train_end", "2021-12-31"))

    val_start = _to_timestamp(
        split_cfg.get("val_start", train_end + timedelta(days=1))
    )
    val_end = _to_timestamp(split_cfg.get("val_end", "2023-12-31"))

    test_start = _to_timestamp(
        split_cfg.get("test_start", val_end + timedelta(days=1))
    )
    test_end = _to_timestamp(split_cfg.get("test_end", "2024-12-24"))

    return {
        "train": SplitRange("train", train_start, train_end),
        "val": SplitRange("val", val_start, val_end),
        "test": SplitRange("test", test_start, test_end),
    }


def split_xarray_by_time(ds, cfg: dict | None = None):
    ranges = get_split_ranges(cfg)

    return {
        name: ds.sel(time=slice(split.start, split.end))
        for name, split in ranges.items()
    }


def split_dataframe_by_time(df, cfg: dict | None = None, time_col: str = "time"):
    ranges = get_split_ranges(cfg)

    df = df.copy()
    df[time_col] = pd.to_datetime(df[time_col])

    out = {}

    for name, split in ranges.items():
        mask = (
            (df[time_col] >= split.start)
            & (df[time_col] <= split.end)
        )
        out[name] = df.loc[mask].copy()

    return out


def iter_time_chunks(
    start: pd.Timestamp,
    end: pd.Timestamp,
    chunk_days: int = 30,
) -> Iterable[Tuple[pd.Timestamp, pd.Timestamp]]:
    """
    Membagi rentang waktu menjadi chunk kecil agar xarray -> pandas
    tidak meledakkan RAM.
    """

    current = pd.Timestamp(start).normalize()
    end = pd.Timestamp(end).normalize()

    while current <= end:
        chunk_end = min(
            current + timedelta(days=chunk_days - 1),
            end
        )

        yield current, chunk_end

        current = chunk_end + timedelta(days=1)