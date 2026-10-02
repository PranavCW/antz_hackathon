"""Parquet read/write for data/raw and data/processed, so stages never hard-code paths."""

import pandas as pd

from src import config


def _path(layer: str, name: str):
    return config.load()["paths"][layer] / f"{name}.parquet"


def read(layer: str, name: str) -> pd.DataFrame:
    return pd.read_parquet(_path(layer, name))


def write(layer: str, name: str, df: pd.DataFrame) -> None:
    df.to_parquet(_path(layer, name), index=False)


def output_path(filename: str):
    return config.load()["paths"]["outputs"] / filename
