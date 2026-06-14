"""Feature engineering for time series sensor data"""

import pandas as pd
import numpy as np
from typing import List, Tuple

LABEL_SOURCE_COL = "temp_mean"
LEAKY_FEATURE_COLS = {LABEL_SOURCE_COL, "temp_median"}
DEFAULT_LOW_THRESHOLD = 18.0
DEFAULT_HIGH_THRESHOLD = 32.0


def get_prediction_feature_columns(columns: List[str]) -> List[str]:
    exclude = LEAKY_FEATURE_COLS | {"label", "cnt", "temp_first", "temp_last", "hour"}
    return [col for col in columns if col not in exclude]


def extract_time_series_features(
    df: pd.DataFrame,
    value_column: str = 'temperature',
    timestamp_column: str = 'timestamp',
    window_size: str = '60min',
    verbose: bool = True
) -> pd.DataFrame:
    if verbose:
        print(f"[FEAT] Extracting features with window size: {window_size}")

    df = df.copy()
    if timestamp_column in df.columns:
        df[timestamp_column] = pd.to_datetime(df[timestamp_column])
        df = df.set_index(timestamp_column)

    resampled = df[value_column].resample(window_size)
    agg = pd.DataFrame({
        'cnt': resampled.count(),
        'temp_mean': resampled.mean(),
        'temp_std': resampled.std(),
        'temp_min': resampled.min(),
        'temp_max': resampled.max(),
        'temp_median': resampled.median(),
        'temp_first': resampled.first(),
        'temp_last': resampled.last()
    })

    agg['temp_range'] = agg['temp_max'] - agg['temp_min']
    agg['temp_diff'] = agg['temp_last'] - agg['temp_first']

    try:
        agg['temp_skew'] = df[value_column].groupby(pd.Grouper(freq=window_size)).skew()
        agg['temp_skew'] = agg['temp_skew'].fillna(0)
    except Exception:
        agg['temp_skew'] = 0

    agg['hour'] = agg.index.hour
    agg['hour_sin'] = np.sin(2 * np.pi * agg['hour'] / 24)
    agg['hour_cos'] = np.cos(2 * np.pi * agg['hour'] / 24)
    agg['weekday'] = agg.index.weekday
    agg = agg.dropna(subset=['cnt']).fillna(0)

    if verbose:
        print(f"[FEAT] Feature engineering complete. Shape: {agg.shape}")
    return agg


def create_labels(
    df: pd.DataFrame,
    value_column: str = 'temp_mean',
    method: str = 'fixed',
    n_classes: int = 3,
    low_threshold: float = DEFAULT_LOW_THRESHOLD,
    high_threshold: float = DEFAULT_HIGH_THRESHOLD,
    verbose: bool = True
) -> pd.Series:
    values = df[value_column]

    if method == 'fixed' and n_classes == 3:
        labels = values.apply(
            lambda x: 0 if x < low_threshold else (2 if x > high_threshold else 1)
        )
        if verbose:
            print("[LABEL] Created 3-class labels using fixed thresholds:")
            print(f"  Class 0: Low (< {low_threshold:.1f}C)")
            print(f"  Class 1: Normal ({low_threshold:.1f} - {high_threshold:.1f}C)")
            print(f"  Class 2: High (> {high_threshold:.1f}C)")
    elif method == 'quantile':
        if n_classes == 3:
            q25 = values.quantile(0.25)
            q75 = values.quantile(0.75)
            labels = values.apply(lambda x: 0 if x < q25 else (2 if x > q75 else 1))
            if verbose:
                print(f"[LABEL] Quantile labels: Low < {q25:.2f}, High > {q75:.2f}")
        else:
            quantiles = np.linspace(0, 1, n_classes + 1)[1:-1]
            thresholds = [values.quantile(q) for q in quantiles]
            labels = pd.cut(values, bins=[-np.inf] + thresholds + [np.inf], labels=range(n_classes))
            labels = labels.astype(int)
    elif method == 'fixed' and n_classes == 4:
        labels = values.apply(lambda x: 0 if x < 10 else (1 if x < 25 else (2 if x < 35 else 3)))
    else:
        raise ValueError(f"Unsupported labeling: method={method}, n_classes={n_classes}")

    if verbose:
        for label, count in labels.value_counts().sort_index().items():
            print(f"  Class {label}: {count} ({100*count/len(labels):.1f}%)")
    return labels


def prepare_features_and_labels(
    df: pd.DataFrame,
    feature_columns: list = None,
    label_column: str = 'label'
) -> Tuple[pd.DataFrame, pd.Series]:
    if feature_columns is None:
        feature_columns = get_prediction_feature_columns(list(df.columns))
    return df[feature_columns].copy(), df[label_column].copy()
