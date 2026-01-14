"""Feature engineering for time series sensor data"""

import pandas as pd
import numpy as np
from typing import Tuple


def extract_time_series_features(
    df: pd.DataFrame,
    value_column: str = 'temperature',
    timestamp_column: str = 'timestamp',
    window_size: str = '60min',
    verbose: bool = True
) -> pd.DataFrame:
    """
    Extract statistical and temporal features from time series data.
    
    Args:
        df: Input DataFrame with time series data
        value_column: Name of the value column to aggregate
        timestamp_column: Name of the timestamp column
        window_size: Size of time windows for aggregation (e.g., '60min', '1H')
        verbose: Whether to print progress information
        
    Returns:
        DataFrame with extracted features
    """
    if verbose:
        print(f"[FEAT] Extracting features with window size: {window_size}")
    
    # Ensure timestamp is datetime and set as index
    df = df.copy()
    if timestamp_column in df.columns:
        df[timestamp_column] = pd.to_datetime(df[timestamp_column])
        df = df.set_index(timestamp_column)
    
    # Resample and compute statistical features
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
    
    # Derived features
    agg['temp_range'] = agg['temp_max'] - agg['temp_min']
    agg['temp_diff'] = agg['temp_last'] - agg['temp_first']
    
    # Skewness (may fail for small windows)
    try:
        agg['temp_skew'] = df[value_column].groupby(pd.Grouper(freq=window_size)).skew()
        agg['temp_skew'] = agg['temp_skew'].fillna(0)
    except Exception:
        agg['temp_skew'] = 0
    
    # Temporal features
    agg['hour'] = agg.index.hour
    agg['hour_sin'] = np.sin(2 * np.pi * agg['hour'] / 24)
    agg['hour_cos'] = np.cos(2 * np.pi * agg['hour'] / 24)
    agg['weekday'] = agg.index.weekday
    
    # Clean up
    agg = agg.dropna(subset=['cnt']).fillna(0)
    
    if verbose:
        print(f"[FEAT] Feature engineering complete. Shape: {agg.shape}")
    
    return agg


def create_labels(
    df: pd.DataFrame,
    value_column: str = 'temp_mean',
    method: str = 'quantile',
    n_classes: int = 3,
    verbose: bool = True
) -> pd.Series:
    """
    Create classification labels from continuous values.
    
    Args:
        df: DataFrame with feature values
        value_column: Column name to use for labeling
        method: Labeling method ('quantile' or 'fixed')
        n_classes: Number of classes to create
        verbose: Whether to print label information
        
    Returns:
        Series with class labels
    """
    values = df[value_column]
    
    if method == 'quantile':
        if n_classes == 3:
            q25 = values.quantile(0.25)
            q75 = values.quantile(0.75)
            
            labels = values.apply(lambda x: 0 if x < q25 else (2 if x > q75 else 1))
            
            if verbose:
                print(f"[LABEL] Created 3-class labels using quantiles:")
                print(f"  Class 0: Low (< {q25:.2f})")
                print(f"  Class 1: Medium ({q25:.2f} - {q75:.2f})")
                print(f"  Class 2: High (> {q75:.2f})")
        else:
            # General quantile-based labeling
            quantiles = np.linspace(0, 1, n_classes + 1)[1:-1]
            thresholds = [values.quantile(q) for q in quantiles]
            labels = pd.cut(values, bins=[-np.inf] + thresholds + [np.inf], labels=range(n_classes))
            labels = labels.astype(int)
    
    elif method == 'fixed':
        # Fixed thresholds (for temperature: Cold, Normal, Warm, Hot)
        if n_classes == 4:
            labels = values.apply(lambda x: 0 if x < 10 else (1 if x < 25 else (2 if x < 35 else 3)))
        else:
            raise ValueError("Fixed method currently supports only 4 classes")
    
    if verbose:
        label_counts = labels.value_counts().sort_index()
        print(f"[LABEL] Label distribution:")
        for label, count in label_counts.items():
            pct = (count / len(labels)) * 100
            print(f"  Class {label}: {count} instances ({pct:.2f}%)")
    
    return labels


def prepare_features_and_labels(
    df: pd.DataFrame,
    feature_columns: list = None,
    label_column: str = 'label'
) -> Tuple[pd.DataFrame, pd.Series]:
    """
    Prepare feature matrix and label vector from DataFrame.
    
    Args:
        df: DataFrame with features and labels
        feature_columns: List of feature column names. If None, uses all except label_column
        label_column: Name of the label column
        
    Returns:
        Tuple of (feature_matrix, labels)
    """
    if feature_columns is None:
        # Default: exclude label and intermediate columns
        exclude_cols = [label_column, 'cnt', 'temp_first', 'temp_last', 'hour']
        feature_columns = [col for col in df.columns if col not in exclude_cols]
    
    X = df[feature_columns].copy()
    y = df[label_column].copy()
    
    return X, y

