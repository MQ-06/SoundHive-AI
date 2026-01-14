"""Time series data preprocessing for beehive sensor data"""

import pandas as pd
import numpy as np
from datetime import datetime
from typing import Optional
import os


def parse_timestamps(df: pd.DataFrame, timestamp_column: str = 'timestamp') -> pd.DataFrame:
    """
    Parse timestamps from string to datetime format.
    
    Args:
        df: Input DataFrame
        timestamp_column: Name of the timestamp column
        
    Returns:
        DataFrame with parsed timestamps
    """
    df = df.copy()
    df[timestamp_column] = pd.to_datetime(df[timestamp_column])
    return df


def sort_chronologically(df: pd.DataFrame, timestamp_column: str = 'timestamp') -> pd.DataFrame:
    """
    Sort data in chronological order.
    
    Args:
        df: Input DataFrame
        timestamp_column: Name of the timestamp column
        
    Returns:
        Chronologically sorted DataFrame
    """
    df = df.copy()
    if not df[timestamp_column].is_monotonic_increasing:
        df = df.sort_values(by=timestamp_column).reset_index(drop=True)
    return df


def handle_missing_values(df: pd.DataFrame, method: str = 'interpolate') -> pd.DataFrame:
    """
    Handle missing values in the dataset.
    
    Args:
        df: Input DataFrame
        method: Method to handle missing values ('interpolate', 'ffill', 'bfill', 'drop')
        
    Returns:
        DataFrame with missing values handled
    """
    df = df.copy()
    
    if df.isnull().sum().sum() == 0:
        return df
    
    if method == 'interpolate':
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        df[numeric_cols] = df[numeric_cols].interpolate(method='linear')
    elif method == 'ffill':
        df = df.fillna(method='ffill')
    elif method == 'bfill':
        df = df.fillna(method='bfill')
    elif method == 'drop':
        df = df.dropna()
    
    return df


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Perform basic data cleaning operations.
    
    Args:
        df: Input DataFrame
        
    Returns:
        Cleaned DataFrame
    """
    df = df.copy()
    
    # Remove duplicates
    duplicates = df.duplicated().sum()
    if duplicates > 0:
        df = df.drop_duplicates()
    
    return df


def preprocess_timeseries(
    df: pd.DataFrame,
    timestamp_column: str = 'timestamp',
    missing_value_method: str = 'interpolate',
    verbose: bool = True
) -> pd.DataFrame:
    """
    Complete preprocessing pipeline for time series data.
    
    Args:
        df: Input DataFrame
        timestamp_column: Name of the timestamp column
        missing_value_method: Method to handle missing values
        verbose: Whether to print progress information
        
    Returns:
        Preprocessed DataFrame
    """
    if verbose:
        print("=" * 80)
        print(" TIME SERIES DATA PREPROCESSING")
        print("=" * 80)
        print(f" Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print("=" * 80)
    
    original_shape = df.shape
    
    # Step 1: Parse timestamps
    if verbose:
        print("\n[1/4] Parsing timestamps...")
    df = parse_timestamps(df, timestamp_column)
    
    # Step 2: Sort chronologically
    if verbose:
        print("[2/4] Sorting chronologically...")
    df = sort_chronologically(df, timestamp_column)
    
    # Step 3: Handle missing values
    if verbose:
        print(f"[3/4] Handling missing values (method: {missing_value_method})...")
    df = handle_missing_values(df, method=missing_value_method)
    
    # Step 4: Clean data
    if verbose:
        print("[4/4] Cleaning data...")
    df = clean_data(df)
    
    if verbose:
        print("\n" + "=" * 80)
        print(" PREPROCESSING COMPLETE!")
        print("=" * 80)
        print(f"Original shape: {original_shape}")
        print(f"Final shape: {df.shape}")
        print(f"Rows removed: {original_shape[0] - df.shape[0]}")
        print("=" * 80)
    
    return df


def save_preprocessed_data(df: pd.DataFrame, output_file: str, verbose: bool = True) -> str:
    """
    Save preprocessed data to CSV file.
    
    Args:
        df: Preprocessed DataFrame
        output_file: Path to output file
        verbose: Whether to print save information
        
    Returns:
        Path to saved file
    """
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    df.to_csv(output_file, index=False)
    
    if verbose:
        print(f"[OK] Preprocessed data saved to: {output_file}")
    
    return output_file

