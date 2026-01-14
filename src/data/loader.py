"""Data loading utilities for beehive sensor data"""

import pandas as pd
import os
from typing import Optional


def load_data(filepath: str, verbose: bool = True) -> pd.DataFrame:
    """
    Load beehive sensor data from CSV file.
    
    Args:
        filepath: Path to the CSV file
        verbose: Whether to print loading information
        
    Returns:
        DataFrame with loaded data
    """
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Data file not found: {filepath}")
    
    df = pd.read_csv(filepath)
    
    if verbose:
        print(f"[OK] Loaded dataset from: {filepath}")
        print(f"[OK] Shape: {df.shape[0]} rows x {df.shape[1]} columns")
        print(f"[OK] Columns: {df.columns.tolist()}")
    
    return df


def load_processed_data(filepath: Optional[str] = None) -> pd.DataFrame:
    """
    Load preprocessed temperature data.
    
    Args:
        filepath: Path to processed data file. If None, uses default path.
        
    Returns:
        DataFrame with processed data
    """
    if filepath is None:
        # Default path relative to project root
        filepath = "data/processed/temperature_cleaned.csv"
    
    return load_data(filepath)

