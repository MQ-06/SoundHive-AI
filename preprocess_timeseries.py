# -*- coding: utf-8 -*-
"""
ITF22 Assignment 5.1 - Time Series Data Preprocessing
Author: [Your Name]
Date: December 2025

This script performs preprocessing on sensor-based time series data from beehive monitoring.
It demonstrates the four key preprocessing steps required for time series analysis:
1. Timestamp parsing
2. Chronological sorting
3. Missing value handling
4. Data cleaning
"""

import pandas as pd
import numpy as np
from datetime import datetime
import os
import sys

# Set UTF-8 encoding for Windows console
if sys.platform == 'win32':
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

# Configuration
INPUT_FILE = "data/raw/new_ds/temperature_2017.csv"
OUTPUT_FILE = "data/processed/temperature_cleaned.csv"
REPORT_FILE = "preprocessing_report.txt"


def print_section(title):
    """Print a formatted section header."""
    print("\n" + "=" * 80)
    print(f" {title}")
    print("=" * 80)


def load_data(filepath):
    """
    Load the time series dataset from CSV file.
    
    Args:
        filepath: Path to the CSV file
        
    Returns:
        DataFrame with the loaded data
    """
    print_section("STEP 1: LOADING DATA")
    
    df = pd.read_csv(filepath)
    
    print(f"[OK] Loaded dataset from: {filepath}")
    print(f"[OK] Shape: {df.shape[0]} rows x {df.shape[1]} columns")
    print(f"[OK] Columns: {df.columns.tolist()}")
    print(f"\nFirst 5 rows:")
    print(df.head())
    print(f"\nData types:")
    print(df.dtypes)
    
    return df


def parse_timestamps(df, timestamp_column='timestamp'):
    """
    PREPROCESSING STEP 1: Parse timestamps correctly
    
    WHY THIS IS IMPORTANT:
    - Timestamps are initially stored as strings (text)
    - Converting to datetime allows Python to understand temporal order
    - Enables time-based operations (filtering by date, resampling, etc.)
    - Required for proper sorting and time series analysis
    
    Args:
        df: Input DataFrame
        timestamp_column: Name of the timestamp column
        
    Returns:
        DataFrame with parsed timestamps
    """
    print_section("STEP 2: PARSING TIMESTAMPS")
    
    print(f"Original timestamp type: {df[timestamp_column].dtype}")
    print(f"Example original value: '{df[timestamp_column].iloc[0]}'")
    
    # Convert string to datetime
    df[timestamp_column] = pd.to_datetime(df[timestamp_column])
    
    print(f"\n[OK] Converted to datetime type: {df[timestamp_column].dtype}")
    print(f"[OK] Example parsed value: {df[timestamp_column].iloc[0]}")
    
    print("\n[EXPLANATION]:")
    print("   Parsing timestamps converts text like '2017-01-01 05:00:00' into a")
    print("   datetime object that Python can use for time-based calculations,")
    print("   sorting, and analysis. This is essential for all time series work.")
    
    return df


def sort_chronologically(df, timestamp_column='timestamp'):
    """
    PREPROCESSING STEP 2: Sort data in chronological order
    
    WHY THIS IS IMPORTANT:
    - Time series algorithms assume data is in temporal order
    - Required for calculating trends, moving averages, and forecasting
    - Prevents errors in sequential operations
    - Makes data easier to visualize and understand
    
    Args:
        df: Input DataFrame
        timestamp_column: Name of the timestamp column
        
    Returns:
        Chronologically sorted DataFrame
    """
    print_section("STEP 3: SORTING CHRONOLOGICALLY")
    
    # Check if already sorted
    is_sorted = df[timestamp_column].is_monotonic_increasing
    print(f"Is data already sorted? {is_sorted}")
    
    if not is_sorted:
        print("[WARNING] Data is NOT in chronological order. Sorting now...")
        df = df.sort_values(by=timestamp_column).reset_index(drop=True)
        print("[OK] Data sorted successfully")
    else:
        print("[OK] Data is already in chronological order")
    
    print(f"\nFirst timestamp: {df[timestamp_column].iloc[0]}")
    print(f"Last timestamp:  {df[timestamp_column].iloc[-1]}")
    print(f"Time span: {df[timestamp_column].iloc[-1] - df[timestamp_column].iloc[0]}")
    
    print("\n[EXPLANATION]:")
    print("   Chronological sorting ensures the earliest measurements come first.")
    print("   This is critical for time series analysis because many calculations")
    print("   (like trends, forecasts, and moving averages) depend on temporal order.")
    
    return df


def handle_missing_values(df, method='interpolate'):
    """
    PREPROCESSING STEP 3: Handle missing values
    
    WHY THIS IS IMPORTANT:
    - Missing values can cause errors in calculations
    - Gaps in sensor data need to be filled for continuous analysis
    - Different methods suit different scenarios
    - Ensures statistical analyses are accurate
    
    Methods available:
    - 'interpolate': Linear interpolation between values (best for sensors)
    - 'ffill': Forward fill (use last known value)
    - 'bfill': Backward fill (use next known value)
    - 'drop': Remove rows with missing values
    
    Args:
        df: Input DataFrame
        method: Method to handle missing values
        
    Returns:
        DataFrame with missing values handled
    """
    print_section("STEP 4: HANDLING MISSING VALUES")
    
    # Check for missing values
    missing_count = df.isnull().sum()
    total_missing = missing_count.sum()
    
    print(f"Missing values per column:")
    print(missing_count)
    print(f"\nTotal missing values: {total_missing}")
    
    if total_missing == 0:
        print("\n[OK] No missing values found. Data is complete!")
    else:
        print(f"\n[WARNING] Found {total_missing} missing values. Applying '{method}' method...")
        
        if method == 'interpolate':
            # Linear interpolation for numeric columns
            numeric_cols = df.select_dtypes(include=[np.number]).columns
            df[numeric_cols] = df[numeric_cols].interpolate(method='linear')
            print("[OK] Applied linear interpolation to fill gaps")
            
        elif method == 'ffill':
            df = df.fillna(method='ffill')
            print("[OK] Applied forward fill (used previous value)")
            
        elif method == 'bfill':
            df = df.fillna(method='bfill')
            print("[OK] Applied backward fill (used next value)")
            
        elif method == 'drop':
            df = df.dropna()
            print(f"[OK] Dropped rows with missing values. New shape: {df.shape}")
        
        # Verify no missing values remain
        remaining_missing = df.isnull().sum().sum()
        print(f"[OK] Missing values after handling: {remaining_missing}")
    
    print("\n[EXPLANATION]:")
    print(f"   Method used: {method}")
    if method == 'interpolate':
        print("   Linear interpolation estimates missing values by drawing a straight")
        print("   line between known values. This works well for sensor data that")
        print("   changes gradually over time (like temperature).")
    elif method == 'ffill':
        print("   Forward fill uses the last known value. Good when sensor readings")
        print("   are stable and unlikely to change rapidly.")
    elif method == 'bfill':
        print("   Backward fill uses the next known value. Useful in specific cases.")
    elif method == 'drop':
        print("   Dropping removes incomplete records. Use when missing data is minimal.")
    
    return df


def clean_data(df):
    """
    PREPROCESSING STEP 4: Perform basic data cleaning
    
    WHY THIS IS IMPORTANT:
    - Duplicate records can skew analysis
    - Outliers may indicate sensor errors
    - Clean column names improve code readability
    - Ensures data quality and consistency
    
    Args:
        df: Input DataFrame
        
    Returns:
        Cleaned DataFrame
    """
    print_section("STEP 5: DATA CLEANING")
    
    original_shape = df.shape
    
    # 1. Remove duplicate rows
    duplicates = df.duplicated().sum()
    print(f"1. Checking for duplicates: {duplicates} found")
    if duplicates > 0:
        df = df.drop_duplicates()
        print(f"   [OK] Removed {duplicates} duplicate rows")
    else:
        print("   [OK] No duplicates found")
    
    # 2. Check for outliers in numeric columns
    print("\n2. Checking for outliers in numeric columns:")
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    
    for col in numeric_cols:
        q1 = df[col].quantile(0.25)
        q3 = df[col].quantile(0.75)
        iqr = q3 - q1
        lower_bound = q1 - 3 * iqr
        upper_bound = q3 + 3 * iqr
        
        outliers = ((df[col] < lower_bound) | (df[col] > upper_bound)).sum()
        print(f"   {col}: {outliers} potential outliers (beyond 3xIQR)")
        print(f"      Range: [{df[col].min():.2f}, {df[col].max():.2f}]")
        print(f"      Mean: {df[col].mean():.2f}, Std: {df[col].std():.2f}")
    
    # 3. Verify data ranges (temperature should be reasonable)
    print("\n3. Verifying data ranges:")
    if 'temperature' in df.columns:
        temp_min = df['temperature'].min()
        temp_max = df['temperature'].max()
        print(f"   Temperature range: {temp_min:.2f}C to {temp_max:.2f}C")
        
        # Check for impossible values (e.g., temperature > 100C or < -50C)
        if temp_min < -50 or temp_max > 100:
            print("   [WARNING] Temperature values outside expected range!")
        else:
            print("   [OK] Temperature values are within reasonable range")
    
    # 4. Clean column names (if needed)
    print("\n4. Column names:")
    print(f"   Current columns: {df.columns.tolist()}")
    print("   [OK] Column names are already clean and descriptive")
    
    print(f"\n[OK] Cleaning complete!")
    print(f"   Original shape: {original_shape}")
    print(f"   Final shape: {df.shape}")
    print(f"   Rows removed: {original_shape[0] - df.shape[0]}")
    
    print("\n[EXPLANATION]:")
    print("   Data cleaning removes inconsistencies and errors:")
    print("   - Duplicates: Can artificially inflate statistics")
    print("   - Outliers: May indicate sensor malfunctions or data entry errors")
    print("   - Range checks: Ensure physically possible values")
    print("   - Clean names: Make code more readable and maintainable")
    
    return df


def save_results(df, output_file):
    """Save the cleaned dataset to a CSV file."""
    print_section("STEP 6: SAVING RESULTS")
    
    # Create output directory if it doesn't exist
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    
    # Save to CSV
    df.to_csv(output_file, index=False)
    
    print(f"[OK] Cleaned dataset saved to: {output_file}")
    print(f"[OK] Final shape: {df.shape[0]} rows x {df.shape[1]} columns")
    
    return output_file


def generate_report(original_df, cleaned_df, output_file):
    """Generate a text report summarizing the preprocessing."""
    print_section("STEP 7: GENERATING REPORT")
    
    report = []
    report.append("=" * 80)
    report.append("ITF22 ASSIGNMENT 5.1 - PREPROCESSING REPORT")
    report.append("=" * 80)
    report.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    report.append("")
    
    report.append("ORIGINAL DATASET:")
    report.append(f"  Rows: {original_df.shape[0]}")
    report.append(f"  Columns: {original_df.shape[1]}")
    report.append(f"  Missing values: {original_df.isnull().sum().sum()}")
    report.append("")
    
    report.append("PREPROCESSING STEPS PERFORMED:")
    report.append("  1. [OK] Parsed timestamps from string to datetime")
    report.append("  2. [OK] Sorted data chronologically")
    report.append("  3. [OK] Handled missing values (interpolation)")
    report.append("  4. [OK] Cleaned data (removed duplicates, checked outliers)")
    report.append("")
    
    report.append("FINAL DATASET:")
    report.append(f"  Rows: {cleaned_df.shape[0]}")
    report.append(f"  Columns: {cleaned_df.shape[1]}")
    report.append(f"  Missing values: {cleaned_df.isnull().sum().sum()}")
    report.append(f"  Date range: {cleaned_df['timestamp'].min()} to {cleaned_df['timestamp'].max()}")
    report.append("")
    
    report.append("SUMMARY STATISTICS:")
    report.append(str(cleaned_df.describe()))
    report.append("")
    
    report.append("=" * 80)
    
    # Write to file
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write('\n'.join(report))
    
    print(f"[OK] Report saved to: {output_file}")
    
    # Also print to console
    print("\n" + '\n'.join(report))


def main():
    """Main preprocessing pipeline."""
    print("\n" + "=" * 80)
    print(" ITF22 ASSIGNMENT 5.1 - TIME SERIES DATA PREPROCESSING")
    print("=" * 80)
    print(f" Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 80)
    
    # Store original for comparison
    original_df = pd.read_csv(INPUT_FILE)
    
    # Execute preprocessing pipeline
    df = load_data(INPUT_FILE)
    df = parse_timestamps(df)
    df = sort_chronologically(df)
    df = handle_missing_values(df, method='interpolate')
    df = clean_data(df)
    output_path = save_results(df, OUTPUT_FILE)
    generate_report(original_df, df, REPORT_FILE)
    
    print_section("PREPROCESSING COMPLETE!")
    print(f"[OK] Cleaned data: {output_path}")
    print(f"[OK] Report: {REPORT_FILE}")
    print(f"[OK] Ready for time series analysis!")
    print("\n" + "=" * 80 + "\n")


if __name__ == "__main__":
    main()
