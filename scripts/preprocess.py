#!/usr/bin/env python
"""
Data preprocessing script for beehive sensor data.
"""

import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.data import load_data, preprocess_timeseries, save_preprocessed_data
import argparse


def main():
    parser = argparse.ArgumentParser(description='Preprocess beehive sensor data')
    parser.add_argument(
        '--input',
        type=str,
        default='data/raw/new_ds/temperature_2017.csv',
        help='Path to input CSV file'
    )
    parser.add_argument(
        '--output',
        type=str,
        default='data/processed/temperature_cleaned.csv',
        help='Path to output CSV file'
    )
    parser.add_argument(
        '--method',
        type=str,
        default='interpolate',
        choices=['interpolate', 'ffill', 'bfill', 'drop'],
        help='Method for handling missing values'
    )
    
    args = parser.parse_args()
    
    # Load data
    print("Loading data...")
    df = load_data(args.input, verbose=True)
    
    # Preprocess
    print("\nPreprocessing data...")
    df_processed = preprocess_timeseries(
        df,
        missing_value_method=args.method,
        verbose=True
    )
    
    # Save
    print("\nSaving preprocessed data...")
    save_preprocessed_data(df_processed, args.output, verbose=True)
    
    print("\n✅ Preprocessing complete!")


if __name__ == '__main__':
    main()

