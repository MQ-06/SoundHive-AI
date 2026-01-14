#!/usr/bin/env python
"""
Train classical machine learning models for beehive health classification.
"""

import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from src.data import load_processed_data
from src.features import extract_time_series_features, create_labels, prepare_features_and_labels
from src.models.classical_ml import (
    train_logistic_regression,
    train_svm,
    train_decision_tree,
    train_random_forest,
    train_xgboost
)
from src.utils.visualization import plot_confusion_matrix, plot_feature_importances
import argparse


def main():
    parser = argparse.ArgumentParser(description='Train classical ML models')
    parser.add_argument(
        '--data',
        type=str,
        default='data/processed/temperature_cleaned.csv',
        help='Path to processed data file'
    )
    parser.add_argument(
        '--window',
        type=str,
        default='60min',
        help='Window size for feature extraction'
    )
    parser.add_argument(
        '--output-dir',
        type=str,
        default='results',
        help='Output directory for results'
    )
    parser.add_argument(
        '--random-state',
        type=int,
        default=42,
        help='Random state for reproducibility'
    )
    
    args = parser.parse_args()
    
    # Create output directories
    os.makedirs(f'{args.output_dir}/figures', exist_ok=True)
    os.makedirs(f'{args.output_dir}/tables', exist_ok=True)
    
    print("=" * 80)
    print(" CLASSICAL ML MODEL TRAINING")
    print("=" * 80)
    
    # Load data
    print("\n[1/5] Loading data...")
    df = load_processed_data(args.data)
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df = df.set_index('timestamp')
    
    # Feature engineering
    print("\n[2/5] Extracting features...")
    features_df = extract_time_series_features(
        df,
        value_column='temperature',
        window_size=args.window,
        verbose=True
    )
    
    # Create labels
    print("\n[3/5] Creating labels...")
    labels = create_labels(features_df, value_column='temp_mean', n_classes=3, verbose=True)
    features_df['label'] = labels
    
    # Prepare features and labels
    X, y = prepare_features_and_labels(features_df)
    
    # Train/test split
    print("\n[4/5] Splitting data...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=args.random_state
    )
    print(f"Train: {X_train.shape}, Test: {X_test.shape}")
    
    # Train models
    print("\n[5/5] Training models...")
    results = []
    
    # Logistic Regression
    print("\n--- Training Logistic Regression ---")
    lr_model, lr_results = train_logistic_regression(
        X_train.values, y_train.values,
        X_test.values, y_test.values,
        random_state=args.random_state
    )
    results.append(lr_results)
    
    # SVM
    print("\n--- Training SVM ---")
    svm_model, svm_results = train_svm(
        X_train.values, y_train.values,
        X_test.values, y_test.values,
        random_state=args.random_state
    )
    results.append(svm_results)
    
    # Decision Tree
    print("\n--- Training Decision Tree ---")
    dt_model, dt_results = train_decision_tree(
        X_train.values, y_train.values,
        X_test.values, y_test.values,
        random_state=args.random_state
    )
    results.append(dt_results)
    
    # Random Forest
    print("\n--- Training Random Forest ---")
    rf_model, rf_results = train_random_forest(
        X_train.values, y_train.values,
        X_test.values, y_test.values,
        random_state=args.random_state
    )
    results.append(rf_results)
    
    # Plot confusion matrix for Random Forest
    plot_confusion_matrix(
        rf_results['confusion_matrix'],
        model_name="Random Forest",
        save_path=f'{args.output_dir}/figures/rf_confusion_matrix.png'
    )
    
    # Plot feature importances
    importances = pd.Series(rf_model.feature_importances_, index=X.columns).sort_values(ascending=False)
    importances.to_csv(f'{args.output_dir}/tables/rf_feature_importances.csv')
    plot_feature_importances(
        importances,
        save_path=f'{args.output_dir}/figures/rf_feature_importances.png'
    )
    
    # XGBoost
    print("\n--- Training XGBoost ---")
    xgb_model, xgb_results = train_xgboost(
        X_train.values, y_train.values,
        X_test.values, y_test.values,
        random_state=args.random_state
    )
    results.append(xgb_results)
    
    # Save results
    results_df = pd.DataFrame([{
        'Model': r['model'],
        'Accuracy': r['accuracy'],
        'Precision': r['precision'],
        'Recall': r['recall'],
        'F1_Score': r['f1']
    } for r in results])
    results_df.to_csv(f'{args.output_dir}/tables/classical_ml_results.csv', index=False)
    
    print("\n" + "=" * 80)
    print(" RESULTS SUMMARY")
    print("=" * 80)
    print(results_df.to_string(index=False))
    print("=" * 80)
    print(f"\n✅ Results saved to {args.output_dir}/")


if __name__ == '__main__':
    main()

