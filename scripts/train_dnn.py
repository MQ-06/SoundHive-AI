#!/usr/bin/env python
"""
Train Deep Neural Network for beehive health classification.
"""

import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from tensorflow.keras.utils import to_categorical
from src.data import load_processed_data
from src.models.deep_learning import build_dnn, train_dnn
from src.utils.visualization import plot_training_history, plot_confusion_matrix
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, classification_report
import argparse


def main():
    parser = argparse.ArgumentParser(description='Train Deep Neural Network')
    parser.add_argument(
        '--data',
        type=str,
        default='data/processed/temperature_cleaned.csv',
        help='Path to processed data file'
    )
    parser.add_argument(
        '--epochs',
        type=int,
        default=30,
        help='Number of training epochs'
    )
    parser.add_argument(
        '--batch-size',
        type=int,
        default=32,
        help='Batch size for training'
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
    print(" DEEP NEURAL NETWORK TRAINING")
    print("=" * 80)
    
    # Load data
    print("\n[1/5] Loading data...")
    df = load_processed_data(args.data)
    
    # Create feature vectors from raw data
    print("\n[2/5] Creating feature vectors...")
    grouped = df.groupby('timestamp')['temperature'].apply(list).reset_index()
    max_sensors = grouped['temperature'].apply(len).max()
    
    feature_vectors = []
    for temp_list in grouped['temperature']:
        padded = temp_list + [np.mean(temp_list)] * (max_sensors - len(temp_list))
        feature_vectors.append(padded)
    
    X = np.array(feature_vectors)
    
    # Create labels
    print("\n[3/5] Creating labels...")
    avg_temperatures = grouped['temperature'].apply(np.mean).values
    y_labels = []
    for temp in avg_temperatures:
        if temp < 10:
            y_labels.append('Cold')
        elif temp < 25:
            y_labels.append('Normal')
        elif temp < 35:
            y_labels.append('Warm')
        else:
            y_labels.append('Hot')
    
    y = np.array(y_labels)
    encoder = LabelEncoder()
    y_encoded = encoder.fit_transform(y)
    num_classes = len(np.unique(y_encoded))
    y = to_categorical(y_encoded)
    
    print(f"Number of classes: {num_classes}")
    print(f"Classes: {encoder.classes_}")
    print(f"Feature shape: {X.shape}")
    
    # Train/test split
    print("\n[4/5] Splitting data...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=args.random_state, stratify=y_encoded
    )
    
    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    print(f"Train: {X_train.shape}, Test: {X_test.shape}")
    
    # Build and train model
    print("\n[5/5] Building and training model...")
    model = build_dnn(
        input_dim=X_train_scaled.shape[1],
        num_classes=num_classes,
        hidden_layers=[256, 128, 64]
    )
    
    model.summary()
    
    trained_model, history = train_dnn(
        model,
        X_train_scaled, y_train,
        X_test_scaled, y_test,
        epochs=args.epochs,
        batch_size=args.batch_size,
        verbose=1
    )
    
    # Evaluate
    print("\nEvaluating model...")
    y_pred_proba = trained_model.predict(X_test_scaled, verbose=0)
    y_pred = np.argmax(y_pred_proba, axis=1)
    y_test_labels = np.argmax(y_test, axis=1)
    
    accuracy = accuracy_score(y_test_labels, y_pred)
    f1 = f1_score(y_test_labels, y_pred, average='macro')
    cm = confusion_matrix(y_test_labels, y_pred)
    
    print(f"\nAccuracy: {accuracy:.4f}")
    print(f"F1-Score (macro): {f1:.4f}")
    print("\nClassification Report:")
    print(classification_report(y_test_labels, y_pred, target_names=encoder.classes_))
    
    # Save results
    results_df = pd.DataFrame({
        'Model': ['Fully Connected DNN'],
        'Accuracy': [f"{accuracy*100:.2f}%"],
        'F1_Score': [f"{f1:.4f}"]
    })
    results_df.to_csv(f'{args.output_dir}/tables/dnn_results.csv', index=False)
    
    # Plot training history
    plot_training_history(
        history,
        save_path=f'{args.output_dir}/figures/dnn_training_history.png'
    )
    
    # Plot confusion matrix
    plot_confusion_matrix(
        cm,
        class_names=encoder.classes_,
        model_name="Deep Neural Network",
        save_path=f'{args.output_dir}/figures/dnn_confusion_matrix.png'
    )
    
    print("\n" + "=" * 80)
    print("✅ Training complete! Results saved to", args.output_dir)
    print("=" * 80)


if __name__ == '__main__':
    main()

