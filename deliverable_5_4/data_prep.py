"""
Data preparation for sequence-based models
Converts raw time-series data into sequences for LSTM/GRU/RNN
"""

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical


def load_data(filepath):
    """Load cleaned temperature dataset"""
    data = pd.read_csv(filepath)
    data['timestamp'] = pd.to_datetime(data['timestamp'])
    return data


def group_by_timestamp(data, feature_col='temperature'):
    """Group multiple sensor readings per timestamp"""
    grouped = data.groupby('timestamp')[feature_col].apply(list).reset_index()
    return grouped


def create_labels(grouped_data, cold_thresh=10, normal_thresh=25, warm_thresh=35, feature_col='temperature'):
    """Create classification labels based on average temperature"""
    avg_temps = grouped_data[feature_col].apply(np.mean).values
    
    labels = []
    for temp in avg_temps:
        if temp < cold_thresh:
            labels.append('Cold')
        elif temp < normal_thresh:
            labels.append('Normal')
        elif temp < warm_thresh:
            labels.append('Warm')
        else:
            labels.append('Hot')
    
    return np.array(labels)


def create_sequences(data_values, labels, sequence_length):
    """
    Create sliding window sequences from time-series data
    
    Args:
        data_values: 1D array of values (one per timestamp)
        labels: Corresponding labels
        sequence_length: Number of timestamps per sequence
    
    Returns:
        X: Sequences array (samples, time_steps, features)
        y: Labels array
    """
    X, y = [], []
    
    for i in range(len(data_values) - sequence_length + 1):
        sequence = data_values[i:i + sequence_length]
        label = labels[i + sequence_length - 1]
        
        X.append(sequence)
        y.append(label)
    
    X = np.array(X)
    y = np.array(y)
    
    # Reshape to 3D: (samples, time_steps, features)
    X = X.reshape((X.shape[0], X.shape[1], 1))
    
    return X, y


def prepare_sequences(data_path, sequence_length, cold_thresh, normal_thresh, warm_thresh, feature_col='temperature'):
    """
    Main function to prepare data for sequence models
    
    Returns:
        X_train, X_test, y_train, y_test, scaler, encoder, num_classes
    """
    # Load and group data
    data = load_data(data_path)
    grouped = group_by_timestamp(data, feature_col)
    
    # Create labels
    labels = create_labels(grouped, cold_thresh, normal_thresh, warm_thresh, feature_col)
    
    # Get average temperature per timestamp for sequences
    avg_temps = grouped[feature_col].apply(np.mean).values
    
    # Normalize temperature values
    scaler = StandardScaler()
    normalized_temps = scaler.fit_transform(avg_temps.reshape(-1, 1)).flatten()
    
    # Create sequences
    X, y = create_sequences(normalized_temps, labels, sequence_length)
    
    # Encode labels
    encoder = LabelEncoder()
    y_encoded = encoder.fit_transform(y)
    num_classes = len(encoder.classes_)
    
    # Convert to one-hot for multi-class
    if num_classes > 2:
        y = to_categorical(y_encoded)
    
    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y_encoded
    )
    
    return X_train, X_test, y_train, y_test, scaler, encoder, num_classes


def get_data_info(X_train, X_test, y_train, y_test, encoder):
    """Print data information"""
    print(f"Training sequences: {X_train.shape[0]}")
    print(f"Test sequences: {X_test.shape[0]}")
    print(f"Sequence length: {X_train.shape[1]} timestamps")
    print(f"Features per timestep: {X_train.shape[2]}")
    print(f"Number of classes: {len(encoder.classes_)}")
    print(f"Classes: {encoder.classes_}")
    
    # Class distribution
    if len(y_train.shape) == 2:
        train_counts = np.sum(y_train, axis=0)
        print("\nTraining class distribution:")
        for i, cls in enumerate(encoder.classes_):
            print(f"  {cls}: {train_counts[i]} samples")

