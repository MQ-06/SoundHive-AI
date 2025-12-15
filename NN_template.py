"""
Deliverable 3 - Fully Connected Deep Neural Network for Classification
Roll no: BITF22M006 & BITF22M033
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.utils import to_categorical
import matplotlib.pyplot as plt
import seaborn as sns
import os
import warnings
warnings.filterwarnings('ignore')

# Set style for better-looking plots
try:
    plt.style.use('seaborn-v0_8-darkgrid')
except:
    plt.style.use('seaborn-darkgrid')
sns.set_palette("husl")

print("=" * 80)
print("Fully Connected Deep Neural Network Classification")
print("=" * 80)
print()

# ============================================================================
# STEP 1: DATA LOADING
# ============================================================================
print("STEP 1: Loading and Preparing Raw Data")
print("-" * 80)

# Load the cleaned temperature dataset
# EXPLANATION: We use the preprocessed data that has been cleaned, but we'll
# use it in its raw form without additional feature engineering
data_path = "data/processed/temperature_cleaned.csv"
data = pd.read_csv(data_path)

print(f"[OK] Loaded dataset: {data.shape[0]} rows, {data.shape[1]} columns")
print(f"[OK] Columns: {data.columns.tolist()}")

# ============================================================================
# STEP 2: CREATE FEATURE VECTORS FROM RAW DATA
# ============================================================================
print("\nSTEP 2: Creating Feature Vectors from Raw Time-Series Data")
print("-" * 80)

# EXPLANATION: The dataset has multiple temperature readings per timestamp
# (multiple sensors). We need to group by timestamp and create feature vectors.
# This is the "flattening" step mentioned in the requirements - converting
# multi-dimensional time-series data into 1D feature vectors per sample.

# Group by timestamp and collect all temperature readings as features
# This creates a feature vector where each element is a sensor reading
grouped = data.groupby('timestamp')['temperature'].apply(list).reset_index()

# Find the maximum number of sensors (readings per timestamp)
max_sensors = grouped['temperature'].apply(len).max()
print(f"[INFO] Maximum number of sensors per timestamp: {max_sensors}")

# Pad shorter sequences and create feature matrix
# EXPLANATION: Not all timestamps have the same number of readings.
# We pad shorter sequences with the mean temperature to create uniform vectors.
feature_vectors = []
for temp_list in grouped['temperature']:
    # Pad to max_sensors length using the mean of available readings
    padded = temp_list + [np.mean(temp_list)] * (max_sensors - len(temp_list))
    feature_vectors.append(padded)

X = np.array(feature_vectors)
print(f"[OK] Feature matrix shape: {X.shape}")
print(f"[OK] Input dimension: {X.shape[1]} features per sample")
print(f"[EXPLANATION] Each sample now has {X.shape[1]} features representing")
print(f"              all sensor readings at that timestamp (raw data).")

# ============================================================================
# STEP 3: CREATE CLASSIFICATION LABELS
# ============================================================================
print("\nSTEP 3: Creating Classification Labels from Raw Data")
print("-" * 80)

# EXPLANATION: Since this is a classification task, we need to create labels.
# We'll create multi-class labels based on average temperature ranges:
# - Cold: < 10°C
# - Normal: 10-25°C  
# - Warm: 25-35°C
# - Hot: > 35°C
# This makes sense for beehive monitoring as different temperature ranges
# indicate different hive states.

# Calculate average temperature per timestamp (using mean of all sensors)
avg_temperatures = grouped['temperature'].apply(np.mean).values

# Create labels based on temperature ranges
# EXPLANATION: This is a domain-appropriate way to create labels from raw data
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

# Count class distribution
unique, counts = np.unique(y, return_counts=True)
print(f"[OK] Created {len(unique)} classes: {unique.tolist()}")
print(f"[OK] Class distribution:")
for cls, cnt in zip(unique, counts):
    print(f"      {cls}: {cnt} samples ({100*cnt/len(y):.2f}%)")

# ============================================================================
# STEP 4: LABEL ENCODING
# ============================================================================
print("\nSTEP 4: Encoding Class Labels")
print("-" * 80)

# EXPLANATION: Neural networks need numeric labels. LabelEncoder converts
# string labels ('Cold', 'Normal', etc.) to integers (0, 1, 2, 3).
encoder = LabelEncoder()
y_encoded = encoder.fit_transform(y)

num_classes = len(np.unique(y_encoded))
print(f"[OK] Number of classes: {num_classes}")
print(f"[OK] Encoded labels: {encoder.classes_}")

# EXPLANATION: For multi-class classification, we need one-hot encoding
# This converts class indices to binary vectors (e.g., class 2 -> [0,0,1,0])
if num_classes > 2:
    y = to_categorical(y_encoded)
    print(f"[OK] Converted to one-hot encoding: shape {y.shape}")
    print(f"[EXPLANATION] One-hot encoding: each sample is a vector where")
    print(f"              only the correct class position is 1, others are 0.")
else:
    y = y_encoded
    print(f"[OK] Binary classification: using integer labels")

# ============================================================================
# STEP 5: TRAIN-TEST SPLIT
# ============================================================================
print("\nSTEP 5: Splitting Dataset into Training and Testing Sets")
print("-" * 80)

# EXPLANATION: We split the data into 80% training and 20% testing.
# This allows us to train on one portion and evaluate on unseen data.
# random_state=42 ensures reproducibility.
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y_encoded
)

print(f"[OK] Training set: {X_train.shape[0]} samples")
print(f"[OK] Testing set: {X_test.shape[0]} samples")
print(f"[OK] Split ratio: 80% training, 20% testing")
print(f"[EXPLANATION] Stratified split ensures each class is represented")
print(f"              proportionally in both training and testing sets.")

# ============================================================================
# STEP 6: FEATURE NORMALIZATION (STANDARD SCALING)
# ============================================================================
print("\nSTEP 6: Normalizing Input Features (Standard Scaling)")
print("-" * 80)

# EXPLANATION: Neural networks work better when features are on similar scales.
# StandardScaler transforms features to have mean=0 and std=1.
# This prevents features with larger values from dominating the learning.
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

print(f"[OK] Applied StandardScaler to normalize features")
print(f"[OK] Training features - Mean: {X_train.mean():.4f}, Std: {X_train.std():.4f}")
print(f"[OK] Testing features - Mean: {X_test.mean():.4f}, Std: {X_test.std():.4f}")
print(f"[EXPLANATION] Normalization ensures all features contribute equally")
print(f"              to the model, improving training stability and convergence.")

# ============================================================================
# STEP 7: BUILD FULLY CONNECTED DNN ARCHITECTURE
# ============================================================================
print("\nSTEP 7: Building Fully Connected Deep Neural Network")
print("-" * 80)

# EXPLANATION: We're building a Sequential model (feed-forward network).
# This is a fully connected DNN where each neuron in one layer connects
# to all neurons in the next layer.

model = Sequential()

# Input layer + First hidden layer
# EXPLANATION: Dense(256) means 256 neurons. ReLU activation introduces
# non-linearity, allowing the model to learn complex patterns.
# input_shape specifies the number of input features.
model.add(Dense(256, activation='relu', input_shape=(X_train.shape[1],)))
print(f"[OK] Added Dense layer: 256 neurons, ReLU activation, input_shape={X_train.shape[1]}")

# Dropout layer
# EXPLANATION: Dropout randomly sets 30% of neurons to 0 during training.
# This prevents overfitting by forcing the model to not rely on specific neurons.
model.add(Dropout(0.3))
print(f"[OK] Added Dropout layer: 0.3 (30% dropout rate)")

# Second hidden layer
model.add(Dense(128, activation='relu'))
print(f"[OK] Added Dense layer: 128 neurons, ReLU activation")

# Dropout layer
model.add(Dropout(0.3))
print(f"[OK] Added Dropout layer: 0.3 (30% dropout rate)")

# Third hidden layer
model.add(Dense(64, activation='relu'))
print(f"[OK] Added Dense layer: 64 neurons, ReLU activation")

# Output layer
# EXPLANATION: For multi-class classification, we use:
# - num_classes neurons (one per class)
# - Softmax activation (converts outputs to probabilities that sum to 1)
# For binary classification, we would use:
# - 1 neuron with Sigmoid activation
if num_classes == 2:
    model.add(Dense(1, activation='sigmoid'))
    print(f"[OK] Added Output layer: 1 neuron, Sigmoid activation (binary classification)")
    loss_function = 'binary_crossentropy'
else:
    model.add(Dense(num_classes, activation='softmax'))
    print(f"[OK] Added Output layer: {num_classes} neurons, Softmax activation (multi-class)")
    loss_function = 'categorical_crossentropy'

# Compile the model
# EXPLANATION: 
# - Optimizer: Adam (adaptive learning rate, works well in practice)
# - Loss: Cross-entropy measures how far predictions are from true labels
# - Metrics: Accuracy tracks performance during training
model.compile(
    optimizer='adam',
    loss=loss_function,
    metrics=['accuracy']
)

print(f"[OK] Model compiled with:")
print(f"     Optimizer: Adam")
print(f"     Loss function: {loss_function}")
print(f"     Metrics: accuracy")

# Display model architecture
print(f"\n[INFO] Model Architecture Summary:")
model.summary()

# ============================================================================
# STEP 8: TRAIN THE MODEL
# ============================================================================
print("\nSTEP 8: Training the Model")
print("-" * 80)

# EXPLANATION: Training parameters:
# - epochs=30: Model sees the entire training set 30 times
# - batch_size=32: Processes 32 samples at a time (memory efficient)
# - validation_split=0.2: Uses 20% of training data for validation during training
print(f"[INFO] Training configuration:")
print(f"     Epochs: 30")
print(f"     Batch size: 32")
print(f"     Validation split: 0.2 (20% of training data)")
print(f"\n[INFO] Starting training...")

history = model.fit(
    X_train, y_train,
    epochs=30,
    batch_size=32,
    validation_split=0.2,
    verbose=1
)

print(f"[OK] Training completed!")

# ============================================================================
# STEP 9: MODEL EVALUATION
# ============================================================================
print("\nSTEP 9: Evaluating Model Performance")
print("-" * 80)

# Make predictions on test set
# EXPLANATION: model.predict() returns probability distributions for each class
y_pred_proba = model.predict(X_test, verbose=0)

# Convert predictions to class labels
# EXPLANATION: For multi-class, predictions are probability vectors.
# We take the class with highest probability (argmax).
if num_classes > 2:
    y_pred = np.argmax(y_pred_proba, axis=1)
    y_test_labels = np.argmax(y_test, axis=1)
else:
    y_pred = (y_pred_proba > 0.5).astype(int).flatten()
    y_test_labels = y_test

# Calculate metrics
accuracy = accuracy_score(y_test_labels, y_pred)
f1 = f1_score(y_test_labels, y_pred, average='macro')
f1_per_class = f1_score(y_test_labels, y_pred, average=None)

print(f"[OK] Evaluation Metrics:")
print(f"     Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)")
print(f"     F1-Score (macro): {f1:.4f}")

# Detailed classification report
print(f"\n[INFO] Detailed Classification Report:")
print(classification_report(y_test_labels, y_pred, target_names=encoder.classes_))


# ============================================================================
# STEP 10: RESULTS TABLE 
# ============================================================================
print("\n" + "=" * 80)
print("EXPERIMENTAL RESULTS")
print("=" * 80)
print(f"{'Model':<30} {'Input Type':<20} {'Accuracy':<15} {'F1-Score':<15}")
print("-" * 80)
print(f"{'Fully Connected DNN':<30} {'Raw Data':<20} {accuracy*100:>6.2f}%        {f1:.4f}")
print("=" * 80)

# Save results to CSV
results_df = pd.DataFrame({
    'Model': ['Fully Connected DNN'],
    'Input_Type': ['Raw Data'],
    'Accuracy': [f"{accuracy*100:.2f}%"],
    'F1_Score': [f"{f1:.4f}"]
})
results_df.to_csv('deliverable_3_results.csv', index=False)
print(f"\n[OK] Results saved to: deliverable_3_results.csv")

