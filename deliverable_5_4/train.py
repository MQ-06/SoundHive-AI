"""
Main training script for sequence-based deep learning classification
Deliverable 5.4: Sequence-Based Deep Learning Models (RNN/LSTM/GRU)
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, classification_report, confusion_matrix

from config import *
from data_prep import prepare_sequences, get_data_info
from model_builder import build_model

# Set style
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")

print("=" * 80)
print("Sequence-Based Deep Learning Classification (LSTM/GRU/RNN)")
print("=" * 80)
print()

# Create output directory
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Prepare data
print("STEP 1: Preparing Sequential Data")
print("-" * 80)
X_train, X_test, y_train, y_test, scaler, encoder, num_classes = prepare_sequences(
    DATA_PATH, SEQUENCE_LENGTH, COLD_THRESHOLD, NORMAL_THRESHOLD, WARM_THRESHOLD, FEATURE_COLUMN
)

print("\nData Summary:")
get_data_info(X_train, X_test, y_train, y_test, encoder)

# Build model
print(f"\nSTEP 2: Building {MODEL_TYPE} Model")
print("-" * 80)
input_shape = (X_train.shape[1], X_train.shape[2])
model = build_model(MODEL_TYPE, input_shape, num_classes, LSTM_UNITS, DROPOUT_RATE, DENSE_UNITS)

print(f"Model architecture:")
model.summary()

# Train model
print(f"\nSTEP 3: Training {MODEL_TYPE} Model")
print("-" * 80)
print(f"Epochs: {EPOCHS}")
print(f"Batch size: {BATCH_SIZE}")
print(f"Validation split: {VALIDATION_SPLIT}")

history = model.fit(
    X_train, y_train,
    epochs=EPOCHS,
    batch_size=BATCH_SIZE,
    validation_split=VALIDATION_SPLIT,
    verbose=1
)

# Evaluate model
print(f"\nSTEP 4: Evaluating Model")
print("-" * 80)
y_pred_proba = model.predict(X_test, verbose=0)

if num_classes > 2:
    y_pred = np.argmax(y_pred_proba, axis=1)
    y_test_labels = np.argmax(y_test, axis=1)
else:
    y_pred = (y_pred_proba > 0.5).astype(int).flatten()
    y_test_labels = y_test

# Calculate metrics
accuracy = accuracy_score(y_test_labels, y_pred)
precision = precision_score(y_test_labels, y_pred, average='macro', zero_division=0)
recall = recall_score(y_test_labels, y_pred, average='macro', zero_division=0)
f1 = f1_score(y_test_labels, y_pred, average='macro', zero_division=0)

print(f"Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)")
print(f"Precision: {precision:.4f}")
print(f"Recall: {recall:.4f}")
print(f"F1-Score: {f1:.4f}")

print(f"\nDetailed Classification Report:")
print(classification_report(y_test_labels, y_pred, target_names=encoder.classes_))

# Save results
results_df = pd.DataFrame({
    'Model': [f'{MODEL_TYPE}'],
    'Input_Type': ['Raw Sequential Data'],
    'Sequence_Length': [SEQUENCE_LENGTH],
    'Accuracy': [f"{accuracy*100:.2f}%"],
    'Precision': [f"{precision:.4f}"],
    'Recall': [f"{recall:.4f}"],
    'F1_Score': [f"{f1:.4f}"]
})

results_path = os.path.join(OUTPUT_DIR, 'results.csv')
results_df.to_csv(results_path, index=False)
print(f"\nResults saved to: {results_path}")

# Plot training history
plt.figure(figsize=(12, 4))

plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='Train Accuracy')
plt.plot(history.history['val_accuracy'], label='Val Accuracy')
plt.title('Model Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
plt.grid(True, alpha=0.3)

plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Val Loss')
plt.title('Model Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, 'training_history.png'), dpi=300, bbox_inches='tight')
plt.close()
print(f"Training history plot saved to: {os.path.join(OUTPUT_DIR, 'training_history.png')}")

# Confusion matrix
cm = confusion_matrix(y_test_labels, y_pred)
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=encoder.classes_, yticklabels=encoder.classes_)
plt.title(f'{MODEL_TYPE} Confusion Matrix')
plt.ylabel('True Label')
plt.xlabel('Predicted Label')
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, 'confusion_matrix.png'), dpi=300, bbox_inches='tight')
plt.close()
print(f"Confusion matrix saved to: {os.path.join(OUTPUT_DIR, 'confusion_matrix.png')}")

# Final summary
print("\n" + "=" * 80)
print("EXPERIMENTAL RESULTS")
print("=" * 80)
print(f"{'Model':<30} {'Input Type':<25} {'Accuracy':<15} {'F1-Score':<15}")
print("-" * 80)
print(f"{MODEL_TYPE:<30} {'Raw Sequential Data':<25} {accuracy*100:>6.2f}%        {f1:.4f}")
print("=" * 80)

print("\n[OK] Training completed successfully!")

