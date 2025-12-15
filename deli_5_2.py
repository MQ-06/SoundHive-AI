"""


1. OBJECTIVE
-----------
The goal of this project is to build baseline classification models using hand-crafted 
features extracted from time-series sensor data, combined with classical machine learning 
classifiers (Logistic Regression, SVM, Decision Tree, Random Forest, and XGBoost/Gradient 
Boosting). The performance of these models will be compared to establish a baseline for 
further deep-learning approaches.

2. DATASET DESCRIPTION
---------------------
- Dataset Name: Beehive Temperature Sensor Data
- Source: Kaggle (https://www.kaggle.com/datasets/se18m502/bee-hive)
- Project: HOBOS (HOneyBee Online Studies)
- Nature/Type: Time-series sensor data (continuous temperature measurements)
- Raw Instances: ~400,000 rows (after preprocessing)
- Aggregated Instances: Variable (depends on window size - 60 minutes)
- Features/Variables: 
  * Raw: timestamp, temperature (°C)
  * Engineered: 12 features (statistical + temporal)
- Target Variable: 3-class classification labels
  * Class 0: Low temperature (below 25th percentile)
  * Class 1: Medium temperature (25th-75th percentile)
  * Class 2: High temperature (above 75th percentile)
- Metadata:
  * Units: Temperature in Celsius (°C)
  * Sampling Rate: Approximately hourly readings
  * Time Period: January 1, 2017 - December 31, 2017
  * Missing Values: Handled via interpolation in preprocessing
- Dataset Quality:
  * Missing values: Minimal (3 missing values, handled)
  * Class imbalance: Checked via label distribution
  * Noise: Sensor data may contain measurement noise
  * Preprocessing: Required (timestamp parsing, sorting, missing value handling)

3. FEATURE ENGINEERING
---------------------
For time-series/sensor data, the following preprocessing and feature extraction steps are applied:

a) Data Segmentation:
   - Segmenting data into 60-minute windows using pandas resample()
   - Each window represents aggregated statistics over 1 hour

b) Statistical Feature Extraction:
   - Mean temperature per window
   - Standard deviation (variability)
   - Minimum and maximum values
   - Median (robust central tendency)
   - Range (max - min)
   - Skewness (distribution shape)
   - First and last values in window
   - Difference between first and last values

c) Temporal Features:
   - Hour of day (0-23) - encoded as sin/cos for cyclical representation
   - Day of week (0-6, Monday=0)
   - Cyclical encoding prevents artificial distance between hours (e.g., 23:00 and 00:00)

d) Feature Transformations:
   - Standardization: Applied to features for LR and SVM (StandardScaler)
   - Missing value handling: Fill NaN with 0 for statistical features
   - Feature selection: Removed intermediate features (cnt, temp_first, temp_last, hour)

4. MODELS / CLASSIFIERS
----------------------
Implemented classical (non-deep-learning) classifiers:

a) Logistic Regression (LR)
   - Hyperparameters: max_iter=1000, random_state=42
   - Scaling: Yes (StandardScaler)
   - Notes: Linear classifier, good baseline

b) Support Vector Machine (SVM)
   - Kernel: RBF (Radial Basis Function)
   - Hyperparameters: random_state=42, cache_size=500, max_iter=1000
   - Scaling: Yes (StandardScaler required)
   - Notes: Can be slow on large datasets

c) Decision Tree (DT)
   - Hyperparameters: random_state=42 (default others)
   - Scaling: No (tree-based models don't require scaling)
   - Notes: Interpretable, prone to overfitting

d) Random Forest (RF)
   - Hyperparameters: n_estimators=100 (adaptive), random_state=42, n_jobs=-1
   - Scaling: No
   - Notes: Ensemble method, provides feature importances

e) XGBoost / Gradient Boosting
   - Hyperparameters: n_estimators=100 (adaptive), random_state=42, n_jobs=-1
   - Scaling: No
   - Notes: XGBoost preferred, falls back to sklearn GradientBoosting if unavailable
   - Hyperparameter tuning: Could use cross-validation for optimization

5. EXPERIMENTAL SETUP & EVALUATION
---------------------------------
a) Data Splitting:
   - Method: Train/Test split (80/20)
   - Stratification: Yes (preserves class distribution)
   - Random seed: 42 (for reproducibility)
   - No separate validation set (using test set for final evaluation)

b) Preprocessing Pipeline:
   - Feature extraction on full dataset (before split)
   - Scaling: Fit StandardScaler on training set only
   - Apply same scaler transform to test set (no data leakage)

c) Training:
   - Each model trained on X_train
   - Fixed random seeds for reproducibility
   - Models that require scaling use X_train_scaled

d) Evaluation Metrics:
   - Accuracy: Overall classification accuracy
   - Precision (macro): Average precision across all classes
   - Recall (macro): Average recall across all classes
   - F1-Score (macro): Harmonic mean of precision and recall
   - Per-class metrics: Precision, Recall, F1 for each class individually
   - Confusion Matrix: Visualized for Random Forest to analyze misclassifications

6. CODE STRUCTURE
---------------
Modular code organization:
- Data loading function (implicit in main flow)
- Feature engineering section (window aggregation, statistical features)
- Model training section (each classifier in separate block)
- Evaluation function (evaluate_model) - reusable for all models
- Results saving (CSV files, visualizations)
- Reproducibility: Fixed random_state=42 throughout

7. RESULTS
--------
Results table saved to: deliverable_5_2_outputs/results_table.csv
See output files for detailed performance metrics.
"""

import sys
import subprocess

# --- Auto-install missing packages ---
def install(package):
    subprocess.check_call([sys.executable, "-m", "pip", "install", package])

try:
    import pandas as pd
except ImportError:
    install("pandas")
    import pandas as pd

try:
    import numpy as np
except ImportError:
    install("numpy")
    import numpy as np

try:
    import matplotlib.pyplot as plt
except ImportError:
    install("matplotlib")
    import matplotlib.pyplot as plt

try:
    import seaborn as sns
except ImportError:
    install("seaborn")
    import seaborn as sns

try:
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import StandardScaler
    from sklearn.linear_model import LogisticRegression
    from sklearn.svm import SVC
    from sklearn.tree import DecisionTreeClassifier
    from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
    from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix, classification_report
except ImportError:
    install("scikit-learn")
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import StandardScaler
    from sklearn.linear_model import LogisticRegression
    from sklearn.svm import SVC
    from sklearn.tree import DecisionTreeClassifier
    from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
    from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix, classification_report

# Try XGBoost, fallback to GradientBoosting if not available
try:
    import xgboost as xgb
    XGBOOST_AVAILABLE = True
except ImportError:
    try:
        install("xgboost")
        import xgboost as xgb
        XGBOOST_AVAILABLE = True
    except:
        XGBOOST_AVAILABLE = False
        print("[WARN] XGBoost not available, using GradientBoosting instead")

# --- CONFIG ---
import os
from datetime import datetime

RANDOM_STATE = 42
# Use relative paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
INPUT_PATH = os.path.join(BASE_DIR, "data", "processed", "temperature_cleaned.csv")
WINDOW = '60min'
OUTPUT_DIR = os.path.join(BASE_DIR, "deliverable_5_2_outputs")
os.makedirs(OUTPUT_DIR, exist_ok=True)

print("=== Deliverable 5.2 Pipeline Started ===")
start_time = datetime.now()

# --- LOAD & SORT DATA ---
print("[LOAD] Loading data...")
df = pd.read_csv(INPUT_PATH)
print(f"[LOAD] Loaded {len(df)} rows")
df['timestamp'] = pd.to_datetime(df['timestamp'])
df = df.sort_values('timestamp').reset_index(drop=True)
df = df.set_index('timestamp')
print(f"[LOAD] Time span: {df.index.min()} -> {df.index.max()}")

# --- DATASET STATISTICS ---
print("\n=== DATASET STATISTICS ===")
print(f"Raw dataset shape: {df.shape[0]} rows x {df.shape[1]} columns")
print(f"Missing values: {df.isnull().sum().sum()}")
print(f"Temperature statistics:")
print(f"  Mean: {df['temperature'].mean():.2f}°C")
print(f"  Std: {df['temperature'].std():.2f}°C")
print(f"  Min: {df['temperature'].min():.2f}°C")
print(f"  Max: {df['temperature'].max():.2f}°C")
print(f"  Median: {df['temperature'].median():.2f}°C")

# --- FEATURE ENGINEERING: WINDOW STATS ---
print("[FEAT] Starting feature engineering...")
# Optimize: Do resample once and compute all stats together
resampled = df['temperature'].resample(WINDOW)
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
agg['temp_range'] = agg['temp_max'] - agg['temp_min']
agg['temp_diff'] = agg['temp_last'] - agg['temp_first']

print("[FEAT] Computing skew (this may take a moment)...")
try:
    agg['temp_skew'] = df['temperature'].groupby(pd.Grouper(freq=WINDOW)).skew()
    agg['temp_skew'] = agg['temp_skew'].fillna(0)  # Fill NaN from windows with <3 values
except Exception as e:
    print(f"[WARN] Skew calculation failed: {e}. Setting to 0.")
    agg['temp_skew'] = 0

# Temporal features
agg['hour'] = agg.index.hour
agg['hour_sin'] = np.sin(2*np.pi*agg['hour']/24)
agg['hour_cos'] = np.cos(2*np.pi*agg['hour']/24)
agg['weekday'] = agg.index.weekday
agg = agg.dropna(subset=['cnt']).fillna(0)
print(f"[FEAT] Feature engineering complete. Shape: {agg.shape}")

# --- FEATURE STATISTICS ---
print("\n=== FEATURE ENGINEERING SUMMARY ===")
print(f"Aggregated dataset shape: {agg.shape[0]} instances x {agg.shape[1]} features")
print(f"Window size: {WINDOW}")
print(f"Features created:")
feature_list = ['temp_mean', 'temp_std', 'temp_min', 'temp_max', 'temp_median', 
                'temp_range', 'temp_diff', 'temp_skew', 'hour_sin', 'hour_cos', 'weekday']
for feat in feature_list:
    if feat in agg.columns:
        print(f"  - {feat}")

# --- CREATE 3-CLASS LABELS ---
q25 = agg['temp_mean'].quantile(0.25)
q75 = agg['temp_mean'].quantile(0.75)
def label_fn(x):
    if x < q25:
        return 0
    elif x > q75:
        return 2
    else:
        return 1
agg['label'] = agg['temp_mean'].apply(label_fn)
label_counts = agg['label'].value_counts().sort_index()
print("\n=== TARGET VARIABLE SUMMARY ===")
print(f"Number of classes: 3")
print(f"Class definitions:")
print(f"  Class 0: Low temperature (< {q25:.2f}°C, 25th percentile)")
print(f"  Class 1: Medium temperature ({q25:.2f}°C - {q75:.2f}°C, 25th-75th percentile)")
print(f"  Class 2: High temperature (> {q75:.2f}°C, 75th percentile)")
print(f"\nLabel distribution:")
for label, count in label_counts.items():
    pct = (count / len(agg)) * 100
    print(f"  Class {label}: {count} instances ({pct:.2f}%)")
print(f"[LABEL] Total instances: {len(agg)}")

# Plot label distribution
plt.figure(figsize=(8, 5))
label_counts.plot(kind='bar', color=['skyblue', 'lightgreen', 'salmon'])
plt.title('Label Distribution (3 Classes)', fontsize=14, fontweight='bold')
plt.xlabel('Class Label', fontsize=12)
plt.ylabel('Count', fontsize=12)
plt.xticks(rotation=0)
plt.grid(axis='y', alpha=0.3)
for i, v in enumerate(label_counts.values):
    plt.text(i, v, str(v), ha='center', va='bottom', fontweight='bold')
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "label_distribution.png"), dpi=300, bbox_inches='tight')
plt.close()
print("[PLOT] Label distribution plot saved.")

# --- BUILD FEATURES & TARGET ---
X = agg.drop(columns=['label','cnt','temp_first','temp_last','hour'])
y = agg['label'].values

# --- TRAIN/TEST SPLIT ---
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=RANDOM_STATE)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
print(f"[SPLIT] Train: {X_train.shape}, Test: {X_test.shape}")

# --- EVALUATION HELPER ---
def evaluate_model(model, X_t, y_t, name, save_cm=False):
    y_pred = model.predict(X_t)
    acc = accuracy_score(y_t, y_pred)
    prec, rec, f1, _ = precision_recall_fscore_support(y_t, y_pred, average='macro', zero_division=0)
    prec_per_class, rec_per_class, f1_per_class, _ = precision_recall_fscore_support(y_t, y_pred, average=None, zero_division=0)
    cm = confusion_matrix(y_t, y_pred)
    print(f"\n[RESULT] {name} -> Acc={acc:.4f}, Prec={prec:.4f}, Rec={rec:.4f}, F1={f1:.4f}")
    print("Confusion Matrix:\n", cm)
    print("Per-class metrics:")
    for i in range(len(prec_per_class)):
        print(f"  Class {i}: Prec={prec_per_class[i]:.4f}, Rec={rec_per_class[i]:.4f}, F1={f1_per_class[i]:.4f}")
    
    # Plot confusion matrix
    if save_cm:
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=True, 
                    xticklabels=[f'Class {i}' for i in range(len(cm))],
                    yticklabels=[f'Class {i}' for i in range(len(cm))])
        plt.title(f'Confusion Matrix - {name}', fontsize=14, fontweight='bold')
        plt.ylabel('True Label', fontsize=12)
        plt.xlabel('Predicted Label', fontsize=12)
        plt.tight_layout()
        plt.savefig(os.path.join(OUTPUT_DIR, f"{name.lower().replace('-', '_')}_confusion_matrix.png"), 
                   dpi=300, bbox_inches='tight')
        plt.close()
        print(f"[PLOT] Confusion matrix saved for {name}")
    
    return dict(model=name, accuracy=acc, precision=prec, recall=rec, f1=f1,
                prec_per_class=prec_per_class, rec_per_class=rec_per_class, f1_per_class=f1_per_class)

# --- TRAIN CLASSICAL ML MODELS ---
results = []

# 1. Logistic Regression (LR)
# Hyperparameters: max_iter=1000, random_state=42
# Scaling: Required (using StandardScaler)
print("\n[1/5] Training Logistic Regression...")
lr = LogisticRegression(max_iter=1000, random_state=RANDOM_STATE)
lr.fit(X_train_scaled, y_train)
results.append(evaluate_model(lr, X_test_scaled, y_test, "LogisticRegression"))

# 2. Support Vector Machine (SVM)
# Hyperparameters: kernel='rbf', random_state=42, cache_size=500, max_iter=1000
# Scaling: Required (using StandardScaler)
# Note: RBF kernel for non-linear classification
print("\n[2/5] Training SVM-RBF (this may take time on large datasets)...")
svm = SVC(kernel='rbf', random_state=RANDOM_STATE, cache_size=500, max_iter=1000)
svm.fit(X_train_scaled, y_train)
results.append(evaluate_model(svm, X_test_scaled, y_test, "SVM-RBF"))

# 3. Decision Tree (DT)
# Hyperparameters: random_state=42 (default: criterion='gini', max_depth=None, etc.)
# Scaling: Not required (tree-based)
print("\n[3/5] Training Decision Tree...")
dt = DecisionTreeClassifier(random_state=RANDOM_STATE)
dt.fit(X_train, y_train)
results.append(evaluate_model(dt, X_test, y_test, "DecisionTree"))

# 4. Random Forest (RF)
# Hyperparameters: n_estimators=100 (adaptive), random_state=42, n_jobs=-1
# Scaling: Not required (tree-based ensemble)
# Note: Provides feature importances for analysis
n_samples = len(X_train)
n_estimators_rf = min(100, max(50, n_samples // 100))  # Adaptive based on dataset size
print(f"\n[4/5] Training Random Forest with {n_estimators_rf} estimators...")
rf = RandomForestClassifier(n_estimators=n_estimators_rf, random_state=RANDOM_STATE, n_jobs=-1, verbose=0)
rf.fit(X_train, y_train)
rf_res = evaluate_model(rf, X_test, y_test, "RandomForest", save_cm=True)
results.append(rf_res)
# Save feature importances
importances = pd.Series(rf.feature_importances_, index=X.columns).sort_values(ascending=False)
importances.to_csv(os.path.join(OUTPUT_DIR,"rf_feature_importances.csv"))
print("[FEAT] Random Forest importances saved.")

# 5. XGBoost / Gradient Boosting
# Hyperparameters: n_estimators=100 (adaptive), random_state=42, n_jobs=-1
# Scaling: Not required (tree-based)
# Note: XGBoost preferred, falls back to sklearn GradientBoosting if unavailable
n_samples = len(X_train)
n_estimators_gb = min(100, max(50, n_samples // 100))  # Adaptive based on dataset size
if XGBOOST_AVAILABLE:
    print(f"\n[5/5] Training XGBoost with {n_estimators_gb} estimators...")
    xgb_model = xgb.XGBClassifier(n_estimators=n_estimators_gb, random_state=RANDOM_STATE, 
                                   n_jobs=-1, eval_metric='mlogloss', verbosity=0)
    xgb_model.fit(X_train, y_train)
    results.append(evaluate_model(xgb_model, X_test, y_test, "XGBoost"))
else:
    print(f"\n[5/5] Training GradientBoosting with {n_estimators_gb} estimators...")
    gb = GradientBoostingClassifier(n_estimators=n_estimators_gb, random_state=RANDOM_STATE, verbose=0)
    gb.fit(X_train, y_train)
    results.append(evaluate_model(gb, X_test, y_test, "GradientBoosting"))

# --- SAVE RESULTS TABLE ---
results_df = pd.DataFrame([{'Model': r['model'], 'Accuracy': r['accuracy'],
                            'Precision_macro': r['precision'], 'Recall_macro': r['recall'],
                            'F1_macro': r['f1']} for r in results]).set_index('Model')
results_df.to_csv(os.path.join(OUTPUT_DIR,"results_table.csv"))
print("\n[FINAL RESULTS]\n", results_df)

# --- GENERATE SUMMARY REPORT ---
print("\n" + "="*80)
print("DELIVERABLE 5.2 EXECUTION SUMMARY")
print("="*80)
print(f"Dataset: Beehive Temperature Sensor Data")
print(f"Raw instances: {len(df):,}")
print(f"Feature-engineered instances: {len(agg):,}")
print(f"Features: {X.shape[1]}")
print(f"Train/Test split: {X_train.shape[0]}/{X_test.shape[0]} (80/20)")
print(f"Models trained: 5 (LR, SVM, DT, RF, XGBoost/GB)")
print(f"\nOutput files saved to: {OUTPUT_DIR}")
print("  - results_table.csv")
print("  - rf_feature_importances.csv")
print("  - label_distribution.png")
print("  - rf_confusion_matrix.png")
print("="*80)
print("Elapsed time:", datetime.now()-start_time)
print("="*80)
