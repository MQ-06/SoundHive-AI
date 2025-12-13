

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
print("[LABEL] 3-class label distribution:\n", label_counts)

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

# Logistic Regression
lr = LogisticRegression(max_iter=1000, random_state=RANDOM_STATE)
lr.fit(X_train_scaled, y_train)
results.append(evaluate_model(lr, X_test_scaled, y_test, "LogisticRegression"))

# SVM RBF (optimized for speed)
print("\n[INFO] Training SVM (this may take time on large datasets)...")
svm = SVC(kernel='rbf', random_state=RANDOM_STATE, cache_size=500, max_iter=1000)
svm.fit(X_train_scaled, y_train)
results.append(evaluate_model(svm, X_test_scaled, y_test, "SVM-RBF"))

# Decision Tree
dt = DecisionTreeClassifier(random_state=RANDOM_STATE)
dt.fit(X_train, y_train)
results.append(evaluate_model(dt, X_test, y_test, "DecisionTree"))

# Random Forest (optimized: reduce estimators if dataset is large)
n_samples = len(X_train)
n_estimators_rf = min(100, max(50, n_samples // 100))  # Adaptive based on dataset size
print(f"\n[INFO] Training Random Forest with {n_estimators_rf} estimators...")
rf = RandomForestClassifier(n_estimators=n_estimators_rf, random_state=RANDOM_STATE, n_jobs=-1, verbose=0)
rf.fit(X_train, y_train)
rf_res = evaluate_model(rf, X_test, y_test, "RandomForest", save_cm=True)
results.append(rf_res)
# Save feature importances
importances = pd.Series(rf.feature_importances_, index=X.columns).sort_values(ascending=False)
importances.to_csv(os.path.join(OUTPUT_DIR,"rf_feature_importances.csv"))
print("[FEAT] Random Forest importances saved.")

# XGBoost (or GradientBoosting fallback) - optimized
n_samples = len(X_train)
n_estimators_gb = min(100, max(50, n_samples // 100))  # Adaptive based on dataset size
if XGBOOST_AVAILABLE:
    print(f"\n[INFO] Training XGBoost with {n_estimators_gb} estimators...")
    xgb_model = xgb.XGBClassifier(n_estimators=n_estimators_gb, random_state=RANDOM_STATE, 
                                   n_jobs=-1, eval_metric='mlogloss', verbosity=0)
    xgb_model.fit(X_train, y_train)
    results.append(evaluate_model(xgb_model, X_test, y_test, "XGBoost"))
else:
    print(f"\n[INFO] Training GradientBoosting with {n_estimators_gb} estimators...")
    gb = GradientBoostingClassifier(n_estimators=n_estimators_gb, random_state=RANDOM_STATE, verbose=0)
    gb.fit(X_train, y_train)
    results.append(evaluate_model(gb, X_test, y_test, "GradientBoosting"))

# --- SAVE RESULTS TABLE ---
results_df = pd.DataFrame([{'Model': r['model'], 'Accuracy': r['accuracy'],
                            'Precision_macro': r['precision'], 'Recall_macro': r['recall'],
                            'F1_macro': r['f1']} for r in results]).set_index('Model')
results_df.to_csv(os.path.join(OUTPUT_DIR,"results_table.csv"))
print("\n[FINAL RESULTS]\n", results_df)

print(f"\nAll outputs saved to: {OUTPUT_DIR}")
print("Elapsed time:", datetime.now()-start_time)
