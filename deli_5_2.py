
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
    from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix
except ImportError:
    install("scikit-learn")
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import StandardScaler
    from sklearn.linear_model import LogisticRegression
    from sklearn.svm import SVC
    from sklearn.tree import DecisionTreeClassifier
    from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
    from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix

# --- CONFIG ---
import os
from datetime import datetime

RANDOM_STATE = 42
INPUT_PATH = r"D:\SoundHive-AI\data\processed\temperature_cleaned.csv"
WINDOW = '60min'
OUTPUT_DIR = r"D:\SoundHive-AI\deliverable_5_2_outputs"
os.makedirs(OUTPUT_DIR, exist_ok=True)

print("=== Deliverable 5.2 Pipeline Started ===")
start_time = datetime.now()

# --- LOAD & SORT DATA ---
df = pd.read_csv(INPUT_PATH)
df['timestamp'] = pd.to_datetime(df['timestamp'])
df = df.sort_values('timestamp').reset_index(drop=True)
df = df.set_index('timestamp')
print(f"[LOAD] {len(df)} rows. Time span: {df.index.min()} -> {df.index.max()}")

# --- FEATURE ENGINEERING: WINDOW STATS ---
agg = df['temperature'].resample(WINDOW).agg(['count','mean','std','min','max','median'])
agg = agg.rename(columns={'count':'cnt','mean':'temp_mean','std':'temp_std','min':'temp_min',
                          'max':'temp_max','median':'temp_median'})
agg['temp_range'] = agg['temp_max'] - agg['temp_min']
agg['temp_first'] = df['temperature'].resample(WINDOW).first()
agg['temp_last'] = df['temperature'].resample(WINDOW).last()
agg['temp_diff'] = agg['temp_last'] - agg['temp_first']
agg['temp_skew'] = df['temperature'].resample(WINDOW).apply(lambda x: x.skew())
agg['hour'] = agg.index.hour
agg['hour_sin'] = np.sin(2*np.pi*agg['hour']/24)
agg['hour_cos'] = np.cos(2*np.pi*agg['hour']/24)
agg['weekday'] = agg.index.weekday
agg = agg.dropna(subset=['cnt']).fillna(0)

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
print("[LABEL] 3-class label distribution:\n", agg['label'].value_counts().sort_index())

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
def evaluate_model(model, X_t, y_t, name):
    y_pred = model.predict(X_t)
    acc = accuracy_score(y_t, y_pred)
    prec, rec, f1, _ = precision_recall_fscore_support(y_t, y_pred, average='macro', zero_division=0)
    cm = confusion_matrix(y_t, y_pred)
    print(f"\n[RESULT] {name} -> Acc={acc:.4f}, Prec={prec:.4f}, Rec={rec:.4f}, F1={f1:.4f}")
    print("Confusion Matrix:\n", cm)
    return dict(model=name, accuracy=acc, precision=prec, recall=rec, f1=f1)

# --- TRAIN CLASSICAL ML MODELS ---
results = []

# Logistic Regression
lr = LogisticRegression(max_iter=1000, random_state=RANDOM_STATE)
lr.fit(X_train_scaled, y_train)
results.append(evaluate_model(lr, X_test_scaled, y_test, "LogisticRegression"))

# SVM RBF
svm = SVC(kernel='rbf', random_state=RANDOM_STATE)
svm.fit(X_train_scaled, y_train)
results.append(evaluate_model(svm, X_test_scaled, y_test, "SVM-RBF"))

# Decision Tree
dt = DecisionTreeClassifier(random_state=RANDOM_STATE)
dt.fit(X_train, y_train)
results.append(evaluate_model(dt, X_test, y_test, "DecisionTree"))

# Random Forest
rf = RandomForestClassifier(n_estimators=100, random_state=RANDOM_STATE, n_jobs=-1)
rf.fit(X_train, y_train)
rf_res = evaluate_model(rf, X_test, y_test, "RandomForest")
results.append(rf_res)
# Save feature importances
importances = pd.Series(rf.feature_importances_, index=X.columns).sort_values(ascending=False)
importances.to_csv(os.path.join(OUTPUT_DIR,"rf_feature_importances.csv"))
print("[FEAT] Random Forest importances saved.")

# Gradient Boosting
gb = GradientBoostingClassifier(n_estimators=100, random_state=RANDOM_STATE)
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
