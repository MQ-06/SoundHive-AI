"""Bootstrap model artifacts on first run."""

from __future__ import annotations

import json
import os
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, precision_recall_fscore_support
from sklearn.model_selection import train_test_split

from src.data.preprocess import preprocess_timeseries
from src.features.engineering import (
    DEFAULT_HIGH_THRESHOLD,
    DEFAULT_LOW_THRESHOLD,
    create_labels,
    extract_time_series_features,
    prepare_features_and_labels,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
MODEL_DIR = PROJECT_ROOT / "models" / "saved_models"
RESULTS_DIR = PROJECT_ROOT / "results"
MODEL_PATH = MODEL_DIR / "rf_model.joblib"
CURRENT_MODEL_VERSION = 1


def _generate_temperature_data(n_hours: int = 8760, seed: int = 42) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    timestamps = pd.date_range(start="2017-01-01", periods=n_hours, freq="h")
    seasonal = 22 + 8 * np.sin(2 * np.pi * (timestamps.dayofyear - 80) / 365)
    daily = 3 * np.sin(2 * np.pi * (timestamps.hour - 6) / 24)
    noise = rng.normal(0, 1.2, n_hours)
    temperature = np.clip(seasonal + daily + noise, 8, 42)
    return pd.DataFrame({"timestamp": timestamps, "temperature": temperature})


def _needs_retrain() -> bool:
    if not MODEL_PATH.exists():
        return True
    meta_path = MODEL_DIR / "label_metadata.json"
    if not meta_path.exists():
        return True
    with open(meta_path, encoding="utf-8") as f:
        return json.load(f).get("model_version") != CURRENT_MODEL_VERSION


def ensure_artifacts(verbose: bool = False) -> dict:
    if not _needs_retrain():
        return {}

    if MODEL_PATH.exists():
        MODEL_PATH.unlink()

    os.environ.setdefault("OMP_NUM_THREADS", "1")
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    (RESULTS_DIR / "figures").mkdir(parents=True, exist_ok=True)
    (RESULTS_DIR / "tables").mkdir(parents=True, exist_ok=True)
    (PROJECT_ROOT / "data" / "demo").mkdir(parents=True, exist_ok=True)

    demo_path = PROJECT_ROOT / "data" / "demo" / "sample_temperature.csv"
    if not demo_path.exists():
        _generate_temperature_data(n_hours=336, seed=99).to_csv(demo_path, index=False)

    raw_df = _generate_temperature_data(n_hours=8760)
    processed = preprocess_timeseries(raw_df, verbose=False).set_index("timestamp")
    features_df = extract_time_series_features(processed, verbose=False)
    labels = create_labels(features_df, method="fixed", verbose=False)
    features_df["label"] = labels

    X, y = prepare_features_and_labels(features_df)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )

    rf = RandomForestClassifier(
        n_estimators=100, max_depth=12, min_samples_leaf=5, random_state=42, n_jobs=1
    )
    rf.fit(X_train, y_train)
    y_pred = rf.predict(X_test)

    acc = accuracy_score(y_test, y_pred)
    prec, rec, f1, _ = precision_recall_fscore_support(y_test, y_pred, average="macro", zero_division=0)
    cm = confusion_matrix(y_test, y_pred)

    joblib.dump(rf, MODEL_PATH)
    with open(MODEL_DIR / "feature_columns.json", "w", encoding="utf-8") as f:
        json.dump(X.columns.tolist(), f, indent=2)
    with open(MODEL_DIR / "label_metadata.json", "w", encoding="utf-8") as f:
        json.dump({
            "model_version": CURRENT_MODEL_VERSION,
            "n_classes": 3,
            "method": "fixed",
            "low_threshold": DEFAULT_LOW_THRESHOLD,
            "high_threshold": DEFAULT_HIGH_THRESHOLD,
            "class_names": {
                "0": "Low Temperature",
                "1": "Normal Temperature",
                "2": "High Temperature",
            },
        }, f, indent=2)

    pd.DataFrame([{"Model": "Random Forest", "Accuracy": acc, "Precision": prec, "Recall": rec, "F1_Score": f1}]
    ).to_csv(RESULTS_DIR / "tables" / "classical_ml_results.csv", index=False)

    importances = pd.Series(rf.feature_importances_, index=X.columns).sort_values(ascending=False)
    importances.to_csv(RESULTS_DIR / "tables" / "rf_feature_importances.csv")

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import seaborn as sns

        fig, ax = plt.subplots(figsize=(6, 5))
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax,
                    xticklabels=["Low", "Normal", "High"], yticklabels=["Low", "Normal", "High"])
        ax.set_title("Confusion Matrix - Random Forest")
        fig.tight_layout()
        fig.savefig(RESULTS_DIR / "figures" / "rf_confusion_matrix.png", dpi=150)
        plt.close(fig)

        fig2, ax2 = plt.subplots(figsize=(8, 5))
        importances.head(10).plot(kind="barh", ax=ax2, color="steelblue")
        ax2.invert_yaxis()
        fig2.tight_layout()
        fig2.savefig(RESULTS_DIR / "figures" / "rf_feature_importances.png", dpi=150)
        plt.close(fig2)
    except Exception:
        pass

    return {"accuracy": acc}
