"""Inference utilities for the SoundHive-AI demo app."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Tuple

import joblib
import pandas as pd

from src.data.preprocess import preprocess_timeseries
from src.features.engineering import (
    create_labels,
    extract_time_series_features,
    prepare_features_and_labels,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
MODEL_DIR = PROJECT_ROOT / "models" / "saved_models"


def load_artifacts() -> Tuple[object, List[str], Dict]:
    from src.bootstrap import ensure_artifacts
    ensure_artifacts()
    model = joblib.load(MODEL_DIR / "rf_model.joblib")
    with open(MODEL_DIR / "feature_columns.json", encoding="utf-8") as f:
        feature_columns = json.load(f)
    with open(MODEL_DIR / "label_metadata.json", encoding="utf-8") as f:
        label_metadata = json.load(f)
    return model, feature_columns, label_metadata


def run_inference_pipeline(df: pd.DataFrame, window_size: str = "60min"):
    processed = preprocess_timeseries(df, verbose=False).set_index("timestamp")
    features_df = extract_time_series_features(processed, verbose=False)

    model, feature_columns, label_metadata = load_artifacts()
    labels = create_labels(
        features_df,
        method=label_metadata.get("method", "fixed"),
        low_threshold=label_metadata.get("low_threshold", 18.0),
        high_threshold=label_metadata.get("high_threshold", 32.0),
        verbose=False,
    )
    features_df["label"] = labels
    X, y = prepare_features_and_labels(features_df)
    X = X[feature_columns]

    predictions = model.predict(X.values)
    probabilities = model.predict_proba(X.values)
    class_names = label_metadata.get("class_names", {})

    pred_df = features_df.copy()
    pred_df["predicted_label"] = predictions
    pred_df["confidence"] = probabilities.max(axis=1)
    for idx in range(probabilities.shape[1]):
        name = class_names.get(str(idx), f"Class {idx}")
        pred_df[f"prob_{name}"] = probabilities[:, idx]
    pred_df["predicted_class"] = pred_df["predicted_label"].map(
        lambda x: class_names.get(str(int(x)), f"Class {int(x)}")
    )
    return features_df, y, pred_df


def health_summary(pred_df: pd.DataFrame) -> Dict[str, object]:
    counts = pred_df["predicted_class"].value_counts()
    conf = pred_df["confidence"]
    return {
        "dominant_health": counts.idxmax(),
        "avg_confidence": float(conf.mean()),
        "min_confidence": float(conf.min()),
        "max_confidence": float(conf.max()),
        "low_confidence_windows": int((conf < 0.7).sum()),
        "distribution": counts.to_dict(),
        "n_windows": len(pred_df),
    }
