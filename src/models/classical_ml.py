"""Classical machine learning models for classification"""

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    confusion_matrix,
    classification_report
)
from typing import Dict, Tuple, Optional
import warnings

# Try to import XGBoost
try:
    import xgboost as xgb
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False


def evaluate_model(
    model,
    X_test: np.ndarray,
    y_test: np.ndarray,
    model_name: str = "Model",
    verbose: bool = True
) -> Dict:
    """
    Evaluate a trained model and return metrics.
    
    Args:
        model: Trained model
        X_test: Test features
        y_test: Test labels
        model_name: Name of the model
        verbose: Whether to print results
        
    Returns:
        Dictionary with evaluation metrics
    """
    y_pred = model.predict(X_test)
    
    acc = accuracy_score(y_test, y_pred)
    prec, rec, f1, _ = precision_recall_fscore_support(
        y_test, y_pred, average='macro', zero_division=0
    )
    prec_per_class, rec_per_class, f1_per_class, _ = precision_recall_fscore_support(
        y_test, y_pred, average=None, zero_division=0
    )
    cm = confusion_matrix(y_test, y_pred)
    
    if verbose:
        print(f"\n[RESULT] {model_name}")
        print(f"  Accuracy: {acc:.4f}")
        print(f"  Precision (macro): {prec:.4f}")
        print(f"  Recall (macro): {rec:.4f}")
        print(f"  F1-Score (macro): {f1:.4f}")
        print(f"\n  Confusion Matrix:\n{cm}")
    
    return {
        'model': model_name,
        'accuracy': acc,
        'precision': prec,
        'recall': rec,
        'f1': f1,
        'precision_per_class': prec_per_class,
        'recall_per_class': rec_per_class,
        'f1_per_class': f1_per_class,
        'confusion_matrix': cm
    }


def train_logistic_regression(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    random_state: int = 42,
    max_iter: int = 1000,
    use_scaling: bool = True
) -> Tuple[LogisticRegression, Dict]:
    """Train Logistic Regression model."""
    if use_scaling:
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
    else:
        X_train_scaled = X_train
        X_test_scaled = X_test
    
    model = LogisticRegression(max_iter=max_iter, random_state=random_state)
    model.fit(X_train_scaled, y_train)
    
    results = evaluate_model(model, X_test_scaled, y_test, "Logistic Regression")
    
    return model, results


def train_svm(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    kernel: str = 'rbf',
    random_state: int = 42,
    cache_size: int = 500,
    max_iter: int = 1000,
    use_scaling: bool = True
) -> Tuple[SVC, Dict]:
    """Train Support Vector Machine model."""
    if use_scaling:
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
    else:
        X_train_scaled = X_train
        X_test_scaled = X_test
    
    model = SVC(kernel=kernel, random_state=random_state, cache_size=cache_size, max_iter=max_iter)
    model.fit(X_train_scaled, y_train)
    
    results = evaluate_model(model, X_test_scaled, y_test, f"SVM-{kernel.upper()}")
    
    return model, results


def train_decision_tree(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    random_state: int = 42
) -> Tuple[DecisionTreeClassifier, Dict]:
    """Train Decision Tree model."""
    model = DecisionTreeClassifier(random_state=random_state)
    model.fit(X_train, y_train)
    
    results = evaluate_model(model, X_test, y_test, "Decision Tree")
    
    return model, results


def train_random_forest(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    n_estimators: int = 100,
    random_state: int = 42,
    n_jobs: int = -1
) -> Tuple[RandomForestClassifier, Dict]:
    """Train Random Forest model."""
    # Adaptive n_estimators based on dataset size
    n_samples = len(X_train)
    n_est = min(n_estimators, max(50, n_samples // 100))
    
    model = RandomForestClassifier(
        n_estimators=n_est,
        random_state=random_state,
        n_jobs=n_jobs,
        verbose=0
    )
    model.fit(X_train, y_train)
    
    results = evaluate_model(model, X_test, y_test, "Random Forest")
    
    return model, results


def train_xgboost(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    n_estimators: int = 100,
    random_state: int = 42,
    n_jobs: int = -1
) -> Tuple[object, Dict]:
    """Train XGBoost or Gradient Boosting model."""
    n_samples = len(X_train)
    n_est = min(n_estimators, max(50, n_samples // 100))
    
    if XGBOOST_AVAILABLE:
        model = xgb.XGBClassifier(
            n_estimators=n_est,
            random_state=random_state,
            n_jobs=n_jobs,
            eval_metric='mlogloss',
            verbosity=0
        )
        model_name = "XGBoost"
    else:
        model = GradientBoostingClassifier(
            n_estimators=n_est,
            random_state=random_state,
            verbose=0
        )
        model_name = "Gradient Boosting"
    
    model.fit(X_train, y_train)
    results = evaluate_model(model, X_test, y_test, model_name)
    
    return model, results

