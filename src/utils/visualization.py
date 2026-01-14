"""Visualization utilities for model results"""

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
import os
from typing import Optional


def plot_label_distribution(
    labels: pd.Series,
    save_path: Optional[str] = None,
    figsize: tuple = (8, 5)
) -> None:
    """Plot distribution of class labels."""
    label_counts = labels.value_counts().sort_index()
    
    plt.figure(figsize=figsize)
    label_counts.plot(kind='bar', color=['skyblue', 'lightgreen', 'salmon', 'orange'])
    plt.title('Label Distribution', fontsize=14, fontweight='bold')
    plt.xlabel('Class Label', fontsize=12)
    plt.ylabel('Count', fontsize=12)
    plt.xticks(rotation=0)
    plt.grid(axis='y', alpha=0.3)
    
    for i, v in enumerate(label_counts.values):
        plt.text(i, v, str(v), ha='center', va='bottom', fontweight='bold')
    
    plt.tight_layout()
    
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
    else:
        plt.show()


def plot_confusion_matrix(
    cm: np.ndarray,
    class_names: Optional[list] = None,
    model_name: str = "Model",
    save_path: Optional[str] = None,
    figsize: tuple = (8, 6)
) -> None:
    """Plot confusion matrix."""
    plt.figure(figsize=figsize)
    
    if class_names is None:
        class_names = [f'Class {i}' for i in range(len(cm))]
    
    sns.heatmap(
        cm,
        annot=True,
        fmt='d',
        cmap='Blues',
        cbar=True,
        xticklabels=class_names,
        yticklabels=class_names
    )
    
    plt.title(f'Confusion Matrix - {model_name}', fontsize=14, fontweight='bold')
    plt.ylabel('True Label', fontsize=12)
    plt.xlabel('Predicted Label', fontsize=12)
    plt.tight_layout()
    
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
    else:
        plt.show()


def plot_training_history(
    history: dict,
    save_path: Optional[str] = None,
    figsize: tuple = (12, 4)
) -> None:
    """Plot training history for deep learning models."""
    fig, axes = plt.subplots(1, 2, figsize=figsize)
    
    # Plot accuracy
    axes[0].plot(history['accuracy'], label='Training Accuracy')
    if 'val_accuracy' in history:
        axes[0].plot(history['val_accuracy'], label='Validation Accuracy')
    axes[0].set_title('Model Accuracy', fontweight='bold')
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Accuracy')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # Plot loss
    axes[1].plot(history['loss'], label='Training Loss')
    if 'val_loss' in history:
        axes[1].plot(history['val_loss'], label='Validation Loss')
    axes[1].set_title('Model Loss', fontweight='bold')
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Loss')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
    else:
        plt.show()


def plot_feature_importances(
    importances: pd.Series,
    top_n: int = 10,
    save_path: Optional[str] = None,
    figsize: tuple = (10, 6)
) -> None:
    """Plot feature importances."""
    top_features = importances.head(top_n)
    
    plt.figure(figsize=figsize)
    top_features.plot(kind='barh', color='steelblue')
    plt.title(f'Top {top_n} Feature Importances', fontsize=14, fontweight='bold')
    plt.xlabel('Importance', fontsize=12)
    plt.ylabel('Feature', fontsize=12)
    plt.gca().invert_yaxis()
    plt.grid(axis='x', alpha=0.3)
    plt.tight_layout()
    
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
    else:
        plt.show()

