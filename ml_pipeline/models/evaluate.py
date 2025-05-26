#!/usr/bin/env python3
"""
Script for evaluating time series classification models.
"""

import os
import sys
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report, roc_curve, auc
)
from typing import Dict, List, Tuple, Optional, Union, Any
import json
import seaborn as sns

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from models.model import load_model_with_metadata
from data_processing.data_processing import load_dataset, create_windows

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description='Evaluate time series classification model.',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    parser.add_argument(
        '--model', 
        type=str,
        required=True,
        help='Path to trained model file (.h5)'
    )
    
    parser.add_argument(
        '--data', 
        type=str,
        required=True,
        help='Path to test data file (.npz or .csv)'
    )
    
    parser.add_argument(
        '--output-dir', 
        type=str,
        default='evaluation',
        help='Directory to save evaluation results'
    )
    
    parser.add_argument(
        '--batch-size', 
        type=int,
        default=32,
        help='Batch size for evaluation'
    )
    
    parser.add_argument(
        '--window-length', 
        type=int,
        help='Window length (required if data is CSV)'
    )
    
    parser.add_argument(
        '--overlap', 
        type=float,
        default=0.5,
        help='Window overlap (used if data is CSV)'
    )
    
    parser.add_argument(
        '--features', 
        type=str,
        help='Comma-separated list of feature columns (required if data is CSV)'
    )
    
    parser.add_argument(
        '--label', 
        type=str,
        help='Label column name (required if data is CSV)'
    )
    
    parser.add_argument(
        '--class-names', 
        type=str,
        help='Comma-separated list of class names'
    )
    
    parser.add_argument(
        '--threshold', 
        type=float,
        default=0.5,
        help='Classification threshold for binary models'
    )
    
    return parser.parse_args()

def load_test_data(
    data_path: str,
    window_length: Optional[int] = None,
    overlap: float = 0.5,
    feature_columns: Optional[List[str]] = None,
    label_column: Optional[str] = None
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Load test data from file.
    
    Args:
        data_path: Path to test data file
        window_length: Window length (required if data is CSV)
        overlap: Window overlap (used if data is CSV)
        feature_columns: List of feature columns (required if data is CSV)
        label_column: Label column name (required if data is CSV)
    
    Returns:
        x_test: Test features
        y_test: Test labels
    """
    if data_path.endswith('.npz'):
        # Load processed data
        windows, labels, _ = load_dataset(data_path)
        return windows, labels
    
    elif data_path.endswith('.csv'):
        # Check required parameters
        if window_length is None:
            raise ValueError("Window length is required for CSV data")
        if feature_columns is None:
            raise ValueError("Feature columns are required for CSV data")
        if label_column is None:
            raise ValueError("Label column is required for CSV data")
        
        # Load raw data
        data = pd.read_csv(data_path)
        
        # Convert feature_columns from string to list if needed
        if isinstance(feature_columns, str):
            feature_columns = feature_columns.split(',')
        
        # Create windows
        windows, labels, _ = create_windows(
            data,
            window_length=window_length,
            overlap=overlap,
            feature_columns=feature_columns,
            label_column=label_column
        )
        
        return windows, labels
    
    else:
        raise ValueError("Unsupported data format. Use .npz or .csv")

def evaluate_model(
    model: tf.keras.Model,
    x_test: np.ndarray,
    y_test: np.ndarray,
    batch_size: int = 32,
    threshold: float = 0.5,
    class_names: Optional[List[str]] = None
) -> Dict[str, Any]:
    """
    Evaluate model performance.
    
    Args:
        model: Trained model
        x_test: Test features
        y_test: Test labels
        batch_size: Batch size for evaluation
        threshold: Classification threshold for binary models
        class_names: List of class names
    
    Returns:
        Dictionary containing evaluation metrics
    """
    # Get model predictions
    y_pred_proba = model.predict(x_test, batch_size=batch_size)
    
    # Convert probabilities to class predictions
    if y_pred_proba.shape[1] > 1:
        # Multi-class
        y_pred = np.argmax(y_pred_proba, axis=1)
    else:
        # Binary
        y_pred = (y_pred_proba > threshold).astype(int).flatten()
    
    # Calculate metrics
    metrics = {}
    metrics['accuracy'] = float(accuracy_score(y_test, y_pred))
    
    # For binary classification
    if len(np.unique(y_test)) == 2:
        metrics['precision'] = float(precision_score(y_test, y_pred))
        metrics['recall'] = float(recall_score(y_test, y_pred))
        metrics['f1'] = float(f1_score(y_test, y_pred))
        
        # ROC curve
        fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
        metrics['roc_auc'] = float(auc(fpr, tpr))
        metrics['roc_curve'] = {
            'fpr': fpr.tolist(),
            'tpr': tpr.tolist()
        }
    
    # For multi-class classification
    else:
        metrics['precision'] = float(precision_score(y_test, y_pred, average='weighted'))
        metrics['recall'] = float(recall_score(y_test, y_pred, average='weighted'))
        metrics['f1'] = float(f1_score(y_test, y_pred, average='weighted'))
        
        # Per-class metrics
        metrics['per_class'] = {}
        for i in range(len(np.unique(y_test))):
            class_name = class_names[i] if class_names and i < len(class_names) else f"Class {i}"
            metrics['per_class'][class_name] = {
                'precision': float(precision_score(y_test, y_pred, average=None)[i]),
                'recall': float(recall_score(y_test, y_pred, average=None)[i]),
                'f1': float(f1_score(y_test, y_pred, average=None)[i])
            }
    
    # Confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    metrics['confusion_matrix'] = cm.tolist()
    
    # Classification report
    report = classification_report(y_test, y_pred, output_dict=True)
    metrics['classification_report'] = report
    
    return metrics, y_pred, y_pred_proba

def plot_confusion_matrix(
    cm: np.ndarray,
    class_names: List[str],
    output_path: str
) -> None:
    """
    Plot confusion matrix.
    
    Args:
        cm: Confusion matrix
        class_names: List of class names
        output_path: Path to save the plot
    """
    plt.figure(figsize=(10, 8))
    sns.heatmap(
        cm,
        annot=True,
        fmt='d',
        cmap='Blues',
        xticklabels=class_names,
        yticklabels=class_names
    )
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion Matrix')
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()

def plot_roc_curve(
    fpr: np.ndarray,
    tpr: np.ndarray,
    roc_auc: float,
    output_path: str
) -> None:
    """
    Plot ROC curve.
    
    Args:
        fpr: False positive rates
        tpr: True positive rates
        roc_auc: Area under ROC curve
        output_path: Path to save the plot
    """
    plt.figure(figsize=(10, 8))
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic')
    plt.legend(loc="lower right")
    plt.grid(True)
    plt.savefig(output_path)
    plt.close()

def plot_class_distribution(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    class_names: List[str],
    output_path: str
) -> None:
    """
    Plot class distribution.
    
    Args:
        y_true: True labels
        y_pred: Predicted labels
        class_names: List of class names
        output_path: Path to save the plot
    """
    plt.figure(figsize=(12, 6))
    
    # Count occurrences
    true_counts = np.bincount(y_true.astype(int), minlength=len(class_names))
    pred_counts = np.bincount(y_pred.astype(int), minlength=len(class_names))
    
    # Plot
    x = np.arange(len(class_names))
    width = 0.35
    
    plt.bar(x - width/2, true_counts, width, label='True')
    plt.bar(x + width/2, pred_counts, width, label='Predicted')
    
    plt.xlabel('Class')
    plt.ylabel('Count')
    plt.title('Class Distribution')
    plt.xticks(x, class_names)
    plt.legend()
    plt.grid(True, axis='y')
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()

def plot_misclassified_examples(
    x_test: np.ndarray,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    class_names: List[str],
    output_dir: str,
    max_examples: int = 10
) -> None:
    """
    Plot misclassified examples.
    
    Args:
        x_test: Test features
        y_true: True labels
        y_pred: Predicted labels
        class_names: List of class names
        output_dir: Directory to save plots
        max_examples: Maximum number of examples to plot
    """
    # Find misclassified examples
    misclassified = np.where(y_true != y_pred)[0]
    
    if len(misclassified) == 0:
        print("No misclassified examples found.")
        return
    
    # Limit number of examples
    n_examples = min(max_examples, len(misclassified))
    indices = np.random.choice(misclassified, n_examples, replace=False)
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Plot each example
    for i, idx in enumerate(indices):
        plt.figure(figsize=(10, 6))
        
        # Plot each feature
        for j in range(x_test.shape[2]):
            plt.plot(x_test[idx, :, j], label=f'Feature {j}')
        
        true_class = int(y_true[idx])
        pred_class = int(y_pred[idx])
        
        true_name = class_names[true_class] if class_names and true_class < len(class_names) else f"Class {true_class}"
        pred_name = class_names[pred_class] if class_names and pred_class < len(class_names) else f"Class {pred_class}"
        
        plt.title(f'Misclassified Example {i+1}\nTrue: {true_name}, Predicted: {pred_name}')
        plt.xlabel('Time')
        plt.ylabel('Value')
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f'misclassified_{i+1}.png'))
        plt.close()

def main():
    """Main function."""
    # Parse arguments
    args = parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Load model
    print(f"Loading model from {args.model}...")
    model, metadata = load_model_with_metadata(args.model)
    
    # Get class names
    if args.class_names:
        class_names = args.class_names.split(',')
    elif 'class_names' in metadata and metadata['class_names']:
        class_names = metadata['class_names']
    else:
        num_classes = metadata.get('num_classes', 2)
        class_names = [f"Class {i}" for i in range(num_classes)]
    
    print(f"Classes: {class_names}")
    
    # Load test data
    print(f"Loading test data from {args.data}...")
    
    # Determine feature columns and label column
    feature_columns = args.features
    if feature_columns is None and 'feature_names' in metadata:
        feature_columns = metadata['feature_names']
    
    x_test, y_test = load_test_data(
        args.data,
        window_length=args.window_length,
        overlap=args.overlap,
        feature_columns=feature_columns,
        label_column=args.label
    )
    
    print(f"Test data shape: {x_test.shape}")
    
    # Evaluate model
    print("Evaluating model...")
    metrics, y_pred, y_pred_proba = evaluate_model(
        model=model,
        x_test=x_test,
        y_test=y_test,
        batch_size=args.batch_size,
        threshold=args.threshold,
        class_names=class_names
    )
    
    # Print metrics
    print("\nEvaluation Metrics:")
    print(f"Accuracy: {metrics['accuracy']:.4f}")
    print(f"Precision: {metrics['precision']:.4f}")
    print(f"Recall: {metrics['recall']:.4f}")
    print(f"F1 Score: {metrics['f1']:.4f}")
    
    if 'roc_auc' in metrics:
        print(f"ROC AUC: {metrics['roc_auc']:.4f}")
    
    # Save metrics to JSON
    metrics_path = os.path.join(args.output_dir, 'metrics.json')
    with open(metrics_path, 'w') as f:
        json.dump(metrics, f, indent=2)
    
    print(f"Metrics saved to {metrics_path}")
    
    # Create plots directory
    plots_dir = os.path.join(args.output_dir, 'plots')
    os.makedirs(plots_dir, exist_ok=True)
    
    # Plot confusion matrix
    cm_path = os.path.join(plots_dir, 'confusion_matrix.png')
    plot_confusion_matrix(
        cm=np.array(metrics['confusion_matrix']),
        class_names=class_names,
        output_path=cm_path
    )
    
    print(f"Confusion matrix saved to {cm_path}")
    
    # Plot ROC curve for binary classification
    if 'roc_curve' in metrics:
        roc_path = os.path.join(plots_dir, 'roc_curve.png')
        plot_roc_curve(
            fpr=np.array(metrics['roc_curve']['fpr']),
            tpr=np.array(metrics['roc_curve']['tpr']),
            roc_auc=metrics['roc_auc'],
            output_path=roc_path
        )
        print(f"ROC curve saved to {roc_path}")
    
    # Plot class distribution
    dist_path = os.path.join(plots_dir, 'class_distribution.png')
    plot_class_distribution(
        y_true=y_test,
        y_pred=y_pred,
        class_names=class_names,
        output_path=dist_path
    )
    
    print(f"Class distribution saved to {dist_path}")
    
    # Plot misclassified examples
    misclassified_dir = os.path.join(plots_dir, 'misclassified')
    plot_misclassified_examples(
        x_test=x_test,
        y_true=y_test,
        y_pred=y_pred,
        class_names=class_names,
        output_dir=misclassified_dir
    )
    
    print(f"Misclassified examples saved to {misclassified_dir}")
    
    print("\nEvaluation complete!")

if __name__ == "__main__":
    main()