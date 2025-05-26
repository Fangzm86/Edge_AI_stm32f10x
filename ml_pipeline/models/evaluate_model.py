#!/usr/bin/env python3
"""
Command-line tool for evaluating trained models.
"""

import os
import sys
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf
from typing import Dict, List, Tuple, Optional, Union, Any
from sklearn.metrics import (
    confusion_matrix, classification_report,
    precision_score, recall_score, f1_score,
    roc_curve, auc, precision_recall_curve,
    average_precision_score
)

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ml_pipeline.data_processing import load_dataset
from ml_pipeline.models.model import load_model_with_metadata

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description='Evaluate a trained model.',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    parser.add_argument(
        '--model', 
        type=str,
        required=True,
        help='Path to trained model file'
    )
    
    parser.add_argument(
        '--data', 
        type=str,
        required=True,
        help='Path to test data file (NPZ format)'
    )
    
    parser.add_argument(
        '--output-dir', 
        type=str,
        default='evaluation_results',
        help='Directory to save evaluation results'
    )
    
    parser.add_argument(
        '--batch-size', 
        type=int,
        default=32,
        help='Batch size for evaluation'
    )
    
    parser.add_argument(
        '--gpu', 
        type=int,
        default=None,
        help='GPU index to use (None for CPU)'
    )
    
    parser.add_argument(
        '--save-predictions', 
        action='store_true',
        help='Save model predictions'
    )
    
    parser.add_argument(
        '--confusion-matrix', 
        action='store_true',
        help='Generate confusion matrix plot'
    )
    
    parser.add_argument(
        '--roc-curve', 
        action='store_true',
        help='Generate ROC curve plot'
    )
    
    parser.add_argument(
        '--precision-recall', 
        action='store_true',
        help='Generate precision-recall curve plot'
    )
    
    parser.add_argument(
        '--feature-importance', 
        action='store_true',
        help='Generate feature importance plot'
    )
    
    parser.add_argument(
        '--misclassified', 
        action='store_true',
        help='Save misclassified examples'
    )
    
    parser.add_argument(
        '--threshold', 
        type=float,
        default=0.5,
        help='Classification threshold for binary classification'
    )
    
    return parser.parse_args()

def plot_confusion_matrix(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    class_names: List[str],
    output_path: str,
    normalize: bool = True
) -> None:
    """
    Plot confusion matrix.
    
    Args:
        y_true: True labels
        y_pred: Predicted labels
        class_names: List of class names
        output_path: Path to save plot
        normalize: Whether to normalize confusion matrix
    """
    # Calculate confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    
    # Normalize if requested
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    
    # Create plot
    plt.figure(figsize=(10, 8))
    sns.heatmap(
        cm,
        annot=True,
        fmt='.2f' if normalize else 'd',
        cmap='Blues',
        xticklabels=class_names,
        yticklabels=class_names
    )
    
    plt.title('Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()

def plot_roc_curve(
    y_true: np.ndarray,
    y_pred_prob: np.ndarray,
    class_names: List[str],
    output_path: str
) -> None:
    """
    Plot ROC curve.
    
    Args:
        y_true: True labels
        y_pred_prob: Predicted probabilities
        class_names: List of class names
        output_path: Path to save plot
    """
    plt.figure(figsize=(10, 8))
    
    if y_pred_prob.shape[1] > 2:
        # Multi-class ROC curve
        fpr = dict()
        tpr = dict()
        roc_auc = dict()
        
        # Convert to one-hot encoding
        y_true_onehot = tf.keras.utils.to_categorical(y_true)
        
        for i in range(len(class_names)):
            fpr[i], tpr[i], _ = roc_curve(y_true_onehot[:, i], y_pred_prob[:, i])
            roc_auc[i] = auc(fpr[i], tpr[i])
            
            plt.plot(
                fpr[i],
                tpr[i],
                label=f'{class_names[i]} (AUC = {roc_auc[i]:.2f})'
            )
    else:
        # Binary classification
        if y_pred_prob.shape[1] == 2:
            y_pred_prob = y_pred_prob[:, 1]
        
        fpr, tpr, _ = roc_curve(y_true, y_pred_prob)
        roc_auc = auc(fpr, tpr)
        
        plt.plot(
            fpr,
            tpr,
            label=f'ROC curve (AUC = {roc_auc:.2f})'
        )
    
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC) Curve')
    plt.legend(loc="lower right")
    plt.grid(True)
    plt.savefig(output_path)
    plt.close()

def plot_precision_recall_curve(
    y_true: np.ndarray,
    y_pred_prob: np.ndarray,
    class_names: List[str],
    output_path: str
) -> None:
    """
    Plot precision-recall curve.
    
    Args:
        y_true: True labels
        y_pred_prob: Predicted probabilities
        class_names: List of class names
        output_path: Path to save plot
    """
    plt.figure(figsize=(10, 8))
    
    if y_pred_prob.shape[1] > 2:
        # Multi-class precision-recall curve
        precision = dict()
        recall = dict()
        avg_precision = dict()
        
        # Convert to one-hot encoding
        y_true_onehot = tf.keras.utils.to_categorical(y_true)
        
        for i in range(len(class_names)):
            precision[i], recall[i], _ = precision_recall_curve(
                y_true_onehot[:, i],
                y_pred_prob[:, i]
            )
            avg_precision[i] = average_precision_score(
                y_true_onehot[:, i],
                y_pred_prob[:, i]
            )
            
            plt.plot(
                recall[i],
                precision[i],
                label=f'{class_names[i]} (AP = {avg_precision[i]:.2f})'
            )
    else:
        # Binary classification
        if y_pred_prob.shape[1] == 2:
            y_pred_prob = y_pred_prob[:, 1]
        
        precision, recall, _ = precision_recall_curve(y_true, y_pred_prob)
        avg_precision = average_precision_score(y_true, y_pred_prob)
        
        plt.plot(
            recall,
            precision,
            label=f'Precision-Recall curve (AP = {avg_precision:.2f})'
        )
    
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall Curve')
    plt.legend(loc="lower left")
    plt.grid(True)
    plt.savefig(output_path)
    plt.close()

def analyze_feature_importance(
    model: tf.keras.Model,
    x_test: np.ndarray,
    feature_names: List[str],
    output_path: str,
    method: str = 'permutation'
) -> None:
    """
    Analyze and plot feature importance.
    
    Args:
        model: Trained model
        x_test: Test data
        feature_names: List of feature names
        output_path: Path to save plot
        method: Feature importance method ('permutation' or 'gradient')
    """
    if method == 'permutation':
        # Calculate permutation importance
        baseline_pred = model.predict(x_test)
        importance_scores = []
        
        for i in range(x_test.shape[2]):  # For each feature
            # Create copy and permute feature
            x_permuted = x_test.copy()
            x_permuted[:, :, i] = np.random.permutation(x_permuted[:, :, i])
            
            # Calculate performance drop
            permuted_pred = model.predict(x_permuted)
            importance = np.mean(np.abs(baseline_pred - permuted_pred))
            importance_scores.append(importance)
    
    elif method == 'gradient':
        # Calculate gradient-based importance
        importance_scores = []
        
        for i in range(x_test.shape[2]):  # For each feature
            with tf.GradientTape() as tape:
                x_tensor = tf.convert_to_tensor(x_test)
                tape.watch(x_tensor)
                predictions = model(x_tensor)
                gradients = tape.gradient(predictions, x_tensor)
                importance = np.mean(np.abs(gradients.numpy()[:, :, i]))
                importance_scores.append(importance)
    
    else:
        raise ValueError(f"Unknown feature importance method: {method}")
    
    # Normalize importance scores
    importance_scores = np.array(importance_scores)
    importance_scores = importance_scores / np.max(importance_scores)
    
    # Sort features by importance
    sorted_idx = np.argsort(importance_scores)
    pos = np.arange(len(sorted_idx))
    
    # Plot feature importance
    plt.figure(figsize=(10, 6))
    plt.barh(pos, importance_scores[sorted_idx])
    plt.yticks(pos, np.array(feature_names)[sorted_idx])
    plt.xlabel('Relative Importance')
    plt.title('Feature Importance')
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()

def save_misclassified_examples(
    x_test: np.ndarray,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    class_names: List[str],
    feature_names: List[str],
    output_dir: str,
    max_examples: int = 10
) -> None:
    """
    Save misclassified examples for analysis.
    
    Args:
        x_test: Test data
        y_true: True labels
        y_pred: Predicted labels
        class_names: List of class names
        feature_names: List of feature names
        output_dir: Directory to save examples
        max_examples: Maximum number of examples to save per class
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Find misclassified examples
    misclassified = np.where(y_true != y_pred)[0]
    
    # Group by true class
    for true_class in range(len(class_names)):
        class_examples = misclassified[y_true[misclassified] == true_class]
        
        if len(class_examples) == 0:
            continue
        
        # Select examples
        selected = class_examples[:max_examples]
        
        # Plot examples
        for i, idx in enumerate(selected):
            plt.figure(figsize=(12, 4))
            
            # Plot each feature
            for j in range(x_test.shape[2]):
                plt.plot(x_test[idx, :, j], label=feature_names[j])
            
            plt.title(f'True: {class_names[true_class]}, Predicted: {class_names[y_pred[idx]]}')
            plt.xlabel('Time')
            plt.ylabel('Value')
            plt.legend()
            plt.grid(True)
            
            plt.savefig(os.path.join(
                output_dir,
                f'misclassified_class{true_class}_example{i+1}.png'
            ))
            plt.close()

def main():
    """Main function."""
    # Parse arguments
    args = parse_args()
    
    # Configure GPU
    if args.gpu is not None:
        gpus = tf.config.list_physical_devices('GPU')
        if gpus:
            try:
                tf.config.set_visible_devices(gpus[args.gpu], 'GPU')
                tf.config.experimental.set_memory_growth(gpus[args.gpu], True)
                print(f"Using GPU {args.gpu}: {gpus[args.gpu]}")
            except RuntimeError as e:
                print(f"Error configuring GPU: {e}")
    
    # Load model and metadata
    print(f"\nLoading model from {args.model}...")
    model, metadata = load_model_with_metadata(args.model)
    
    # Load test data
    print(f"\nLoading test data from {args.data}...")
    x_test, y_test, _ = load_dataset(args.data)
    print(f"Loaded {len(x_test)} test samples")
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Get class names and feature names
    class_names = metadata.get('class_names')
    if class_names is None:
        class_names = [f"Class {i}" for i in range(metadata['num_classes'])]
    
    feature_names = metadata.get('feature_names')
    if feature_names is None:
        feature_names = [f"Feature {i}" for i in range(x_test.shape[2])]
    
    # Evaluate model
    print("\nEvaluating model...")
    test_loss, test_accuracy = model.evaluate(x_test, y_test, batch_size=args.batch_size)
    print(f"\nTest Loss: {test_loss:.4f}")
    print(f"Test Accuracy: {test_accuracy:.4f}")
    
    # Get predictions
    y_pred_prob = model.predict(x_test, batch_size=args.batch_size)
    
    # Convert probabilities to class indices
    if y_pred_prob.shape[1] > 1:
        y_pred = np.argmax(y_pred_prob, axis=1)
    else:
        y_pred = (y_pred_prob > args.threshold).astype(int).flatten()
    
    # Calculate metrics
    precision = precision_score(y_test, y_pred, average='weighted')
    recall = recall_score(y_test, y_pred, average='weighted')
    f1 = f1_score(y_test, y_pred, average='weighted')
    
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1 Score: {f1:.4f}")
    
    # Generate classification report
    report = classification_report(y_test, y_pred, target_names=class_names)
    print("\nClassification Report:")
    print(report)
    
    # Save evaluation results
    results = {
        'test_loss': test_loss,
        'test_accuracy': test_accuracy,
        'precision': precision,
        'recall': recall,
        'f1_score': f1,
        'classification_report': report
    }
    
    with open(os.path.join(args.output_dir, 'evaluation_results.txt'), 'w') as f:
        for key, value in results.items():
            f.write(f"{key}:\n{value}\n\n")
    
    # Save predictions if requested
    if args.save_predictions:
        predictions = pd.DataFrame({
            'true_label': y_test,
            'predicted_label': y_pred
        })
        
        if y_pred_prob.shape[1] > 1:
            for i, class_name in enumerate(class_names):
                predictions[f'probability_{class_name}'] = y_pred_prob[:, i]
        else:
            predictions['probability'] = y_pred_prob.flatten()
        
        predictions.to_csv(
            os.path.join(args.output_dir, 'predictions.csv'),
            index=False
        )
    
    # Generate confusion matrix plot
    if args.confusion_matrix:
        print("\nGenerating confusion matrix plot...")
        plot_confusion_matrix(
            y_test,
            y_pred,
            class_names,
            os.path.join(args.output_dir, 'confusion_matrix.png')
        )
    
    # Generate ROC curve plot
    if args.roc_curve:
        print("\nGenerating ROC curve plot...")
        plot_roc_curve(
            y_test,
            y_pred_prob,
            class_names,
            os.path.join(args.output_dir, 'roc_curve.png')
        )
    
    # Generate precision-recall curve plot
    if args.precision_recall:
        print("\nGenerating precision-recall curve plot...")
        plot_precision_recall_curve(
            y_test,
            y_pred_prob,
            class_names,
            os.path.join(args.output_dir, 'precision_recall_curve.png')
        )
    
    # Generate feature importance plot
    if args.feature_importance:
        print("\nAnalyzing feature importance...")
        analyze_feature_importance(
            model,
            x_test,
            feature_names,
            os.path.join(args.output_dir, 'feature_importance.png')
        )
    
    # Save misclassified examples
    if args.misclassified:
        print("\nSaving misclassified examples...")
        save_misclassified_examples(
            x_test,
            y_test,
            y_pred,
            class_names,
            feature_names,
            os.path.join(args.output_dir, 'misclassified_examples')
        )
    
    print(f"\nEvaluation results saved to {args.output_dir}")

if __name__ == "__main__":
    main()