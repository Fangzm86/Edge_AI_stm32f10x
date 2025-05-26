#!/usr/bin/env python3
"""
Command-line tool for training models.
"""

import os
import sys
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
from typing import Dict, List, Tuple, Optional, Union, Any

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ml_pipeline.data_processing import load_dataset, split_data
from ml_pipeline.data_processing import load_config as load_data_config
from ml_pipeline.models.model import create_model, get_optimizer, get_learning_rate_scheduler
from ml_pipeline.models.model import save_model_with_metadata, get_model_summary, get_model_size

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description='Train a model for time series classification.',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    parser.add_argument(
        '--data', 
        type=str,
        required=True,
        help='Path to input data file (NPZ format)'
    )
    
    parser.add_argument(
        '--output-dir', 
        type=str,
        default='trained_models',
        help='Directory to save trained model'
    )
    
    parser.add_argument(
        '--model-name', 
        type=str,
        default='model',
        help='Name for the trained model'
    )
    
    parser.add_argument(
        '--config', 
        type=str,
        help='Path to model configuration file'
    )
    
    parser.add_argument(
        '--architecture', 
        type=str,
        choices=['lstm', 'cnn', 'hybrid', 'tcn', 'transformer', 'resnet'],
        help='Model architecture'
    )
    
    parser.add_argument(
        '--batch-size', 
        type=int,
        help='Batch size for training'
    )
    
    parser.add_argument(
        '--epochs', 
        type=int,
        help='Number of training epochs'
    )
    
    parser.add_argument(
        '--learning-rate', 
        type=float,
        help='Learning rate'
    )
    
    parser.add_argument(
        '--optimizer', 
        type=str,
        choices=['adam', 'sgd', 'rmsprop', 'adagrad', 'adadelta', 'adamax', 'nadam'],
        help='Optimizer'
    )
    
    parser.add_argument(
        '--validation-split', 
        type=float,
        help='Validation split ratio'
    )
    
    parser.add_argument(
        '--use-class-weights', 
        action='store_true',
        help='Use class weights for imbalanced data'
    )
    
    parser.add_argument(
        '--early-stopping', 
        action='store_true',
        help='Enable early stopping'
    )
    
    parser.add_argument(
        '--reduce-lr', 
        action='store_true',
        help='Enable learning rate reduction on plateau'
    )
    
    parser.add_argument(
        '--tensorboard', 
        action='store_true',
        help='Enable TensorBoard logging'
    )
    
    parser.add_argument(
        '--visualize', 
        action='store_true',
        help='Visualize training results'
    )
    
    parser.add_argument(
        '--random-seed', 
        type=int,
        default=42,
        help='Random seed for reproducibility'
    )
    
    parser.add_argument(
        '--gpu', 
        type=int,
        default=None,
        help='GPU index to use (None for CPU)'
    )
    
    return parser.parse_args()

def load_model_config(config_path: str) -> Dict[str, Any]:
    """
    Load model configuration from file.
    
    Args:
        config_path: Path to configuration file
    
    Returns:
        Configuration dictionary
    """
    if config_path.endswith('.yaml') or config_path.endswith('.yml'):
        import yaml
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
    elif config_path.endswith('.json'):
        import json
        with open(config_path, 'r') as f:
            config = json.load(f)
    else:
        raise ValueError(f"Unsupported configuration file format: {config_path}")
    
    return config

def update_config_from_args(config: Dict[str, Any], args: Any) -> Dict[str, Any]:
    """
    Update configuration from command line arguments.
    
    Args:
        config: Configuration dictionary
        args: Command line arguments
    
    Returns:
        Updated configuration dictionary
    """
    updated_config = config.copy()
    
    # Update model section
    if args.architecture:
        updated_config['model']['architecture'] = args.architecture
    
    # Update training section
    if args.batch_size:
        updated_config['training']['batch_size'] = args.batch_size
    
    if args.epochs:
        updated_config['training']['epochs'] = args.epochs
    
    if args.learning_rate:
        updated_config['training']['learning_rate'] = args.learning_rate
    
    if args.optimizer:
        updated_config['training']['optimizer'] = args.optimizer
    
    if args.validation_split:
        updated_config['training']['validation_split'] = args.validation_split
    
    if args.use_class_weights:
        updated_config['training']['use_class_weights'] = args.use_class_weights
    
    if args.early_stopping:
        updated_config['training']['early_stopping_patience'] = 10
    
    if args.reduce_lr:
        updated_config['training']['reduce_lr_patience'] = 5
    
    if args.random_seed:
        updated_config['training']['random_seed'] = args.random_seed
    
    return updated_config

def get_callbacks(config: Dict[str, Any], output_dir: str, model_name: str) -> List[tf.keras.callbacks.Callback]:
    """
    Get callbacks for model training.
    
    Args:
        config: Training configuration dictionary
        output_dir: Directory to save callback outputs
        model_name: Name for the model
    
    Returns:
        List of Keras callbacks
    """
    callbacks = []
    
    # Early stopping
    early_stopping_patience = config.get('early_stopping_patience')
    if early_stopping_patience:
        callbacks.append(tf.keras.callbacks.EarlyStopping(
            monitor='val_loss',
            patience=early_stopping_patience,
            restore_best_weights=True,
            verbose=1
        ))
    
    # Learning rate scheduler
    lr_scheduler = get_learning_rate_scheduler(config)
    if lr_scheduler:
        callbacks.append(lr_scheduler)
    
    # Model checkpoint
    checkpoint_config = config.get('checkpointing', {})
    if checkpoint_config.get('enabled', True):
        checkpoint_dir = os.path.join(output_dir, 'checkpoints')
        os.makedirs(checkpoint_dir, exist_ok=True)
        
        callbacks.append(tf.keras.callbacks.ModelCheckpoint(
            filepath=os.path.join(checkpoint_dir, f"{model_name}_{{epoch:02d}}_{{val_loss:.4f}}.h5"),
            monitor=checkpoint_config.get('monitor', 'val_loss'),
            save_best_only=checkpoint_config.get('save_best_only', True),
            save_weights_only=checkpoint_config.get('save_weights_only', False),
            mode=checkpoint_config.get('mode', 'min'),
            verbose=checkpoint_config.get('verbose', 1),
            period=checkpoint_config.get('period', 1)
        ))
    
    # TensorBoard
    tensorboard_dir = os.path.join(output_dir, 'logs')
    os.makedirs(tensorboard_dir, exist_ok=True)
    
    callbacks.append(tf.keras.callbacks.TensorBoard(
        log_dir=os.path.join(tensorboard_dir, model_name),
        histogram_freq=1,
        write_graph=True,
        write_images=False,
        update_freq='epoch',
        profile_batch=0
    ))
    
    return callbacks

def get_class_weights(labels: np.ndarray) -> Dict[int, float]:
    """
    Calculate class weights for imbalanced data.
    
    Args:
        labels: Training labels
    
    Returns:
        Dictionary mapping class indices to weights
    """
    # Count class occurrences
    unique_classes, class_counts = np.unique(labels, return_counts=True)
    
    # Calculate weights
    n_samples = len(labels)
    n_classes = len(unique_classes)
    
    weights = {}
    for i, class_idx in enumerate(unique_classes):
        weights[int(class_idx)] = n_samples / (n_classes * class_counts[i])
    
    return weights

def visualize_training_results(
    history: tf.keras.callbacks.History,
    output_dir: str,
    model_name: str
) -> None:
    """
    Visualize training results.
    
    Args:
        history: Training history
        output_dir: Directory to save visualizations
        model_name: Name for the model
    """
    # Create output directory
    vis_dir = os.path.join(output_dir, 'visualizations')
    os.makedirs(vis_dir, exist_ok=True)
    
    # Plot training & validation loss
    plt.figure(figsize=(12, 4))
    
    plt.subplot(1, 2, 1)
    plt.plot(history.history['loss'], label='Training Loss')
    if 'val_loss' in history.history:
        plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title('Model Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    
    plt.subplot(1, 2, 2)
    plt.plot(history.history['accuracy'], label='Training Accuracy')
    if 'val_accuracy' in history.history:
        plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
    plt.title('Model Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig(os.path.join(vis_dir, f"{model_name}_training_history.png"))
    plt.close()
    
    # Plot other metrics if available
    metrics = [key for key in history.history.keys() if not key.startswith('val_') and key not in ['loss', 'accuracy']]
    
    if metrics:
        plt.figure(figsize=(12, 4 * ((len(metrics) + 1) // 2)))
        
        for i, metric in enumerate(metrics):
            plt.subplot((len(metrics) + 1) // 2, 2, i + 1)
            plt.plot(history.history[metric], label=f'Training {metric}')
            if f'val_{metric}' in history.history:
                plt.plot(history.history[f'val_{metric}'], label=f'Validation {metric}')
            plt.title(f'Model {metric}')
            plt.xlabel('Epoch')
            plt.ylabel(metric)
            plt.legend()
            plt.grid(True)
        
        plt.tight_layout()
        plt.savefig(os.path.join(vis_dir, f"{model_name}_training_metrics.png"))
        plt.close()

def evaluate_model(
    model: tf.keras.Model,
    x_test: np.ndarray,
    y_test: np.ndarray,
    output_dir: str,
    model_name: str,
    class_names: List[str] = None
) -> Dict[str, Any]:
    """
    Evaluate model on test data.
    
    Args:
        model: Trained model
        x_test: Test data
        y_test: Test labels
        output_dir: Directory to save evaluation results
        model_name: Name for the model
        class_names: List of class names
    
    Returns:
        Dictionary containing evaluation results
    """
    # Create output directory
    eval_dir = os.path.join(output_dir, 'evaluation')
    os.makedirs(eval_dir, exist_ok=True)
    
    # Evaluate model
    test_loss, test_accuracy = model.evaluate(x_test, y_test, verbose=1)
    
    # Get predictions
    y_pred_prob = model.predict(x_test)
    
    # Convert probabilities to class indices
    if y_pred_prob.shape[1] > 1:
        y_pred = np.argmax(y_pred_prob, axis=1)
    else:
        y_pred = (y_pred_prob > 0.5).astype(int).flatten()
    
    # Calculate confusion matrix
    from sklearn.metrics import confusion_matrix, classification_report
    from sklearn.metrics import precision_score, recall_score, f1_score
    
    cm = confusion_matrix(y_test, y_pred)
    
    # Calculate metrics
    precision = precision_score(y_test, y_pred, average='weighted')
    recall = recall_score(y_test, y_pred, average='weighted')
    f1 = f1_score(y_test, y_pred, average='weighted')
    
    # Generate classification report
    if class_names is None:
        class_names = [f"Class {i}" for i in range(len(np.unique(y_test)))]
    
    report = classification_report(y_test, y_pred, target_names=class_names)
    
    # Save report
    with open(os.path.join(eval_dir, f"{model_name}_classification_report.txt"), 'w') as f:
        f.write(f"Test Loss: {test_loss}\n")
        f.write(f"Test Accuracy: {test_accuracy}\n")
        f.write(f"Precision: {precision}\n")
        f.write(f"Recall: {recall}\n")
        f.write(f"F1 Score: {f1}\n\n")
        f.write(report)
    
    # Visualize confusion matrix
    plt.figure(figsize=(10, 8))
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title('Confusion Matrix')
    plt.colorbar()
    
    tick_marks = np.arange(len(class_names))
    plt.xticks(tick_marks, class_names, rotation=45)
    plt.yticks(tick_marks, class_names)
    
    # Add text annotations
    thresh = cm.max() / 2.0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(j, i, format(cm[i, j], 'd'),
                    horizontalalignment="center",
                    color="white" if cm[i, j] > thresh else "black")
    
    plt.tight_layout()
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.savefig(os.path.join(eval_dir, f"{model_name}_confusion_matrix.png"))
    plt.close()
    
    # Return results
    return {
        'test_loss': test_loss,
        'test_accuracy': test_accuracy,
        'precision': precision,
        'recall': recall,
        'f1_score': f1,
        'confusion_matrix': cm.tolist(),
        'classification_report': report
    }

def main():
    """Main function."""
    # Parse arguments
    args = parse_args()
    
    # Set random seed
    np.random.seed(args.random_seed)
    tf.random.set_seed(args.random_seed)
    
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
    
    # Load data
    print(f"\nLoading data from {args.data}...")
    windows, labels, scaler = load_dataset(args.data)
    print(f"Loaded {len(windows)} samples with shape {windows.shape}")
    
    # Get number of classes
    num_classes = len(np.unique(labels))
    print(f"Number of classes: {num_classes}")
    
    # Split data if not already split
    if args.data.endswith('_train.npz'):
        # Load validation data
        val_path = args.data.replace('_train.npz', '_val.npz')
        if os.path.exists(val_path):
            print(f"Loading validation data from {val_path}...")
            x_val, y_val, _ = load_dataset(val_path)
            print(f"Loaded {len(x_val)} validation samples")
            
            # Use training data as is
            x_train, y_train = windows, labels
        else:
            # Split training data
            print("Validation data not found. Splitting training data...")
            x_train, y_train, x_val, y_val, _, _ = split_data(
                windows, labels,
                train_ratio=0.8,
                val_ratio=0.2,
                test_ratio=0.0
            )
    else:
        # Split data
        print("Splitting data into training and validation sets...")
        x_train, y_train, x_val, y_val, _, _ = split_data(
            windows, labels,
            train_ratio=0.8,
            val_ratio=0.2,
            test_ratio=0.0
        )
    
    print(f"Training data: {len(x_train)} samples")
    print(f"Validation data: {len(x_val)} samples")
    
    # Load model configuration
    if args.config:
        print(f"\nLoading model configuration from {args.config}...")
        config = load_model_config(args.config)
    else:
        # Use default configuration
        print("\nUsing default model configuration...")
        config = {
            'model': {
                'architecture': 'lstm',
                'layer_sizes': [64, 32],
                'dropout_rate': 0.3,
                'activation': 'tanh',
                'use_batch_norm': True
            },
            'training': {
                'batch_size': 32,
                'epochs': 100,
                'learning_rate': 0.001,
                'optimizer': 'adam',
                'loss': 'sparse_categorical_crossentropy',
                'metrics': ['accuracy'],
                'early_stopping_patience': 10,
                'reduce_lr_patience': 5,
                'use_class_weights': True,
                'validation_split': 0.2
            }
        }
    
    # Update configuration from command line arguments
    config = update_config_from_args(config, args)
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Create model
    print("\nCreating model...")
    input_shape = x_train.shape[1:]
    model = create_model(config['model'], input_shape, num_classes)
    
    # Print model summary
    model.summary()
    
    # Get model size
    model_size = get_model_size(model)
    print(f"\nModel size: {model_size['model_size_bytes'] / 1024:.2f} KB")
    print(f"Total parameters: {model_size['total_parameters']}")
    print(f"Trainable parameters: {model_size['trainable_parameters']}")
    
    # Configure optimizer
    optimizer = get_optimizer(
        config['training'].get('optimizer', 'adam'),
        config['training'].get('learning_rate', 0.001),
        config['training'].get('optimizer_params', {})
    )
    
    # Configure loss and metrics
    loss = config['training'].get('loss', 'sparse_categorical_crossentropy')
    metrics = config['training'].get('metrics', ['accuracy'])
    
    # Compile model
    model.compile(
        optimizer=optimizer,
        loss=loss,
        metrics=metrics
    )
    
    # Get callbacks
    callbacks = get_callbacks(config['training'], args.output_dir, args.model_name)
    
    # Get class weights
    if config['training'].get('use_class_weights', True):
        class_weights = get_class_weights(y_train)
        print("\nUsing class weights:")
        for class_idx, weight in class_weights.items():
            print(f"  Class {class_idx}: {weight:.4f}")
    else:
        class_weights = None
    
    # Train model
    print("\nTraining model...")
    history = model.fit(
        x_train, y_train,
        batch_size=config['training'].get('batch_size', 32),
        epochs=config['training'].get('epochs', 100),
        validation_data=(x_val, y_val),
        callbacks=callbacks,
        class_weight=class_weights,
        verbose=1
    )
    
    # Visualize training results
    if args.visualize:
        print("\nVisualizing training results...")
        visualize_training_results(history, args.output_dir, args.model_name)
    
    # Evaluate model
    print("\nEvaluating model...")
    # Try to load test data
    test_path = args.data.replace('_train.npz', '_test.npz')
    if os.path.exists(test_path):
        print(f"Loading test data from {test_path}...")
        x_test, y_test, _ = load_dataset(test_path)
        print(f"Loaded {len(x_test)} test samples")
    else:
        # Use validation data for evaluation
        print("Test data not found. Using validation data for evaluation...")
        x_test, y_test = x_val, y_val
    
    # Get class names
    class_names = None
    if args.data.endswith('.npz'):
        try:
            data = np.load(args.data, allow_pickle=True)
            if 'class_names' in data:
                class_names = data['class_names']
        except:
            pass
    
    # Evaluate model
    evaluation = evaluate_model(
        model, x_test, y_test,
        args.output_dir, args.model_name,
        class_names=class_names
    )
    
    # Save model
    print("\nSaving model...")
    # Get feature names
    feature_names = None
    if args.data.endswith('.npz'):
        try:
            data = np.load(args.data, allow_pickle=True)
            if 'feature_names' in data:
                feature_names = data['feature_names']
        except:
            pass
    
    # Save model with metadata
    model_path = save_model_with_metadata(
        model,
        args.output_dir,
        args.model_name,
        config['model'],
        input_shape,
        num_classes,
        class_names=class_names,
        feature_names=feature_names
    )
    
    print(f"Model saved to {model_path}")
    print("\nTraining complete!")

if __name__ == "__main__":
    main()