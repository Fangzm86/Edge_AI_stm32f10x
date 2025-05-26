#!/usr/bin/env python3
"""
Script for training time series classification models.
"""

import os
import sys
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
from datetime import datetime
from typing import Dict, List, Tuple, Optional, Union, Any
import yaml
import json

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from models.model import create_model, save_model_with_metadata, get_learning_rate_scheduler
from data_processing.data_processing import load_dataset
from data_processing.config_utils import load_config, print_config_summary

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description='Train time series classification model.',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    parser.add_argument(
        '--data', 
        type=str,
        required=True,
        help='Path to processed data file (.npz)'
    )
    
    parser.add_argument(
        '--config', 
        type=str,
        default='model_config.yaml',
        help='Path to model configuration file'
    )
    
    parser.add_argument(
        '--output-dir', 
        type=str,
        default='models',
        help='Directory to save trained model'
    )
    
    parser.add_argument(
        '--name', 
        type=str,
        default=None,
        help='Model name (default: auto-generated based on architecture and timestamp)'
    )
    
    parser.add_argument(
        '--epochs', 
        type=int,
        help='Number of training epochs (overrides config)'
    )
    
    parser.add_argument(
        '--batch-size', 
        type=int,
        help='Batch size (overrides config)'
    )
    
    parser.add_argument(
        '--learning-rate', 
        type=float,
        help='Learning rate (overrides config)'
    )
    
    parser.add_argument(
        '--no-early-stopping', 
        action='store_true',
        help='Disable early stopping'
    )
    
    parser.add_argument(
        '--gpu', 
        type=int,
        default=None,
        help='GPU device ID to use (default: use all available GPUs)'
    )
    
    parser.add_argument(
        '--no-validation', 
        action='store_true',
        help='Disable validation split (use all data for training)'
    )
    
    parser.add_argument(
        '--test-split', 
        type=float,
        default=0.0,
        help='Fraction of data to use for testing (0.0 to 1.0)'
    )
    
    parser.add_argument(
        '--seed', 
        type=int,
        default=42,
        help='Random seed for reproducibility'
    )
    
    return parser.parse_args()

def update_config_from_args(config, args):
    """Update configuration with command line arguments."""
    # Update training settings
    if args.epochs:
        config['training']['epochs'] = args.epochs
    
    if args.batch_size:
        config['training']['batch_size'] = args.batch_size
    
    if args.learning_rate:
        config['training']['learning_rate'] = args.learning_rate
    
    if args.no_early_stopping:
        config['training']['early_stopping_patience'] = -1
    
    if args.no_validation:
        config['training']['validation_split'] = 0.0
    
    return config

def prepare_data(
    data_path: str,
    validation_split: float = 0.2,
    test_split: float = 0.0,
    random_seed: int = 42
) -> Tuple[
    np.ndarray, np.ndarray,  # Training data
    Optional[np.ndarray], Optional[np.ndarray],  # Validation data
    Optional[np.ndarray], Optional[np.ndarray]   # Test data
]:
    """
    Load and prepare data for training.
    
    Args:
        data_path: Path to processed data file
        validation_split: Fraction of data to use for validation
        test_split: Fraction of data to use for testing
        random_seed: Random seed for reproducibility
    
    Returns:
        x_train: Training features
        y_train: Training labels
        x_val: Validation features (or None if validation_split=0)
        y_val: Validation labels (or None if validation_split=0)
        x_test: Test features (or None if test_split=0)
        y_test: Test labels (or None if test_split=0)
    """
    # Load data
    windows, labels, _ = load_dataset(data_path)
    
    # Set random seed
    np.random.seed(random_seed)
    
    # Shuffle data
    indices = np.arange(len(windows))
    np.random.shuffle(indices)
    windows = windows[indices]
    labels = labels[indices]
    
    # Split data
    if test_split > 0:
        test_size = int(len(windows) * test_split)
        x_test = windows[:test_size]
        y_test = labels[:test_size]
        windows = windows[test_size:]
        labels = labels[test_size:]
    else:
        x_test = None
        y_test = None
    
    if validation_split > 0:
        val_size = int(len(windows) * validation_split)
        x_val = windows[:val_size]
        y_val = labels[:val_size]
        x_train = windows[val_size:]
        y_train = labels[val_size:]
    else:
        x_val = None
        y_val = None
        x_train = windows
        y_train = labels
    
    return x_train, y_train, x_val, y_val, x_test, y_test

def compute_class_weights(labels: np.ndarray) -> Dict[int, float]:
    """
    Compute class weights for imbalanced datasets.
    
    Args:
        labels: Training labels
    
    Returns:
        Dictionary mapping class indices to weights
    """
    # Count class occurrences
    unique, counts = np.unique(labels, return_counts=True)
    
    # Compute weights
    total = len(labels)
    n_classes = len(unique)
    weights = {}
    
    for i, count in zip(unique, counts):
        weights[int(i)] = total / (n_classes * count)
    
    return weights

def create_callbacks(
    config: Dict[str, Any],
    output_dir: str,
    model_name: str
) -> List[tf.keras.callbacks.Callback]:
    """
    Create training callbacks.
    
    Args:
        config: Training configuration
        output_dir: Directory to save logs and checkpoints
        model_name: Model name
    
    Returns:
        List of callbacks
    """
    callbacks = []
    
    # Create log directory
    log_dir = os.path.join(output_dir, 'logs', model_name)
    os.makedirs(log_dir, exist_ok=True)
    
    # TensorBoard callback
    callbacks.append(tf.keras.callbacks.TensorBoard(
        log_dir=log_dir,
        histogram_freq=1,
        write_graph=True,
        update_freq='epoch'
    ))
    
    # Model checkpoint callback
    checkpoint_path = os.path.join(output_dir, 'checkpoints', f"{model_name}_best.h5")
    os.makedirs(os.path.dirname(checkpoint_path), exist_ok=True)
    
    callbacks.append(tf.keras.callbacks.ModelCheckpoint(
        checkpoint_path,
        monitor='val_loss' if config['training'].get('validation_split', 0) > 0 else 'loss',
        save_best_only=True,
        save_weights_only=False,
        verbose=1
    ))
    
    # Early stopping callback
    early_stopping_patience = config['training'].get('early_stopping_patience', 10)
    if early_stopping_patience > 0:
        callbacks.append(tf.keras.callbacks.EarlyStopping(
            monitor='val_loss' if config['training'].get('validation_split', 0) > 0 else 'loss',
            patience=early_stopping_patience,
            restore_best_weights=True,
            verbose=1
        ))
    
    # Learning rate scheduler
    lr_scheduler = get_learning_rate_scheduler(config['training'])
    if lr_scheduler:
        callbacks.append(lr_scheduler)
    
    # CSV logger
    csv_path = os.path.join(log_dir, 'training_log.csv')
    callbacks.append(tf.keras.callbacks.CSVLogger(
        csv_path,
        separator=',',
        append=False
    ))
    
    return callbacks

def plot_training_history(
    history: tf.keras.callbacks.History,
    output_dir: str,
    model_name: str
) -> None:
    """
    Plot training history.
    
    Args:
        history: Training history
        output_dir: Directory to save plots
        model_name: Model name
    """
    # Create output directory
    plot_dir = os.path.join(output_dir, 'plots')
    os.makedirs(plot_dir, exist_ok=True)
    
    # Plot training and validation loss
    plt.figure(figsize=(12, 4))
    
    plt.subplot(1, 2, 1)
    plt.plot(history.history['loss'], label='Training Loss')
    if 'val_loss' in history.history:
        plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title('Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    
    plt.subplot(1, 2, 2)
    plt.plot(history.history['accuracy'], label='Training Accuracy')
    if 'val_accuracy' in history.history:
        plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
    plt.title('Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig(os.path.join(plot_dir, f"{model_name}_training_history.png"))
    plt.close()
    
    # Save history to JSON
    with open(os.path.join(plot_dir, f"{model_name}_history.json"), 'w') as f:
        json.dump(history.history, f, indent=2)

def main():
    """Main function."""
    # Parse arguments
    args = parse_args()
    
    # Set random seed
    np.random.seed(args.seed)
    tf.random.set_seed(args.seed)
    
    # Configure GPU
    if args.gpu is not None:
        gpus = tf.config.list_physical_devices('GPU')
        if gpus:
            try:
                tf.config.set_visible_devices(gpus[args.gpu], 'GPU')
                print(f"Using GPU {args.gpu}: {gpus[args.gpu]}")
            except RuntimeError as e:
                print(f"Error setting GPU: {e}")
    
    # Load configuration
    config = load_config(args.config)
    if config is None:
        print("Error loading configuration. Exiting.")
        sys.exit(1)
    
    # Update configuration with command line arguments
    config = update_config_from_args(config, args)
    
    # Print configuration summary
    print_config_summary(config)
    
    # Prepare data
    print("\nPreparing data...")
    x_train, y_train, x_val, y_val, x_test, y_test = prepare_data(
        args.data,
        validation_split=config['training'].get('validation_split', 0.2),
        test_split=args.test_split,
        random_seed=args.seed
    )
    
    print(f"Training data shape: {x_train.shape}")
    if x_val is not None:
        print(f"Validation data shape: {x_val.shape}")
    if x_test is not None:
        print(f"Test data shape: {x_test.shape}")
    
    # Get input shape and number of classes
    input_shape = (x_train.shape[1], x_train.shape[2])
    num_classes = len(np.unique(y_train))
    
    print(f"Input shape: {input_shape}")
    print(f"Number of classes: {num_classes}")
    
    # Generate model name if not provided
    if args.name is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        args.name = f"{config['model']['architecture']}_{timestamp}"
    
    # Create model
    print("\nCreating model...")
    model = create_model(
        config=config['model'],
        input_shape=input_shape,
        num_classes=num_classes
    )
    
    model.summary()
    
    # Create callbacks
    callbacks = create_callbacks(
        config=config,
        output_dir=args.output_dir,
        model_name=args.name
    )
    
    # Compute class weights if enabled
    class_weights = None
    if config['training'].get('use_class_weights', False):
        class_weights = compute_class_weights(y_train)
        print("\nClass weights:")
        for class_id, weight in class_weights.items():
            print(f"  Class {class_id}: {weight:.4f}")
    
    # Train model
    print("\nTraining model...")
    history = model.fit(
        x_train, y_train,
        epochs=config['training'].get('epochs', 100),
        batch_size=config['training'].get('batch_size', 32),
        validation_data=(x_val, y_val) if x_val is not None else None,
        callbacks=callbacks,
        class_weight=class_weights,
        verbose=1
    )
    
    # Plot training history
    plot_training_history(
        history=history,
        output_dir=args.output_dir,
        model_name=args.name
    )
    
    # Evaluate model on test data if available
    if x_test is not None:
        print("\nEvaluating model on test data...")
        test_loss, test_acc = model.evaluate(x_test, y_test, verbose=1)
        print(f"Test loss: {test_loss:.4f}")
        print(f"Test accuracy: {test_acc:.4f}")
    
    # Save model with metadata
    model_path = save_model_with_metadata(
        model=model,
        output_dir=args.output_dir,
        model_name=args.name,
        config=config,
        input_shape=input_shape,
        num_classes=num_classes
    )
    
    print(f"\nModel saved to {model_path}")
    print(f"Training complete!")

if __name__ == "__main__":
    main()