#!/usr/bin/env python3
"""
Command-line tool for processing time series data.
"""

import os
import sys
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from typing import Dict, List, Any, Optional

from data_processing import (
    create_windows, normalize_data, normalize_windows,
    augment_windows, extract_features, save_dataset
)
from config_utils import (
    load_config, get_default_config, print_config_summary,
    update_config_from_args, save_config
)

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description='Process time series data for machine learning.',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    parser.add_argument(
        '--input', 
        type=str,
        required=True,
        help='Path to input CSV file'
    )
    
    parser.add_argument(
        '--output', 
        type=str,
        required=True,
        help='Path to output NPZ file'
    )
    
    parser.add_argument(
        '--config', 
        type=str,
        help='Path to configuration file'
    )
    
    parser.add_argument(
        '--window-length', 
        type=int,
        help='Window length'
    )
    
    parser.add_argument(
        '--overlap', 
        type=float,
        help='Window overlap (0.0 to 1.0)'
    )
    
    parser.add_argument(
        '--window-method', 
        type=str,
        choices=['overlap', 'step'],
        help='Windowing method'
    )
    
    parser.add_argument(
        '--features', 
        type=str,
        help='Comma-separated list of feature columns'
    )
    
    parser.add_argument(
        '--label', 
        type=str,
        help='Label column name'
    )
    
    parser.add_argument(
        '--normalize', 
        type=str,
        choices=['standard', 'minmax', 'robust', 'none'],
        help='Normalization method'
    )
    
    parser.add_argument(
        '--window-normalize', 
        type=str,
        choices=['zscore', 'minmax', 'robust', 'none'],
        help='Window normalization method'
    )
    
    parser.add_argument(
        '--augment', 
        action='store_true',
        help='Enable data augmentation'
    )
    
    parser.add_argument(
        '--extract-features', 
        action='store_true',
        help='Extract features from windows'
    )
    
    parser.add_argument(
        '--split', 
        action='store_true',
        help='Split data into train/val/test sets'
    )
    
    parser.add_argument(
        '--visualize', 
        action='store_true',
        help='Visualize data processing steps'
    )
    
    parser.add_argument(
        '--save-config', 
        type=str,
        help='Save configuration to file'
    )
    
    return parser.parse_args()

def load_data(input_path: str, config: Dict[str, Any]) -> pd.DataFrame:
    """
    Load data from CSV file.
    
    Args:
        input_path: Path to input CSV file
        config: Configuration dictionary
    
    Returns:
        DataFrame containing time series data
    """
    # Get data loading parameters
    delimiter = config['data'].get('delimiter', ',')
    header = config['data'].get('header', 0)
    
    # Load data
    try:
        data = pd.read_csv(input_path, delimiter=delimiter, header=header)
        return data
    except Exception as e:
        print(f"Error loading data: {str(e)}")
        sys.exit(1)

def visualize_data(
    data: pd.DataFrame,
    windows: np.ndarray,
    normalized_windows: np.ndarray,
    augmented_windows: Optional[np.ndarray],
    labels: np.ndarray,
    config: Dict[str, Any],
    output_dir: str
) -> None:
    """
    Visualize data processing steps.
    
    Args:
        data: Original DataFrame
        windows: Created windows
        normalized_windows: Normalized windows
        augmented_windows: Augmented windows (or None)
        labels: Window labels
        config: Configuration dictionary
        output_dir: Directory to save visualizations
    """
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Get feature columns
    feature_columns = config['data'].get('feature_columns', [])
    if not feature_columns:
        feature_columns = list(data.columns)
        label_column = config['data'].get('label_column')
        if label_column in feature_columns:
            feature_columns.remove(label_column)
    
    # Visualize original data
    plt.figure(figsize=(12, 6))
    for column in feature_columns[:min(6, len(feature_columns))]:
        plt.plot(data[column].values[:1000], label=column)
    plt.title('Original Data (first 1000 samples)')
    plt.xlabel('Sample')
    plt.ylabel('Value')
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(output_dir, 'original_data.png'))
    plt.close()
    
    # Visualize windows
    n_features = min(3, windows.shape[2])
    plt.figure(figsize=(15, 10))
    
    # Plot windows for each class
    unique_labels = np.unique(labels)
    for i, label in enumerate(unique_labels[:min(3, len(unique_labels))]):
        # Find windows with this label
        class_indices = np.where(labels == label)[0]
        if len(class_indices) == 0:
            continue
        
        # Select a window
        window_idx = class_indices[0]
        window = windows[window_idx]
        
        # Plot features
        for j in range(n_features):
            plt.subplot(len(unique_labels[:min(3, len(unique_labels))]), n_features, i*n_features + j + 1)
            plt.plot(window[:, j])
            plt.title(f'Class {label}, Feature {j}')
            plt.xlabel('Time')
            plt.ylabel('Value')
            plt.grid(True)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'windows.png'))
    plt.close()
    
    # Visualize normalized windows
    plt.figure(figsize=(15, 10))
    
    # Plot normalized windows for each class
    for i, label in enumerate(unique_labels[:min(3, len(unique_labels))]):
        # Find windows with this label
        class_indices = np.where(labels == label)[0]
        if len(class_indices) == 0:
            continue
        
        # Select a window
        window_idx = class_indices[0]
        window = normalized_windows[window_idx]
        
        # Plot features
        for j in range(n_features):
            plt.subplot(len(unique_labels[:min(3, len(unique_labels))]), n_features, i*n_features + j + 1)
            plt.plot(window[:, j])
            plt.title(f'Normalized, Class {label}, Feature {j}')
            plt.xlabel('Time')
            plt.ylabel('Value')
            plt.grid(True)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'normalized_windows.png'))
    plt.close()
    
    # Visualize augmented windows if available
    if augmented_windows is not None:
        plt.figure(figsize=(15, 10))
        
        # Find original and augmented windows for a class
        for i, label in enumerate(unique_labels[:min(3, len(unique_labels))]):
            # Find original windows with this label
            orig_indices = np.where(labels == label)[0]
            if len(orig_indices) == 0:
                continue
            
            # Select an original window
            orig_window = normalized_windows[orig_indices[0]]
            
            # Find augmented windows with this label
            aug_indices = np.where(labels == label)[0]
            if len(aug_indices) <= len(orig_indices):
                continue
            
            # Select an augmented window
            aug_window = augmented_windows[aug_indices[len(orig_indices)]]
            
            # Plot comparison
            for j in range(n_features):
                plt.subplot(len(unique_labels[:min(3, len(unique_labels))]), n_features, i*n_features + j + 1)
                plt.plot(orig_window[:, j], label='Original')
                plt.plot(aug_window[:, j], label='Augmented')
                plt.title(f'Class {label}, Feature {j}')
                plt.xlabel('Time')
                plt.ylabel('Value')
                plt.legend()
                plt.grid(True)
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'augmented_windows.png'))
        plt.close()
    
    # Visualize class distribution
    plt.figure(figsize=(10, 6))
    unique, counts = np.unique(labels, return_counts=True)
    plt.bar(unique, counts)
    plt.title('Class Distribution')
    plt.xlabel('Class')
    plt.ylabel('Count')
    plt.grid(True, axis='y')
    plt.savefig(os.path.join(output_dir, 'class_distribution.png'))
    plt.close()
    
    print(f"Visualizations saved to {output_dir}")

def main():
    """Main function."""
    # Parse arguments
    args = parse_args()
    
    # Load configuration
    if args.config:
        config = load_config(args.config)
        if config is None:
            print("Error loading configuration. Using default configuration.")
            config = get_default_config()
    else:
        config = get_default_config()
    
    # Update configuration from command line arguments
    config = update_config_from_args(config, args)
    
    # Print configuration summary
    print_config_summary(config)
    
    # Save configuration if requested
    if args.save_config:
        if save_config(config, args.save_config):
            print(f"Configuration saved to {args.save_config}")
        else:
            print(f"Error saving configuration to {args.save_config}")
    
    # Load data
    print(f"\nLoading data from {args.input}...")
    data = load_data(args.input, config)
    print(f"Loaded {len(data)} samples with {len(data.columns)} columns")
    
    # Get parameters
    feature_columns = config['data'].get('feature_columns', [])
    if not feature_columns:
        feature_columns = list(data.columns)
        label_column = config['data'].get('label_column')
        if label_column in feature_columns:
            feature_columns.remove(label_column)
    
    label_column = config['data'].get('label_column')
    window_length = config['window'].get('length', 128)
    overlap = config['window'].get('overlap', 0.5)
    window_method = config['window'].get('method', 'overlap')
    
    # Create windows
    print(f"\nCreating windows (length={window_length}, overlap={overlap})...")
    windows, labels, scaler = create_windows(
        data,
        window_length=window_length,
        overlap=overlap,
        feature_columns=feature_columns,
        label_column=label_column,
        method=window_method
    )
    
    print(f"Created {len(windows)} windows with shape {windows.shape}")
    
    # Normalize data
    norm_method = config['normalization'].get('method', 'standard')
    if norm_method != 'none':
        print(f"\nNormalizing data (method={norm_method})...")
        normalized_data, scaler = normalize_data(
            data[feature_columns].values,
            method=norm_method,
            scaler=scaler
        )
    
    # Normalize windows
    window_norm_method = config['normalization'].get('window_normalization', 'zscore')
    if window_norm_method != 'none':
        print(f"\nNormalizing windows (method={window_norm_method})...")
        normalized_windows = normalize_windows(
            windows,
            method=window_norm_method
        )
    else:
        normalized_windows = windows
    
    # Augment data
    if config['augmentation'].get('enabled', False):
        print("\nAugmenting data...")
        augmented_windows, augmented_labels = augment_windows(
            normalized_windows,
            labels,
            config['augmentation']
        )
        print(f"Augmented data: {len(augmented_windows)} windows")
    else:
        augmented_windows = normalized_windows
        augmented_labels = labels
    
    # Extract features
    if config['features'].get('extract', False):
        print("\nExtracting features...")
        features = extract_features(
            augmented_windows,
            feature_types=config['features'].get('types', ['statistical', 'frequency'])
        )
        print(f"Extracted {features.shape[1]} features")
        
        # Replace windows with extracted features
        augmented_windows = features
    
    # Split data
    if config['output']['split'].get('enabled', False):
        print("\nSplitting data...")
        train_ratio = config['output']['split'].get('train_ratio', 0.7)
        val_ratio = config['output']['split'].get('val_ratio', 0.15)
        test_ratio = config['output']['split'].get('test_ratio', 0.15)
        
        # Shuffle data
        indices = np.arange(len(augmented_windows))
        np.random.shuffle(indices)
        shuffled_windows = augmented_windows[indices]
        shuffled_labels = augmented_labels[indices]
        
        # Calculate split indices
        n_samples = len(shuffled_windows)
        train_idx = int(n_samples * train_ratio)
        val_idx = train_idx + int(n_samples * val_ratio)
        
        # Split data
        train_windows = shuffled_windows[:train_idx]
        train_labels = shuffled_labels[:train_idx]
        
        val_windows = shuffled_windows[train_idx:val_idx]
        val_labels = shuffled_labels[train_idx:val_idx]
        
        test_windows = shuffled_windows[val_idx:]
        test_labels = shuffled_labels[val_idx:]
        
        # Save splits
        output_base = os.path.splitext(args.output)[0]
        
        save_dataset(train_windows, train_labels, f"{output_base}_train.npz", scaler)
        save_dataset(val_windows, val_labels, f"{output_base}_val.npz", scaler)
        save_dataset(test_windows, test_labels, f"{output_base}_test.npz", scaler)
        
        print(f"Saved train split: {len(train_windows)} windows")
        print(f"Saved validation split: {len(val_windows)} windows")
        print(f"Saved test split: {len(test_windows)} windows")
    else:
        # Save all data
        save_dataset(augmented_windows, augmented_labels, args.output, scaler)
        print(f"\nSaved {len(augmented_windows)} windows to {args.output}")
    
    # Visualize data
    if args.visualize:
        print("\nGenerating visualizations...")
        output_dir = os.path.join(os.path.dirname(args.output), 'visualizations')
        visualize_data(
            data,
            windows,
            normalized_windows,
            augmented_windows if config['augmentation'].get('enabled', False) else None,
            labels,
            config,
            output_dir
        )
    
    print("\nData processing complete!")

if __name__ == "__main__":
    main()