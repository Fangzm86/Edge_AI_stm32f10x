"""
Data Processing Module for Time Series Analysis

This module provides functions for processing time series data,
including windowing, feature extraction, and data transformation.
"""

import numpy as np
import pandas as pd
from typing import Union, Tuple, Optional, List
from sklearn.preprocessing import MinMaxScaler


def create_windows(
    data: pd.DataFrame,
    window_length: int,
    step_size: Optional[int] = None,
    overlap: Optional[float] = None,
    feature_columns: Optional[List[str]] = None,
    label_column: str = 'class',
    scaler: Optional[MinMaxScaler] = None
) -> Tuple[np.ndarray, np.ndarray, Optional[MinMaxScaler]]:
    """
    Convert time series data into windowed format with corresponding labels.
    Data is first normalized using sklearn's MinMaxScaler, then windowed.
    
    Args:
        data: pandas DataFrame with shape (num_samples, feature_len + 1)
        window_length: Number of time steps in each window
        step_size: Number of time steps to move between windows (default: 1)
        overlap: Percentage of overlap between windows (0.0 to 1.0)
                 If provided, overrides step_size
        feature_columns: List of column names to use as features
                        If None, all columns except label_column are used
        label_column: Name of the column containing class labels (default: 'class')
        scaler: Optional pre-fitted MinMaxScaler. If None, a new one will be created
    
    Returns:
        Tuple of (windows, labels, scaler):
            - windows: numpy.ndarray with shape (num_windows, window_length, feature_length)
            - labels: numpy.ndarray with shape (num_windows,)
            - scaler: fitted MinMaxScaler for inverse transform if needed
    """
    # Validate inputs
    if window_length <= 0:
        raise ValueError("window_length must be positive")
    
    if window_length > len(data):
        raise ValueError(f"window_length ({window_length}) cannot be larger than data length ({len(data)})")
    
    # Determine step size
    if overlap is not None:
        if not 0 <= overlap < 1:
            raise ValueError("overlap must be between 0.0 and 1.0")
        step_size = max(1, int(window_length * (1 - overlap)))
    elif step_size is None:
        step_size = 1
    
    if step_size <= 0:
        raise ValueError("step_size must be positive")
    
    # Verify label column exists
    if label_column not in data.columns:
        raise ValueError(f"Label column '{label_column}' not found in data")
    
    # Extract labels
    labels = data[label_column].values
    
    # Select features
    if feature_columns is not None:
        data_features = data[feature_columns]
    else:
        # Use all columns except label column
        data_features = data.drop(columns=[label_column])
    
    # Normalize features using MinMaxScaler
    if scaler is None:
        scaler = MinMaxScaler()
        data_normalized = scaler.fit_transform(data_features)
    else:
        data_normalized = scaler.transform(data_features)
    
    # Calculate number of windows
    num_samples = len(data)
    num_windows = (num_samples - window_length) // step_size + 1
    
    if num_windows <= 0:
        raise ValueError(f"No windows could be created with the given parameters. "
                         f"Try reducing window_length ({window_length}) or increasing data length ({num_samples}).")
    
    # Get feature dimension
    feature_length = data_normalized.shape[1]
    
    # Initialize output arrays
    windows = np.zeros((num_windows, window_length, feature_length))
    window_labels = np.zeros(num_windows, dtype=labels.dtype)
    
    # Create windows and get corresponding labels
    for i in range(num_windows):
        start_idx = i * step_size
        end_idx = start_idx + window_length
        windows[i] = data_normalized[start_idx:end_idx]
        window_labels[i] = labels[start_idx]  # Use label from start of window
    
    return windows, window_labels, scaler


def sliding_window(
    data: Union[pd.DataFrame, np.ndarray],
    window_length: int,
    stride: int = 1,
    label_column: str = 'class',
    scaler: Optional[MinMaxScaler] = None
) -> Tuple[np.ndarray, np.ndarray, Optional[MinMaxScaler]]:
    """
    A simpler version of create_windows that uses numpy's stride_tricks.
    Data is first normalized using sklearn's MinMaxScaler, then windowed.
    
    Args:
        data: pandas DataFrame or numpy array with shape (num_samples, feature_len)
        window_length: Number of time steps in each window
        stride: Number of time steps to move between windows (default: 1)
        label_column: Name of the column containing class labels (default: 'class')
        scaler: Optional pre-fitted MinMaxScaler. If None, a new one will be created
    
    Returns:
        Tuple of (windows, labels, scaler):
            - windows: numpy.ndarray with shape (num_windows, window_length, feature_length)
            - labels: numpy.ndarray with shape (num_windows,)
            - scaler: fitted MinMaxScaler for inverse transform if needed
    """
    if isinstance(data, pd.DataFrame):
        # Extract labels
        if label_column not in data.columns:
            raise ValueError(f"Label column '{label_column}' not found in data")
        labels = data[label_column].values
        # Get features
        data_features = data.drop(columns=[label_column])
        
        # Normalize features using MinMaxScaler
        if scaler is None:
            scaler = MinMaxScaler()
            data_normalized = scaler.fit_transform(data_features)
        else:
            data_normalized = scaler.transform(data_features)
    else:
        raise ValueError("When using sliding_window, data must be a pandas DataFrame with a label column")
    
    # Get dimensions
    num_samples, feature_length = data_normalized.shape
    
    # Calculate number of windows
    num_windows = (num_samples - window_length) // stride + 1
    
    if num_windows <= 0:
        raise ValueError(f"No windows could be created with the given parameters.")
    
    # Use stride tricks to create windows efficiently
    windows = np.lib.stride_tricks.sliding_window_view(
        data_normalized, 
        window_shape=(window_length, feature_length)
    )[::stride, 0, :, :]
    
    # Get labels for each window (using start of window)
    window_labels = labels[::stride][:num_windows]
    
    return windows, window_labels, scaler


def normalize_windows(windows: np.ndarray, method: str = 'minmax') -> np.ndarray:
    """
    Normalize each window independently.
    Note: This function is kept for additional normalization needs after windowing.
    
    Args:
        windows: numpy array with shape (num_windows, window_length, feature_length)
        method: Normalization method ('minmax', 'zscore', or 'robust')
    
    Returns:
        Normalized windows with the same shape
    """
    normalized = np.zeros_like(windows)
    
    for i in range(windows.shape[0]):
        window = windows[i]
        
        if method == 'minmax':
            # Min-max normalization (scale to [0, 1])
            min_vals = np.min(window, axis=0, keepdims=True)
            max_vals = np.max(window, axis=0, keepdims=True)
            denominator = max_vals - min_vals
            # Avoid division by zero
            denominator[denominator == 0] = 1
            normalized[i] = (window - min_vals) / denominator
            
        elif method == 'zscore':
            # Z-score normalization (mean=0, std=1)
            mean = np.mean(window, axis=0, keepdims=True)
            std = np.std(window, axis=0, keepdims=True)
            # Avoid division by zero
            std[std == 0] = 1
            normalized[i] = (window - mean) / std
            
        elif method == 'robust':
            # Robust scaling using median and IQR
            median = np.median(window, axis=0, keepdims=True)
            q75, q25 = np.percentile(window, [75, 25], axis=0, keepdims=True)
            iqr = q75 - q25
            # Avoid division by zero
            iqr[iqr == 0] = 1
            normalized[i] = (window - median) / iqr
            
        else:
            raise ValueError(f"Unknown normalization method: {method}")
    
    return normalized


if __name__ == "__main__":
    from config_utils import load_config, get_default_config, print_config_summary
    
    # Load and validate configuration
    config = load_config('config.yaml')
    if config is None:
        print("Using default configuration")
        config = get_default_config()
    
    # Print configuration summary
    print_config_summary(config)
    
    if config is not None:
        # Set random seed for reproducibility
        np.random.seed(config['data']['sample_generation']['random_seed'])
        
        # Create sample data
        num_samples = config['data']['sample_generation']['num_samples']
        sample_data = pd.DataFrame({
            'sensor1': np.random.random(num_samples),
            'sensor2': np.random.random(num_samples),
            'sensor3': np.random.random(num_samples),
            'class': np.random.randint(0, 2, num_samples)
        })
        
        # Get parameters from config
        window_length = config['window']['length']
        window_method = config['window']['method']
        overlap = config['window']['overlap']
        step_size = config['window']['step_size']
        feature_columns = config['data']['feature_columns']
        label_column = config['data']['label_column']
        window_norm_method = config['normalization']['window_normalization']
        
        print(f"\nProcessing with parameters:")
        print(f"Window length: {window_length}")
        print(f"Window method: {window_method}")
        print(f"Feature columns: {feature_columns}")
        
        # Create windows based on method
        if window_method == 'overlap':
            windows, labels, scaler = create_windows(
                sample_data,
                window_length=window_length,
                overlap=overlap,
                feature_columns=feature_columns,
                label_column=label_column
            )
            print(f"\nCreated windows with {overlap*100}% overlap:")
        else:
            windows, labels, scaler = create_windows(
                sample_data,
                window_length=window_length,
                step_size=step_size,
                feature_columns=feature_columns,
                label_column=label_column
            )
            print(f"\nCreated windows with step size {step_size}:")
        
        print(f"Window shape: {windows.shape}")
        print(f"Labels shape: {labels.shape}")
        
        # Apply additional window normalization if specified
        if window_norm_method != 'none':
            norm_windows = normalize_windows(windows, method=window_norm_method)
            print(f"\nApplied {window_norm_method} normalization:")
            print(f"Normalized window shape: {norm_windows.shape}")
        
        # Verify normalization
        print("\nVerifying original MinMaxScaler normalization:")
        print(f"Min values: {np.min(windows[0])}")
        print(f"Max values: {np.max(windows[0])}")
        
        # Example of inverse transform
        original_data = scaler.inverse_transform(windows[0])
        print("\nExample of inverse transform for first window:")
        print(f"Original shape: {original_data.shape}")
        
        # Print label distribution
        unique_labels, counts = np.unique(labels, return_counts=True)
        print("\nLabel distribution:")
        for label, count in zip(unique_labels, counts):
            print(f"Label {label}: {count} windows ({count/len(labels)*100:.2f}%)")
    else:
        print("Using default parameters as configuration could not be loaded")
        # Original example code here...

    # Try with MaFaulDa dataset
    # Run under higher level directory
    import yaml
    with open('config.yaml', 'r') as file:
        config = yaml.safe_load(file)
    # Load data
    ds_config = config['datasets'][config['active_dataset']]
    print(f"Loading dataset from {ds_config['path']}")
    data = pd.read_csv(ds_config['path'])
    data_columns = ds_config['features']
    norm_windows, labels, _ = create_windows(
        data,
        window_length=config['data_processing']['window']['length'],
        step_size=config['data_processing']['window']['step_size'],
        overlap=None,
        feature_columns=data_columns,
        label_column=ds_config['label_column'],
    )
    print(norm_windows[0])
    print(norm_windows.shape)
