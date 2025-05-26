"""
Time series data processing utilities for ML Pipeline.
"""

import os
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Union, Any
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler

def create_windows(
    data: pd.DataFrame,
    window_length: int,
    overlap: float = 0.5,
    step_size: Optional[int] = None,
    feature_columns: List[str] = None,
    label_column: str = 'label',
    method: str = 'overlap'
) -> Tuple[np.ndarray, np.ndarray, Any]:
    """
    Create windows from time series data.
    
    Args:
        data: DataFrame containing time series data
        window_length: Length of each window
        overlap: Overlap between consecutive windows (0.0 to 1.0)
        step_size: Step size between consecutive windows (used when method is 'step')
        feature_columns: List of feature column names
        label_column: Name of the label column
        method: Windowing method ('overlap' or 'step')
    
    Returns:
        windows: Array of windows with shape (n_windows, window_length, n_features)
        labels: Array of labels with shape (n_windows,)
        scaler: Fitted scaler object
    """
    # Check inputs
    if feature_columns is None:
        feature_columns = list(data.columns)
        if label_column in feature_columns:
            feature_columns.remove(label_column)
    
    # Extract features and labels
    features = data[feature_columns].values
    labels = data[label_column].values if label_column in data.columns else None
    
    # Create scaler
    scaler = StandardScaler()
    scaler.fit(features)
    
    # Calculate step size
    if method == 'overlap':
        if step_size is None:
            step_size = int(window_length * (1 - overlap))
    else:
        if step_size is None:
            step_size = window_length
    
    # Create windows
    n_samples = len(features)
    n_features = len(feature_columns)
    
    # Calculate number of windows
    n_windows = (n_samples - window_length) // step_size + 1
    
    # Initialize arrays
    windows = np.zeros((n_windows, window_length, n_features))
    window_labels = np.zeros(n_windows)
    
    # Fill arrays
    for i in range(n_windows):
        start_idx = i * step_size
        end_idx = start_idx + window_length
        
        windows[i] = features[start_idx:end_idx]
        
        if labels is not None:
            # Use most frequent label in window
            window_labels[i] = np.bincount(labels[start_idx:end_idx].astype(int)).argmax()
    
    return windows, window_labels, scaler

def normalize_data(
    data: np.ndarray,
    method: str = 'standard',
    scaler: Any = None
) -> Tuple[np.ndarray, Any]:
    """
    Normalize time series data.
    
    Args:
        data: Data to normalize
        method: Normalization method ('standard', 'minmax', or 'robust')
        scaler: Pre-fitted scaler (optional)
    
    Returns:
        normalized_data: Normalized data
        scaler: Fitted scaler object
    """
    # Reshape data if needed
    original_shape = data.shape
    if len(original_shape) > 2:
        data_2d = data.reshape(-1, original_shape[-1])
    else:
        data_2d = data
    
    # Create scaler if not provided
    if scaler is None:
        if method == 'standard':
            scaler = StandardScaler()
        elif method == 'minmax':
            scaler = MinMaxScaler()
        elif method == 'robust':
            scaler = RobustScaler()
        else:
            raise ValueError(f"Unknown normalization method: {method}")
        
        # Fit scaler
        scaler.fit(data_2d)
    
    # Transform data
    normalized_data_2d = scaler.transform(data_2d)
    
    # Reshape back to original shape
    if len(original_shape) > 2:
        normalized_data = normalized_data_2d.reshape(original_shape)
    else:
        normalized_data = normalized_data_2d
    
    return normalized_data, scaler

def normalize_windows(
    windows: np.ndarray,
    method: str = 'zscore'
) -> np.ndarray:
    """
    Normalize each window individually.
    
    Args:
        windows: Windows to normalize with shape (n_windows, window_length, n_features)
        method: Normalization method ('zscore', 'minmax', 'robust', or 'none')
    
    Returns:
        normalized_windows: Normalized windows
    """
    if method == 'none':
        return windows
    
    n_windows, window_length, n_features = windows.shape
    normalized_windows = np.zeros_like(windows)
    
    for i in range(n_windows):
        for j in range(n_features):
            window = windows[i, :, j]
            
            if method == 'zscore':
                # Z-score normalization
                mean = np.mean(window)
                std = np.std(window)
                if std > 0:
                    normalized_windows[i, :, j] = (window - mean) / std
                else:
                    normalized_windows[i, :, j] = window - mean
            
            elif method == 'minmax':
                # Min-max normalization
                min_val = np.min(window)
                max_val = np.max(window)
                if max_val > min_val:
                    normalized_windows[i, :, j] = (window - min_val) / (max_val - min_val)
                else:
                    normalized_windows[i, :, j] = np.zeros_like(window)
            
            elif method == 'robust':
                # Robust normalization
                median = np.median(window)
                q1 = np.percentile(window, 25)
                q3 = np.percentile(window, 75)
                iqr = q3 - q1
                if iqr > 0:
                    normalized_windows[i, :, j] = (window - median) / iqr
                else:
                    normalized_windows[i, :, j] = window - median
    
    return normalized_windows

def augment_windows(
    windows: np.ndarray,
    labels: np.ndarray,
    config: Dict[str, Any]
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Apply data augmentation to windows.
    
    Args:
        windows: Windows to augment with shape (n_windows, window_length, n_features)
        labels: Window labels with shape (n_windows,)
        config: Augmentation configuration
    
    Returns:
        augmented_windows: Augmented windows
        augmented_labels: Augmented labels
    """
    if not config.get('enabled', False):
        return windows, labels
    
    n_windows, window_length, n_features = windows.shape
    augmented_windows = [windows]
    augmented_labels = [labels]
    
    # Jitter (add noise)
    if config.get('jitter_sigma', 0) > 0:
        jitter_sigma = config['jitter_sigma']
        jitter_windows = windows + np.random.normal(0, jitter_sigma, windows.shape)
        augmented_windows.append(jitter_windows)
        augmented_labels.append(labels)
    
    # Scaling (multiply by random factor)
    if config.get('scaling_sigma', 0) > 0:
        scaling_sigma = config['scaling_sigma']
        scaling_factors = np.random.normal(1, scaling_sigma, (n_windows, 1, n_features))
        scaling_windows = windows * scaling_factors
        augmented_windows.append(scaling_windows)
        augmented_labels.append(labels)
    
    # Rotation (for multi-dimensional data)
    if config.get('rotation_sigma', 0) > 0 and n_features >= 2:
        rotation_sigma = config['rotation_sigma']
        rotation_windows = np.zeros_like(windows)
        
        for i in range(n_windows):
            # Apply small random rotation to each window
            # This is a simplified version that works for 2D or 3D data
            if n_features == 2:
                # 2D rotation
                theta = np.random.normal(0, rotation_sigma)
                c, s = np.cos(theta), np.sin(theta)
                R = np.array([[c, -s], [s, c]])
                
                for j in range(window_length):
                    rotation_windows[i, j] = np.dot(windows[i, j], R)
            
            elif n_features >= 3:
                # 3D rotation (simplified to rotation around z-axis)
                theta = np.random.normal(0, rotation_sigma)
                c, s = np.cos(theta), np.sin(theta)
                R = np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])
                
                for j in range(window_length):
                    rotation_windows[i, j, :3] = np.dot(windows[i, j, :3], R)
                    if n_features > 3:
                        rotation_windows[i, j, 3:] = windows[i, j, 3:]
        
        augmented_windows.append(rotation_windows)
        augmented_labels.append(labels)
    
    # Permutation (shuffle segments)
    if config.get('permutation_segments', 0) > 0:
        n_segments = config['permutation_segments']
        segment_length = window_length // n_segments
        
        if segment_length > 1:
            permutation_windows = np.zeros_like(windows)
            
            for i in range(n_windows):
                # Create segments
                segments = []
                for j in range(n_segments):
                    start_idx = j * segment_length
                    end_idx = start_idx + segment_length
                    segments.append(windows[i, start_idx:end_idx])
                
                # Shuffle segments
                np.random.shuffle(segments)
                
                # Reconstruct window
                for j in range(n_segments):
                    start_idx = j * segment_length
                    end_idx = start_idx + segment_length
                    permutation_windows[i, start_idx:end_idx] = segments[j]
                
                # Copy remaining samples
                if window_length % n_segments != 0:
                    permutation_windows[i, n_segments*segment_length:] = \
                        windows[i, n_segments*segment_length:]
            
            augmented_windows.append(permutation_windows)
            augmented_labels.append(labels)
    
    # Time warping
    if config.get('time_warping', False):
        # Simple time warping by linear interpolation
        warping_windows = np.zeros_like(windows)
        
        for i in range(n_windows):
            # Generate random warping path
            src_points = np.linspace(0, window_length-1, 5)
            dst_points = src_points + np.random.normal(0, window_length*0.05, 5)
            dst_points[0] = 0
            dst_points[-1] = window_length-1
            
            # Apply warping to each feature
            for j in range(n_features):
                warping_windows[i, :, j] = np.interp(
                    np.arange(window_length),
                    dst_points,
                    np.interp(src_points, np.arange(window_length), windows[i, :, j])
                )
        
        augmented_windows.append(warping_windows)
        augmented_labels.append(labels)
    
    # Combine all augmented data
    combined_windows = np.vstack(augmented_windows)
    combined_labels = np.hstack(augmented_labels)
    
    return combined_windows, combined_labels

def extract_features(
    windows: np.ndarray,
    feature_types: List[str] = ['statistical', 'frequency']
) -> np.ndarray:
    """
    Extract features from windows.
    
    Args:
        windows: Windows with shape (n_windows, window_length, n_features)
        feature_types: Types of features to extract
    
    Returns:
        features: Extracted features
    """
    n_windows, window_length, n_features = windows.shape
    feature_list = []
    
    # Statistical features
    if 'statistical' in feature_types:
        # Initialize array for statistical features
        stat_features = np.zeros((n_windows, n_features * 7))
        
        for i in range(n_windows):
            for j in range(n_features):
                window = windows[i, :, j]
                
                # Calculate statistical features
                mean = np.mean(window)
                std = np.std(window)
                min_val = np.min(window)
                max_val = np.max(window)
                median = np.median(window)
                skew = np.mean(((window - mean) / std) ** 3) if std > 0 else 0
                kurtosis = np.mean(((window - mean) / std) ** 4) - 3 if std > 0 else 0
                
                # Store features
                idx = j * 7
                stat_features[i, idx] = mean
                stat_features[i, idx+1] = std
                stat_features[i, idx+2] = min_val
                stat_features[i, idx+3] = max_val
                stat_features[i, idx+4] = median
                stat_features[i, idx+5] = skew
                stat_features[i, idx+6] = kurtosis
        
        feature_list.append(stat_features)
    
    # Frequency domain features
    if 'frequency' in feature_types:
        # Initialize array for frequency features
        freq_features = np.zeros((n_windows, n_features * 3))
        
        for i in range(n_windows):
            for j in range(n_features):
                window = windows[i, :, j]
                
                # Apply FFT
                fft_values = np.abs(np.fft.rfft(window))
                fft_freq = np.fft.rfftfreq(window_length)
                
                # Calculate frequency features
                dominant_freq = fft_freq[np.argmax(fft_values)]
                spectral_energy = np.sum(fft_values**2) / len(fft_values)
                spectral_entropy = -np.sum((fft_values**2 / np.sum(fft_values**2)) * 
                                          np.log2(fft_values**2 / np.sum(fft_values**2) + 1e-10))
                
                # Store features
                idx = j * 3
                freq_features[i, idx] = dominant_freq
                freq_features[i, idx+1] = spectral_energy
                freq_features[i, idx+2] = spectral_entropy
        
        feature_list.append(freq_features)
    
    # Combine all features
    if feature_list:
        combined_features = np.hstack(feature_list)
        return combined_features
    else:
        return np.zeros((n_windows, 0))

def save_dataset(
    windows: np.ndarray,
    labels: np.ndarray,
    output_path: str,
    scaler: Any = None
) -> None:
    """
    Save processed dataset to file.
    
    Args:
        windows: Windows with shape (n_windows, window_length, n_features)
        labels: Labels with shape (n_windows,)
        output_path: Path to save the dataset
        scaler: Fitted scaler object
    """
    # Create output directory if needed
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    
    # Save data
    if scaler is not None:
        np.savez(output_path, windows=windows, labels=labels, scaler=scaler)
    else:
        np.savez(output_path, windows=windows, labels=labels)

def load_dataset(
    input_path: str
) -> Tuple[np.ndarray, np.ndarray, Any]:
    """
    Load processed dataset from file.
    
    Args:
        input_path: Path to the dataset
    
    Returns:
        windows: Windows with shape (n_windows, window_length, n_features)
        labels: Labels with shape (n_windows,)
        scaler: Fitted scaler object
    """
    # Load data
    data = np.load(input_path, allow_pickle=True)
    windows = data['windows']
    labels = data['labels']
    
    # Load scaler if available
    scaler = data.get('scaler', None)
    
    return windows, labels, scaler

def split_data(
    windows: np.ndarray,
    labels: np.ndarray,
    train_ratio: float = 0.7,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
    random_seed: int = 42
) -> Tuple[
    np.ndarray, np.ndarray,  # Training data
    np.ndarray, np.ndarray,  # Validation data
    np.ndarray, np.ndarray   # Test data
]:
    """
    Split data into training, validation, and test sets.
    
    Args:
        windows: Windows with shape (n_windows, window_length, n_features)
        labels: Labels with shape (n_windows,)
        train_ratio: Ratio of training data
        val_ratio: Ratio of validation data
        test_ratio: Ratio of test data
        random_seed: Random seed for reproducibility
    
    Returns:
        x_train: Training windows
        y_train: Training labels
        x_val: Validation windows
        y_val: Validation labels
        x_test: Test windows
        y_test: Test labels
    """
    # Check ratios
    if train_ratio + val_ratio + test_ratio != 1.0:
        raise ValueError("Ratios must sum to 1.0")
    
    # Set random seed
    np.random.seed(random_seed)
    
    # Shuffle data
    indices = np.arange(len(windows))
    np.random.shuffle(indices)
    windows = windows[indices]
    labels = labels[indices]
    
    # Calculate split indices
    n_samples = len(windows)
    train_idx = int(n_samples * train_ratio)
    val_idx = train_idx + int(n_samples * val_ratio)
    
    # Split data
    x_train = windows[:train_idx]
    y_train = labels[:train_idx]
    
    x_val = windows[train_idx:val_idx]
    y_val = labels[train_idx:val_idx]
    
    x_test = windows[val_idx:]
    y_test = labels[val_idx:]
    
    return x_train, y_train, x_val, y_val, x_test, y_test

def generate_synthetic_data(
    n_samples: int,
    n_features: int,
    n_classes: int,
    window_length: int,
    random_seed: int = 42
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Generate synthetic time series data for testing.
    
    Args:
        n_samples: Number of samples to generate
        n_features: Number of features
        n_classes: Number of classes
        window_length: Length of each window
        random_seed: Random seed for reproducibility
    
    Returns:
        windows: Generated windows
        labels: Generated labels
    """
    # Set random seed
    np.random.seed(random_seed)
    
    # Initialize arrays
    windows = np.zeros((n_samples, window_length, n_features))
    labels = np.zeros(n_samples)
    
    # Generate data for each class
    samples_per_class = n_samples // n_classes
    
    for c in range(n_classes):
        # Generate class-specific parameters
        frequencies = np.random.uniform(0.1, 0.5, n_features)
        amplitudes = np.random.uniform(0.5, 2.0, n_features)
        phases = np.random.uniform(0, 2*np.pi, n_features)
        
        # Generate samples for this class
        start_idx = c * samples_per_class
        end_idx = start_idx + samples_per_class
        
        for i in range(start_idx, min(end_idx, n_samples)):
            # Generate time series for each feature
            for j in range(n_features):
                t = np.linspace(0, 2*np.pi, window_length)
                signal = amplitudes[j] * np.sin(frequencies[j] * t + phases[j])
                noise = np.random.normal(0, 0.2, window_length)
                windows[i, :, j] = signal + noise
            
            # Set label
            labels[i] = c
    
    return windows, labels

if __name__ == "__main__":
    # Example usage
    print("Generating synthetic data...")
    windows, labels = generate_synthetic_data(
        n_samples=100,
        n_features=3,
        n_classes=2,
        window_length=128
    )
    
    print(f"Generated {len(windows)} windows with shape {windows.shape}")
    
    # Normalize windows
    print("Normalizing windows...")
    normalized_windows = normalize_windows(windows)
    
    # Augment data
    print("Augmenting data...")
    augmentation_config = {
        'enabled': True,
        'jitter_sigma': 0.05,
        'scaling_sigma': 0.1,
        'rotation_sigma': 0.1
    }
    augmented_windows, augmented_labels = augment_windows(
        normalized_windows, labels, augmentation_config
    )
    
    print(f"Augmented data: {len(augmented_windows)} windows")
    
    # Extract features
    print("Extracting features...")
    features = extract_features(augmented_windows)
    
    print(f"Extracted {features.shape[1]} features")
    
    # Save dataset
    print("Saving dataset...")
    save_dataset(augmented_windows, augmented_labels, "example_data.npz")
    
    print("Done!")