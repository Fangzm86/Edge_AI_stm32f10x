"""
Example usage of data processing functions for time series data.
Demonstrates MinMaxScaler normalization and windowing.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from data_processing import create_windows, sliding_window, normalize_windows

def generate_sample_data(num_samples=1000, num_features=3):
    """
    Generate sample time series data with labels.
    Labels are generated based on the pattern of sensor1:
    - Label 1 when sensor1 > 0
    - Label 0 when sensor1 <= 0
    """
    t = np.linspace(0, 10, num_samples)
    # Generate data with different scales to demonstrate normalization
    sensor1 = 100 * np.sin(2 * np.pi * t) + np.random.normal(0, 10, num_samples)
    sensor2 = np.cos(2 * np.pi * t) + np.random.normal(0, 0.1, num_samples)
    sensor3 = 50 * np.sin(4 * np.pi * t) + np.random.normal(0, 5, num_samples)
    
    data = {
        'sensor1': sensor1,
        'sensor2': sensor2,
        'sensor3': sensor3,
        'class': (sensor1 > 0).astype(int)  # Binary classification based on sensor1
    }
    return pd.DataFrame(data)

def plot_data_comparison(original_data, normalized_data, title_prefix=""):
    """Plot original and normalized data for comparison"""
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))
    
    # Plot original data
    for column in original_data.columns:
        if column != 'class':
            ax1.plot(original_data[column], label=column)
    ax1.set_title(f"{title_prefix}Original Data")
    ax1.legend()
    ax1.grid(True)
    
    # Plot normalized data
    for i in range(normalized_data.shape[1]):
        ax2.plot(normalized_data[:, i], label=f'Normalized {i+1}')
    ax2.set_title(f"{title_prefix}Normalized Data")
    ax2.legend()
    ax2.grid(True)
    
    plt.tight_layout()
    return fig

def plot_windows_with_labels(windows, labels, title="Windowed Data"):
    """Plot the first few windows of data with their labels"""
    num_windows_to_plot = min(5, windows.shape[0])
    fig, axes = plt.subplots(num_windows_to_plot, 1, figsize=(12, 2*num_windows_to_plot))
    fig.suptitle(title)
    
    for i in range(num_windows_to_plot):
        for j in range(windows.shape[2]):  # For each feature
            axes[i].plot(windows[i, :, j], label=f'Feature {j+1}')
        axes[i].set_title(f'Window {i+1} (Label: {labels[i]})')
        axes[i].legend()
        axes[i].grid(True)
        # Add label indicator as background color
        axes[i].set_facecolor(('lightgreen' if labels[i] == 1 else 'lightpink'))
        axes[i].set_alpha(0.3)
    
    plt.tight_layout()
    return fig

def main():
    from config_utils import load_config, get_default_config, print_config_summary
    
    # Load configuration
    config = load_config('config.yaml')
    if config is None:
        print("Using default configuration")
        config = get_default_config()
    
    print_config_summary(config)
    
    # Generate sample data
    print("\nGenerating sample data...")
    data = generate_sample_data(
        num_samples=config['data']['sample_generation']['num_samples'],
        num_features=len(config['data']['feature_columns'])
    )
    print(f"Data shape: {data.shape}")
    
    # Plot original data
    plot_data_comparison(
        data,
        MinMaxScaler().fit_transform(data[config['data']['feature_columns']]),
        "Before Windowing: "
    )
    
    # Get window parameters
    window_length = config['window']['length']
    window_method = config['window']['method']
    
    # Create windows based on configuration
    if window_method == 'overlap':
        print(f"\nCreating windows with {config['window']['overlap']*100}% overlap...")
        windows1, labels1, scaler1 = create_windows(
            data,
            window_length=window_length,
            overlap=config['window']['overlap'],
            feature_columns=config['data']['feature_columns'],
            label_column=config['data']['label_column']
        )
    else:
        print(f"\nCreating windows with step size {config['window']['step_size']}...")
        windows1, labels1, scaler1 = create_windows(
            data,
            window_length=window_length,
            step_size=config['window']['step_size'],
            feature_columns=config['data']['feature_columns'],
            label_column=config['data']['label_column']
        )
    
    print(f"Windows shape: {windows1.shape}")
    print(f"Labels shape: {labels1.shape}")
    
    # Apply additional window normalization if specified
    if config['normalization']['window_normalization'] != 'none':
        norm_windows = normalize_windows(
            windows1,
            method=config['normalization']['window_normalization']
        )
        print(f"\nApplied {config['normalization']['window_normalization']} normalization:")
        print(f"Normalized window shape: {norm_windows.shape}")
    
    # Demonstrate inverse transform
    print("\nDemonstrating inverse transform...")
    original_window = scaler1.inverse_transform(windows1[0])
    print(f"Original data range:")
    for i in range(original_window.shape[1]):
        print(f"Feature {i+1}: [{np.min(original_window[:, i]):.2f}, {np.max(original_window[:, i]):.2f}]")
    
    # Optional: Additional window normalization
    print("\nApplying additional window normalization...")
    norm_windows = normalize_windows(windows1, method='zscore')
    
    # Plot results
    print("\nPlotting results...")
    plot_windows_with_labels(windows1, labels1, "MinMaxScaler Normalized Windows")
    plot_windows_with_labels(norm_windows, labels1, "Additional Z-Score Normalized Windows")
    
    # Print statistics
    print("\nWindow Statistics:")
    print("MinMaxScaler Normalized:")
    print(f"  Min: {np.min(windows1):.3f}")
    print(f"  Max: {np.max(windows1):.3f}")
    print("Z-Score Normalized:")
    print(f"  Mean: {np.mean(norm_windows):.3f}")
    print(f"  Std:  {np.std(norm_windows):.3f}")
    
    plt.show()

if __name__ == "__main__":
    main()