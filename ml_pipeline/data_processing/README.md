# Time Series Data Processing

This module provides tools for processing time series data from sensors, creating windowed datasets, and preparing data for model training. It is designed to work with the model training pipeline for embedded systems.

## Table of Contents

- [Installation](#installation)
- [Dependencies](#dependencies)
- [Usage](#usage)
- [Data Format](#data-format)
- [Configuration](#configuration)
- [Windowing Methods](#windowing-methods)
- [Normalization](#normalization)
- [API Reference](#api-reference)
- [FAQ](#faq)

## Installation

```bash
# From the project root directory
pip install -e ml_pipeline/
```

## Dependencies

- NumPy
- Pandas
- Scikit-learn
- PyYAML

## Usage

### Basic Usage

```python
from ml_pipeline.data_processing import create_windows, normalize_windows, load_config

# Load configuration
config = load_config('config.yaml')

# Load your data (replace with your data loading code)
import pandas as pd
data = pd.read_csv('sensor_data.csv')

# Create windows
windows, labels, scaler = create_windows(
    data,
    window_length=config['window']['length'],
    overlap=config['window']['overlap'],
    feature_columns=config['data']['feature_columns'],
    label_column=config['data']['label_column']
)

# Apply additional normalization if needed
normalized_windows = normalize_windows(
    windows,
    method=config['normalization']['window_normalization']
)

print(f"Created {len(windows)} windows with shape {windows.shape}")
```

### Command-line Processing

```bash
# Process data using the command-line tool
python process_data.py --input sensor_data.csv \
                       --output processed_data.npz \
                       --config config.yaml
```

## Data Format

The module expects input data in CSV format with the following structure:

- Each row represents a time point
- Columns represent different sensor measurements or features
- One column should contain class labels (for supervised learning)

Example:

```
timestamp,accel_x,accel_y,accel_z,gyro_x,gyro_y,gyro_z,label
1623456789,0.12,0.34,-0.98,0.01,0.02,-0.01,0
1623456790,0.15,0.32,-0.97,0.02,0.03,-0.02,0
1623456791,0.67,0.89,-0.45,1.23,0.98,-0.65,1
...
```

## Configuration

The module uses a YAML configuration file to specify data processing parameters:

```yaml
# Data configuration
data:
  feature_columns: ['accel_x', 'accel_y', 'accel_z', 'gyro_x', 'gyro_y', 'gyro_z']
  label_column: 'label'
  sample_generation:
    num_samples: 1000
    random_seed: 42
    class_distribution: [0.5, 0.5]

# Window configuration
window:
  length: 128
  method: 'overlap'  # 'overlap' or 'step'
  overlap: 0.5       # Used when method is 'overlap'
  step_size: 32      # Used when method is 'step'

# Normalization configuration
normalization:
  method: 'standard'  # 'standard', 'minmax', or 'robust'
  window_normalization: 'none'  # 'none', 'zscore', 'minmax', or 'robust'
```

## Windowing Methods

The module supports two windowing methods:

### Overlap Method

Windows are created with a specified overlap percentage. For example, with a window length of 100 and an overlap of 0.5, each window will share 50% of its data with the next window.

```python
windows, labels, scaler = create_windows(
    data,
    window_length=100,
    overlap=0.5,
    feature_columns=['accel_x', 'accel_y', 'accel_z'],
    label_column='label'
)
```

### Step Method

Windows are created with a fixed step size between consecutive windows. For example, with a window length of 100 and a step size of 25, each window will start 25 time points after the previous one.

```python
windows, labels, scaler = create_windows(
    data,
    window_length=100,
    step_size=25,
    feature_columns=['accel_x', 'accel_y', 'accel_z'],
    label_column='label'
)
```

## Normalization

The module supports several normalization methods:

### Feature Normalization

Applied to each feature across the entire dataset:

- **Standard**: Zero mean and unit variance
- **MinMax**: Scale to range [0, 1]
- **Robust**: Scale using median and interquartile range

### Window Normalization

Applied to each window individually:

- **Z-score**: Zero mean and unit variance for each window
- **MinMax**: Scale each window to range [0, 1]
- **Robust**: Scale each window using median and interquartile range

## API Reference

### `create_windows(data, window_length, overlap=None, step_size=None, feature_columns=None, label_column=None, scaler_path=None)`

Creates windowed dataset from time series data.

**Parameters:**
- `data`: DataFrame containing time series data
- `window_length`: Number of time points in each window
- `overlap`: Overlap between consecutive windows (0.0 to 1.0)
- `step_size`: Step size between consecutive windows (alternative to overlap)
- `feature_columns`: List of column names to use as features
- `label_column`: Column name containing class labels
- `scaler_path`: Path to saved scaler (optional)

**Returns:**
- `windows`: Numpy array of shape (n_windows, window_length, n_features)
- `labels`: Numpy array of shape (n_windows,)
- `scaler`: Fitted scaler object

### `normalize_windows(windows, method='zscore')`

Applies normalization to each window individually.

**Parameters:**
- `windows`: Numpy array of shape (n_windows, window_length, n_features)
- `method`: Normalization method ('zscore', 'minmax', or 'robust')

**Returns:**
- Normalized windows array

### `load_config(config_path)`

Loads configuration from YAML file.

**Parameters:**
- `config_path`: Path to configuration file

**Returns:**
- Configuration dictionary

## FAQ

### Q: How do I choose the right window length?

A: The optimal window length depends on your specific application:
- For activity recognition: typically 1-5 seconds of data (e.g., 50-250 samples at 50Hz)
- For gesture recognition: typically 0.5-2 seconds
- For vibration analysis: depends on the frequency of interest

Try different window lengths and evaluate model performance.

### Q: What overlap value should I use?

A: Higher overlap (e.g., 0.75) provides more training data but with higher redundancy. Lower overlap (e.g., 0.25) provides less redundant data but might miss important transitions. A common starting point is 0.5 (50% overlap).

### Q: Should I normalize my data?

A: Yes, normalization is generally recommended for machine learning models. For time series data:
1. First normalize each feature across the entire dataset
2. Optionally apply window normalization if your application involves comparing patterns regardless of amplitude

### Q: How do I handle missing data?

A: The module doesn't handle missing data automatically. You should preprocess your data to:
1. Interpolate missing values
2. Drop windows with too many missing values
3. Use a mask channel to indicate missing data

### Q: Can I use this for multivariate time series?

A: Yes, the module is designed for multivariate time series. Simply include all relevant features in the `feature_columns` parameter.

### Q: How do I save processed data for later use?

A: Use the `save_dataset` and `load_dataset` functions:

```python
from ml_pipeline.data_processing import save_dataset, load_dataset

# Save processed data
save_dataset(windows, labels, 'processed_data.npz', scaler=scaler)

# Load processed data
windows, labels, scaler = load_dataset('processed_data.npz')
```

## Contributing

Feel free to submit issues and enhancement requests!

## License

This project is licensed under the MIT License - see the LICENSE file for details.