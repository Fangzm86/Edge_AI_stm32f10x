# Time Series Data Processing Module

This module provides tools for processing time series data with configurable parameters.

## Features

- Window creation with flexible parameters
- Integrated MinMaxScaler normalization
- Label handling for supervised learning
- Configuration file support
- Comprehensive error handling
- Unit tests and examples

## Installation

```bash
pip install numpy pandas matplotlib scikit-learn pyyaml
```

## Configuration

The module uses `config.yaml` for parameters:

```yaml
# Data parameters
data:
  feature_columns: [sensor1, sensor2, sensor3]
  label_column: class
  sample_generation:
    num_samples: 1000
    random_seed: 42

# Window parameters  
window:
  length: 128
  method: overlap  # or step_size
  overlap: 0.5
  step_size: 32

# Normalization
normalization:
  use_minmax_scaler: true
  window_normalization: zscore
```

## Usage

### Basic Usage

```python
from data_processing import create_windows
from config_utils import load_config

# Load configuration
config = load_config('config.yaml')

# Process data
windows, labels, scaler = create_windows(
    data,
    window_length=config['window']['length'],
    overlap=config['window']['overlap'],
    feature_columns=config['data']['feature_columns'],
    label_column=config['data']['label_column']
)
```

### Configuration Tools

`config_utils.py` provides:

- `load_config()` - Load and validate config file
- `get_default_config()` - Get default parameters  
- `validate_config()` - Validate configuration
- `print_config_summary()` - Print config overview

## Examples

See `data_processing_example.py` for complete examples.

## Testing

Run tests:
```bash
python -m unittest test_data_processing.py
```

## License

[Add your license here]