"""
ML Pipeline Data Processing Package
=================================

This package provides tools and utilities for processing time series data,
particularly focused on sensor data for embedded machine learning applications.

Main Components
--------------
- Data loading and preprocessing
- Window creation and manipulation
- Data normalization
- Data augmentation
- Feature extraction
- Configuration management

Example Usage
------------
>>> from ml_pipeline.data_processing import create_windows, normalize_windows
>>> from ml_pipeline.data_processing import load_config
>>>
>>> # Load configuration
>>> config = load_config('config.yaml')
>>>
>>> # Create windows from data
>>> windows, labels, scaler = create_windows(
...     data,
...     window_length=128,
...     overlap=0.5,
...     feature_columns=['accel_x', 'accel_y', 'accel_z']
... )
>>>
>>> # Normalize windows
>>> normalized_windows = normalize_windows(windows, method='zscore')
"""

__version__ = '0.1.0'

from .data_processing import (
    create_windows,
    normalize_data,
    normalize_windows,
    augment_windows,
    extract_features,
    save_dataset,
    load_dataset,
    split_data,
    generate_synthetic_data
)

from .config_utils import (
    load_config,
    get_default_config,
    print_config_summary,
    save_config,
    merge_configs,
    update_config_from_args
)

# Define public API
__all__ = [
    # Data processing functions
    'create_windows',
    'normalize_data',
    'normalize_windows',
    'augment_windows',
    'extract_features',
    'save_dataset',
    'load_dataset',
    'split_data',
    'generate_synthetic_data',
    
    # Configuration functions
    'load_config',
    'get_default_config',
    'print_config_summary',
    'save_config',
    'merge_configs',
    'update_config_from_args'
]

# Package metadata
__author__ = 'Tencent Cloud'
__email__ = 'support@tencentcloud.com'
__description__ = 'Time series data processing tools for embedded machine learning'
__url__ = 'https://github.com/TencentCloud/ml-pipeline'
__license__ = 'Apache License 2.0'

def get_version():
    """Get the version of the package."""
    return __version__

def get_config_path():
    """Get the path to the default configuration file."""
    import os
    return os.path.join(os.path.dirname(__file__), 'config.yaml')

def get_package_info():
    """Get package information."""
    return {
        'name': 'ml_pipeline.data_processing',
        'version': __version__,
        'author': __author__,
        'email': __email__,
        'description': __description__,
        'url': __url__,
        'license': __license__
    }

# Optional: Set up logging
import logging

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

# Create console handler with formatting
if not logger.handlers:
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

# Clean up namespace
del logging