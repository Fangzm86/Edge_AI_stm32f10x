"""
ML Pipeline Package
==================

A comprehensive machine learning pipeline for embedded systems, providing tools
for data processing, model development, and deployment.

Main Components
--------------
- Data Processing: Tools for processing time series data
- Models: Deep learning models optimized for embedded systems
- Deployment: Utilities for deploying models to embedded targets

Example Usage
------------
>>> from ml_pipeline.data_processing import create_windows, normalize_windows
>>> from ml_pipeline.models import create_model
>>> from ml_pipeline.models import convert_to_tflite
>>>
>>> # Process data
>>> windows, labels, scaler = create_windows(
...     data,
...     window_length=128,
...     overlap=0.5
... )
>>> normalized_windows = normalize_windows(windows)
>>>
>>> # Create and train model
>>> model = create_model(config, input_shape=(128, 6), num_classes=3)
>>> model.fit(normalized_windows, labels, epochs=100)
>>>
>>> # Convert model for deployment
>>> tflite_model, tflite_path = convert_to_tflite(
...     model,
...     quantize='float16',
...     optimize=True
... )

Package Structure
---------------
ml_pipeline/
    ├── data_processing/    # Data processing utilities
    │   ├── process_data.py # Command-line data processing tool
    │   ├── config.yaml    # Data processing configuration
    │   └── ...
    │
    ├── models/            # Model definitions and utilities
    │   ├── model.py       # Model architectures
    │   ├── train_model.py # Command-line training tool
    │   ├── model_config.yaml # Model configuration
    │   └── ...
    │
    └── ...

Command-line Tools
----------------
- process_data.py: Process time series data for machine learning
- train_model.py: Train models with various architectures
- evaluate_model.py: Evaluate trained models
- convert_model.py: Convert models for embedded deployment

Configuration
------------
The package uses YAML configuration files for both data processing and model
parameters. Default configurations are provided but can be overridden:

- data_processing/config.yaml: Data processing parameters
- models/model_config.yaml: Model architecture and training parameters
"""

__version__ = '0.1.0'

# Import subpackages
from . import data_processing
from . import models

# Define public API
__all__ = [
    'data_processing',
    'models'
]

# Package metadata
__author__ = 'Tencent Cloud'
__email__ = 'support@tencentcloud.com'
__description__ = 'Machine learning pipeline for embedded systems'
__url__ = 'https://github.com/TencentCloud/ml-pipeline'
__license__ = 'Apache License 2.0'

def get_version():
    """Get the version of the package."""
    return __version__

def get_package_info():
    """Get package information."""
    return {
        'name': 'ml_pipeline',
        'version': __version__,
        'author': __author__,
        'email': __email__,
        'description': __description__,
        'url': __url__,
        'license': __license__,
        'subpackages': {
            'data_processing': data_processing.get_package_info(),
            'models': models.get_package_info()
        }
    }

def get_config_paths():
    """Get paths to all configuration files."""
    return {
        'data_processing': data_processing.get_config_path(),
        'models': models.get_config_path()
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

# Print welcome message when package is imported
logger.info(f"ML Pipeline v{__version__} loaded")
logger.info("Use help(ml_pipeline) for package documentation")