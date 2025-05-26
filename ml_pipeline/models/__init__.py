"""
ML Pipeline Models Package
=========================

This package provides tools and utilities for creating, training, evaluating,
and converting machine learning models for embedded systems.

Main Components
--------------
- Model architecture definitions
- Model training utilities
- Model evaluation tools
- Model conversion for embedded deployment

Example Usage
------------
>>> from ml_pipeline.models import create_model
>>> from ml_pipeline.models import save_model_with_metadata
>>>
>>> # Create model
>>> model = create_model(
...     config,
...     input_shape=(128, 6),
...     num_classes=3
... )
>>>
>>> # Train model
>>> model.fit(x_train, y_train, epochs=100, validation_data=(x_val, y_val))
>>>
>>> # Save model with metadata
>>> save_model_with_metadata(
...     model,
...     'output_dir',
...     'model_name',
...     config,
...     input_shape=(128, 6),
...     num_classes=3
... )
"""

__version__ = '0.1.0'

from .model import (
    create_model,
    create_lstm_model,
    create_cnn_model,
    create_hybrid_model,
    create_tcn_model,
    create_transformer_model,
    create_resnet_model,
    get_optimizer,
    get_learning_rate_scheduler,
    save_model_with_metadata,
    load_model_with_metadata,
    get_model_summary,
    get_model_size
)

from .convert_model import (
    convert_to_tflite,
    generate_c_array,
    analyze_model,
    convert_model,
    get_representative_dataset
)

# Define public API
__all__ = [
    # Model creation functions
    'create_model',
    'create_lstm_model',
    'create_cnn_model',
    'create_hybrid_model',
    'create_tcn_model',
    'create_transformer_model',
    'create_resnet_model',
    
    # Model training functions
    'get_optimizer',
    'get_learning_rate_scheduler',
    
    # Model saving/loading functions
    'save_model_with_metadata',
    'load_model_with_metadata',
    'get_model_summary',
    'get_model_size',
    
    # Model conversion functions
    'convert_to_tflite',
    'generate_c_array',
    'analyze_model',
    'convert_model',
    'get_representative_dataset'
]

# Package metadata
__author__ = 'Tencent Cloud'
__email__ = 'support@tencentcloud.com'
__description__ = 'Machine learning models for embedded systems'
__url__ = 'https://github.com/TencentCloud/ml-pipeline'
__license__ = 'Apache License 2.0'

def get_version():
    """Get the version of the package."""
    return __version__

def get_config_path():
    """Get the path to the default model configuration file."""
    import os
    return os.path.join(os.path.dirname(__file__), 'model_config.yaml')

def get_package_info():
    """Get package information."""
    return {
        'name': 'ml_pipeline.models',
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