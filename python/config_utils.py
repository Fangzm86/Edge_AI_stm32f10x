"""
Configuration utilities for time series data processing.
Handles loading and validation of YAML configuration files.
"""

import yaml
from typing import Dict, Any, Optional
import os

def validate_config(config: Dict[str, Any]) -> bool:
    """
    Validate configuration dictionary.
    
    Args:
        config: Configuration dictionary loaded from YAML
    
    Returns:
        bool: True if configuration is valid, False otherwise
    
    Raises:
        ValueError: If required fields are missing or invalid
    """
    required_sections = ['data', 'window', 'normalization']
    required_fields = {
        'data': ['label_column', 'sample_generation'],
        'window': ['length', 'method'],
        'normalization': ['use_minmax_scaler', 'window_normalization']
    }
    
    # Check required sections
    for section in required_sections:
        if section not in config:
            raise ValueError(f"Missing required section: {section}")
    
    # Check required fields in each section
    for section, fields in required_fields.items():
        for field in fields:
            if field not in config[section]:
                raise ValueError(f"Missing required field '{field}' in section '{section}'")
    
    # Validate specific fields
    if config['window']['method'] not in ['overlap', 'step_size']:
        raise ValueError("Window method must be 'overlap' or 'step_size'")
    
    if config['window']['method'] == 'overlap':
        overlap = config['window'].get('overlap')
        if overlap is None or not 0 <= overlap < 1:
            raise ValueError("Overlap must be between 0.0 and 1.0")
    
    if config['window']['method'] == 'step_size':
        step_size = config['window'].get('step_size')
        if step_size is None or step_size <= 0:
            raise ValueError("Step size must be positive")
    
    if config['normalization']['window_normalization'] not in ['none', 'minmax', 'zscore', 'robust']:
        raise ValueError("Window normalization method must be 'none', 'minmax', 'zscore', or 'robust'")
    
    return True

def load_config(config_path: str) -> Optional[Dict[str, Any]]:
    """
    Load and validate configuration from YAML file.
    
    Args:
        config_path: Path to YAML configuration file
    
    Returns:
        Optional[Dict[str, Any]]: Configuration dictionary if successful, None otherwise
    """
    try:
        # Check if file exists
        if not os.path.exists(config_path):
            print(f"Configuration file not found: {config_path}")
            return None
        
        # Load YAML file
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        
        # Validate configuration
        if validate_config(config):
            print(f"Configuration loaded successfully from {config_path}")
            return config
    
    except yaml.YAMLError as e:
        print(f"Error parsing YAML file: {e}")
    except ValueError as e:
        print(f"Invalid configuration: {e}")
    except Exception as e:
        print(f"Unexpected error loading configuration: {e}")
    
    return None

def get_default_config() -> Dict[str, Any]:
    """
    Get default configuration when config file is not available.
    
    Returns:
        Dict[str, Any]: Default configuration dictionary
    """
    return {
        'data': {
            'feature_columns': ['sensor1', 'sensor2', 'sensor3'],
            'label_column': 'class',
            'sample_generation': {
                'num_samples': 1000,
                'random_seed': 42
            }
        },
        'window': {
            'length': 128,
            'method': 'overlap',
            'overlap': 0.5,
            'step_size': 32
        },
        'normalization': {
            'use_minmax_scaler': True,
            'window_normalization': 'none',
            'feature_range': [0, 1]
        }
    }

def print_config_summary(config: Dict[str, Any]) -> None:
    """
    Print a summary of the configuration.
    
    Args:
        config: Configuration dictionary
    """
    print("\nConfiguration Summary:")
    print("\nData Settings:")
    print(f"- Feature columns: {config['data'].get('feature_columns', 'all')}")
    print(f"- Label column: {config['data']['label_column']}")
    print(f"- Sample size: {config['data']['sample_generation']['num_samples']}")
    
    print("\nWindow Settings:")
    print(f"- Length: {config['window']['length']}")
    print(f"- Method: {config['window']['method']}")
    if config['window']['method'] == 'overlap':
        print(f"- Overlap: {config['window']['overlap']*100}%")
    else:
        print(f"- Step size: {config['window']['step_size']}")
    
    print("\nNormalization Settings:")
    print(f"- Use MinMaxScaler: {config['normalization']['use_minmax_scaler']}")
    print(f"- Window normalization: {config['normalization']['window_normalization']}")

if __name__ == "__main__":
    # Example usage
    config = load_config('config.yaml')
    if config is None:
        print("Using default configuration")
        config = get_default_config()
    
    print_config_summary(config)