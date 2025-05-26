"""
Configuration utilities for ML Pipeline.
"""

import os
import sys
import yaml
import json
import argparse
from typing import Dict, List, Tuple, Optional, Union, Any

def load_config(config_path: str) -> Dict[str, Any]:
    """
    Load configuration from file.
    
    Args:
        config_path: Path to configuration file (YAML or JSON)
    
    Returns:
        Configuration dictionary
    """
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Configuration file not found: {config_path}")
    
    if config_path.endswith('.yaml') or config_path.endswith('.yml'):
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
    elif config_path.endswith('.json'):
        with open(config_path, 'r') as f:
            config = json.load(f)
    else:
        raise ValueError(f"Unsupported configuration file format: {config_path}")
    
    return config

def get_default_config() -> Dict[str, Any]:
    """
    Get default configuration.
    
    Returns:
        Default configuration dictionary
    """
    # Get path to default configuration file
    default_config_path = os.path.join(os.path.dirname(__file__), 'config.yaml')
    
    # Load default configuration
    return load_config(default_config_path)

def save_config(config: Dict[str, Any], output_path: str) -> None:
    """
    Save configuration to file.
    
    Args:
        config: Configuration dictionary
        output_path: Path to save configuration file
    """
    # Create directory if it doesn't exist
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    if output_path.endswith('.yaml') or output_path.endswith('.yml'):
        with open(output_path, 'w') as f:
            yaml.dump(config, f, default_flow_style=False, sort_keys=False)
    elif output_path.endswith('.json'):
        with open(output_path, 'w') as f:
            json.dump(config, f, indent=2)
    else:
        raise ValueError(f"Unsupported configuration file format: {output_path}")

def merge_configs(base_config: Dict[str, Any], override_config: Dict[str, Any]) -> Dict[str, Any]:
    """
    Merge two configuration dictionaries.
    
    Args:
        base_config: Base configuration dictionary
        override_config: Override configuration dictionary
    
    Returns:
        Merged configuration dictionary
    """
    merged_config = base_config.copy()
    
    def _merge_dicts(base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
        for key, value in override.items():
            if key in base and isinstance(base[key], dict) and isinstance(value, dict):
                base[key] = _merge_dicts(base[key], value)
            else:
                base[key] = value
        return base
    
    return _merge_dicts(merged_config, override_config)

def validate_config(config: Dict[str, Any], schema: Dict[str, Any] = None) -> Tuple[bool, List[str]]:
    """
    Validate configuration against schema.
    
    Args:
        config: Configuration dictionary
        schema: Schema dictionary (optional)
    
    Returns:
        is_valid: Whether the configuration is valid
        errors: List of validation errors
    """
    if schema is None:
        # Use default schema
        schema_path = os.path.join(os.path.dirname(__file__), 'config_schema.yaml')
        if os.path.exists(schema_path):
            schema = load_config(schema_path)
        else:
            # No schema available, assume valid
            return True, []
    
    # Try to use jsonschema if available
    try:
        import jsonschema
        
        errors = []
        try:
            jsonschema.validate(instance=config, schema=schema)
            return True, []
        except jsonschema.exceptions.ValidationError as e:
            return False, [str(e)]
    except ImportError:
        # Fall back to basic validation
        return _basic_validate_config(config, schema)

def _basic_validate_config(config: Dict[str, Any], schema: Dict[str, Any]) -> Tuple[bool, List[str]]:
    """
    Basic configuration validation without jsonschema.
    
    Args:
        config: Configuration dictionary
        schema: Schema dictionary
    
    Returns:
        is_valid: Whether the configuration is valid
        errors: List of validation errors
    """
    errors = []
    
    # Check required properties
    if 'required' in schema:
        for required_prop in schema['required']:
            if required_prop not in config:
                errors.append(f"Missing required property: {required_prop}")
    
    # Check property types
    if 'properties' in schema:
        for prop_name, prop_schema in schema['properties'].items():
            if prop_name in config:
                prop_value = config[prop_name]
                
                # Check type
                if 'type' in prop_schema:
                    expected_type = prop_schema['type']
                    
                    if expected_type == 'object' and not isinstance(prop_value, dict):
                        errors.append(f"Property '{prop_name}' should be an object")
                    elif expected_type == 'array' and not isinstance(prop_value, list):
                        errors.append(f"Property '{prop_name}' should be an array")
                    elif expected_type == 'string' and not isinstance(prop_value, str):
                        errors.append(f"Property '{prop_name}' should be a string")
                    elif expected_type == 'number' and not isinstance(prop_value, (int, float)):
                        errors.append(f"Property '{prop_name}' should be a number")
                    elif expected_type == 'integer' and not isinstance(prop_value, int):
                        errors.append(f"Property '{prop_name}' should be an integer")
                    elif expected_type == 'boolean' and not isinstance(prop_value, bool):
                        errors.append(f"Property '{prop_name}' should be a boolean")
                
                # Check enum
                if 'enum' in prop_schema and prop_value not in prop_schema['enum']:
                    errors.append(f"Property '{prop_name}' should be one of: {prop_schema['enum']}")
                
                # Check minimum/maximum
                if isinstance(prop_value, (int, float)):
                    if 'minimum' in prop_schema and prop_value < prop_schema['minimum']:
                        errors.append(f"Property '{prop_name}' should be >= {prop_schema['minimum']}")
                    if 'maximum' in prop_schema and prop_value > prop_schema['maximum']:
                        errors.append(f"Property '{prop_name}' should be <= {prop_schema['maximum']}")
                
                # Check nested objects
                if 'properties' in prop_schema and isinstance(prop_value, dict):
                    nested_valid, nested_errors = _basic_validate_config(prop_value, prop_schema)
                    if not nested_valid:
                        errors.extend([f"{prop_name}.{e}" for e in nested_errors])
                
                # Check array items
                if 'items' in prop_schema and isinstance(prop_value, list):
                    for i, item in enumerate(prop_value):
                        if 'type' in prop_schema['items']:
                            expected_type = prop_schema['items']['type']
                            
                            if expected_type == 'object' and not isinstance(item, dict):
                                errors.append(f"Item {i} in '{prop_name}' should be an object")
                            elif expected_type == 'array' and not isinstance(item, list):
                                errors.append(f"Item {i} in '{prop_name}' should be an array")
                            elif expected_type == 'string' and not isinstance(item, str):
                                errors.append(f"Item {i} in '{prop_name}' should be a string")
                            elif expected_type == 'number' and not isinstance(item, (int, float)):
                                errors.append(f"Item {i} in '{prop_name}' should be a number")
                            elif expected_type == 'integer' and not isinstance(item, int):
                                errors.append(f"Item {i} in '{prop_name}' should be an integer")
                            elif expected_type == 'boolean' and not isinstance(item, bool):
                                errors.append(f"Item {i} in '{prop_name}' should be a boolean")
    
    return len(errors) == 0, errors

def print_config_summary(config: Dict[str, Any]) -> None:
    """
    Print a summary of the configuration.
    
    Args:
        config: Configuration dictionary
    """
    print("\nConfiguration Summary:")
    print("======================")
    
    # Data section
    if 'data' in config:
        print("\nData:")
        data_config = config['data']
        print(f"  Feature columns: {data_config.get('feature_columns', 'All')}")
        print(f"  Label column: {data_config.get('label_column', 'None')}")
        print(f"  NA strategy: {data_config.get('na_strategy', 'None')}")
    
    # Window section
    if 'window' in config:
        print("\nWindow:")
        window_config = config['window']
        print(f"  Length: {window_config.get('length', 'None')}")
        print(f"  Overlap: {window_config.get('overlap', 'None')}")
        print(f"  Method: {window_config.get('method', 'None')}")
    
    # Normalization section
    if 'normalization' in config:
        print("\nNormalization:")
        norm_config = config['normalization']
        print(f"  Method: {norm_config.get('method', 'None')}")
        print(f"  Window normalization: {norm_config.get('window_normalization', 'None')}")
    
    # Augmentation section
    if 'augmentation' in config:
        print("\nAugmentation:")
        aug_config = config['augmentation']
        print(f"  Enabled: {aug_config.get('enabled', False)}")
        if aug_config.get('enabled', False):
            methods = aug_config.get('methods', {})
            enabled_methods = [m for m, v in methods.items() if v.get('enabled', False)]
            print(f"  Methods: {', '.join(enabled_methods) if enabled_methods else 'None'}")
            print(f"  Ratio: {aug_config.get('augmentation_ratio', 'None')}")
    
    # Features section
    if 'features' in config:
        print("\nFeature Extraction:")
        feat_config = config['features']
        print(f"  Extract: {feat_config.get('extract', False)}")
        if feat_config.get('extract', False):
            print(f"  Types: {', '.join(feat_config.get('types', []))}")
    
    # Output section
    if 'output' in config:
        print("\nOutput:")
        out_config = config['output']
        print(f"  Format: {out_config.get('format', 'None')}")
        print(f"  Compress: {out_config.get('compress', False)}")
        
        split_config = out_config.get('split', {})
        if split_config.get('enabled', False):
            print(f"  Split: Train={split_config.get('train_ratio', 0.7)}, "
                  f"Val={split_config.get('val_ratio', 0.15)}, "
                  f"Test={split_config.get('test_ratio', 0.15)}")
    
    # Pipeline section
    if 'pipeline' in config:
        print("\nPipeline:")
        pipe_config = config['pipeline']
        print(f"  Jobs: {pipe_config.get('n_jobs', 'None')}")
        print(f"  Verbose: {pipe_config.get('verbose', 'None')}")
        print(f"  Batch size: {pipe_config.get('batch_size', 'None')}")
    
    print("\n")

def update_config_from_args(config: Dict[str, Any], args: argparse.Namespace) -> Dict[str, Any]:
    """
    Update configuration from command line arguments.
    
    Args:
        config: Configuration dictionary
        args: Command line arguments
    
    Returns:
        Updated configuration dictionary
    """
    updated_config = config.copy()
    
    # Convert args to dictionary
    args_dict = vars(args)
    
    # Update data section
    if 'data' in updated_config:
        data_config = updated_config['data']
        
        if 'feature_columns' in args_dict and args_dict['feature_columns'] is not None:
            data_config['feature_columns'] = args_dict['feature_columns']
        
        if 'label_column' in args_dict and args_dict['label_column'] is not None:
            data_config['label_column'] = args_dict['label_column']
        
        if 'delimiter' in args_dict and args_dict['delimiter'] is not None:
            data_config['delimiter'] = args_dict['delimiter']
        
        if 'na_strategy' in args_dict and args_dict['na_strategy'] is not None:
            data_config['na_strategy'] = args_dict['na_strategy']
    
    # Update window section
    if 'window' in updated_config:
        window_config = updated_config['window']
        
        if 'window_length' in args_dict and args_dict['window_length'] is not None:
            window_config['length'] = args_dict['window_length']
        
        if 'window_overlap' in args_dict and args_dict['window_overlap'] is not None:
            window_config['overlap'] = args_dict['window_overlap']
        
        if 'window_method' in args_dict and args_dict['window_method'] is not None:
            window_config['method'] = args_dict['window_method']
    
    # Update normalization section
    if 'normalization' in updated_config:
        norm_config = updated_config['normalization']
        
        if 'normalization_method' in args_dict and args_dict['normalization_method'] is not None:
            norm_config['method'] = args_dict['normalization_method']
        
        if 'window_normalization' in args_dict and args_dict['window_normalization'] is not None:
            norm_config['window_normalization'] = args_dict['window_normalization']
    
    # Update augmentation section
    if 'augmentation' in updated_config:
        aug_config = updated_config['augmentation']
        
        if 'augmentation_enabled' in args_dict:
            aug_config['enabled'] = args_dict['augmentation_enabled']
        
        if 'augmentation_ratio' in args_dict and args_dict['augmentation_ratio'] is not None:
            aug_config['augmentation_ratio'] = args_dict['augmentation_ratio']
    
    # Update output section
    if 'output' in updated_config:
        out_config = updated_config['output']
        
        if 'output_format' in args_dict and args_dict['output_format'] is not None:
            out_config['format'] = args_dict['output_format']
        
        if 'compress' in args_dict:
            out_config['compress'] = args_dict['compress']
        
        if 'split_enabled' in args_dict:
            if 'split' not in out_config:
                out_config['split'] = {}
            out_config['split']['enabled'] = args_dict['split_enabled']
        
        if 'train_ratio' in args_dict and args_dict['train_ratio'] is not None:
            if 'split' not in out_config:
                out_config['split'] = {}
            out_config['split']['train_ratio'] = args_dict['train_ratio']
        
        if 'val_ratio' in args_dict and args_dict['val_ratio'] is not None:
            if 'split' not in out_config:
                out_config['split'] = {}
            out_config['split']['val_ratio'] = args_dict['val_ratio']
        
        if 'test_ratio' in args_dict and args_dict['test_ratio'] is not None:
            if 'split' not in out_config:
                out_config['split'] = {}
            out_config['split']['test_ratio'] = args_dict['test_ratio']
    
    # Update pipeline section
    if 'pipeline' in updated_config:
        pipe_config = updated_config['pipeline']
        
        if 'n_jobs' in args_dict and args_dict['n_jobs'] is not None:
            pipe_config['n_jobs'] = args_dict['n_jobs']
        
        if 'verbose' in args_dict and args_dict['verbose'] is not None:
            pipe_config['verbose'] = args_dict['verbose']
        
        if 'batch_size' in args_dict and args_dict['batch_size'] is not None:
            pipe_config['batch_size'] = args_dict['batch_size']
    
    return updated_config

def add_config_arguments(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    """
    Add configuration arguments to argument parser.
    
    Args:
        parser: Argument parser
    
    Returns:
        Updated argument parser
    """
    # Configuration file
    parser.add_argument(
        '--config',
        type=str,
        help='Path to configuration file'
    )
    
    # Data arguments
    data_group = parser.add_argument_group('Data')
    data_group.add_argument(
        '--feature-columns',
        type=str,
        nargs='+',
        help='Feature columns to use'
    )
    data_group.add_argument(
        '--label-column',
        type=str,
        help='Label column to use'
    )
    data_group.add_argument(
        '--delimiter',
        type=str,
        help='Delimiter for CSV files'
    )
    data_group.add_argument(
        '--na-strategy',
        type=str,
        choices=['interpolate', 'drop', 'fill'],
        help='Strategy for handling missing values'
    )
    
    # Window arguments
    window_group = parser.add_argument_group('Window')
    window_group.add_argument(
        '--window-length',
        type=int,
        help='Window length'
    )
    window_group.add_argument(
        '--window-overlap',
        type=float,
        help='Window overlap'
    )
    window_group.add_argument(
        '--window-method',
        type=str,
        choices=['overlap', 'step'],
        help='Window creation method'
    )
    
    # Normalization arguments
    norm_group = parser.add_argument_group('Normalization')
    norm_group.add_argument(
        '--normalization-method',
        type=str,
        choices=['standard', 'minmax', 'robust', 'none'],
        help='Normalization method'
    )
    norm_group.add_argument(
        '--window-normalization',
        type=str,
        choices=['zscore', 'minmax', 'robust', 'none'],
        help='Window normalization method'
    )
    
    # Augmentation arguments
    aug_group = parser.add_argument_group('Augmentation')
    aug_group.add_argument(
        '--augmentation-enabled',
        action='store_true',
        help='Enable data augmentation'
    )
    aug_group.add_argument(
        '--augmentation-ratio',
        type=float,
        help='Augmentation ratio'
    )
    
    # Output arguments
    out_group = parser.add_argument_group('Output')
    out_group.add_argument(
        '--output-format',
        type=str,
        choices=['npz', 'csv'],
        help='Output format'
    )
    out_group.add_argument(
        '--compress',
        action='store_true',
        help='Compress output'
    )
    out_group.add_argument(
        '--split-enabled',
        action='store_true',
        help='Enable data splitting'
    )
    out_group.add_argument(
        '--train-ratio',
        type=float,
        help='Training data ratio'
    )
    out_group.add_argument(
        '--val-ratio',
        type=float,
        help='Validation data ratio'
    )
    out_group.add_argument(
        '--test-ratio',
        type=float,
        help='Test data ratio'
    )
    
    # Pipeline arguments
    pipe_group = parser.add_argument_group('Pipeline')
    pipe_group.add_argument(
        '--n-jobs',
        type=int,
        help='Number of parallel jobs'
    )
    pipe_group.add_argument(
        '--verbose',
        type=int,
        choices=[0, 1, 2],
        help='Verbosity level'
    )
    pipe_group.add_argument(
        '--batch-size',
        type=int,
        help='Batch size for processing'
    )
    
    return parser

if __name__ == "__main__":
    # Example usage
    print("Configuration utilities loaded.")
    print("Use the provided functions to load and manage configurations.")