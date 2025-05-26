"""
Model conversion utilities for ML Pipeline.
"""

import os
import numpy as np
import tensorflow as tf
from typing import Dict, List, Tuple, Optional, Union, Any

def convert_to_tflite(
    model: Union[tf.keras.Model, str],
    quantize: str = 'none',
    optimize: bool = True,
    representative_dataset: Optional[Any] = None,
    output_dir: str = 'converted_models',
    model_name: str = 'model'
) -> Tuple[bytes, str]:
    """
    Convert a Keras model to TensorFlow Lite format.
    
    Args:
        model: Keras model or path to saved model
        quantize: Quantization method ('none', 'float16', 'int8', 'full_int8')
        optimize: Whether to apply optimizations
        representative_dataset: Representative dataset for quantization
        output_dir: Directory to save converted model
        model_name: Name for the converted model
    
    Returns:
        tflite_model: Converted model as bytes
        tflite_path: Path to saved model
    """
    # Load model if path is provided
    if isinstance(model, str):
        model = tf.keras.models.load_model(model)
    
    # Create converter
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    
    # Apply optimizations
    if optimize:
        converter.optimizations = [tf.lite.Optimize.DEFAULT]
    
    # Apply quantization
    quantize = quantize.lower()
    if quantize == 'float16':
        converter.target_spec.supported_types = [tf.float16]
    elif quantize in ['int8', 'full_int8']:
        converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
        converter.inference_input_type = tf.int8 if quantize == 'full_int8' else tf.float32
        converter.inference_output_type = tf.int8 if quantize == 'full_int8' else tf.float32
        
        if representative_dataset is not None:
            converter.representative_dataset = representative_dataset
        else:
            raise ValueError("Representative dataset is required for int8 quantization")
    
    # Convert model
    tflite_model = converter.convert()
    
    # Save model
    os.makedirs(output_dir, exist_ok=True)
    tflite_path = os.path.join(output_dir, f"{model_name}_{quantize}.tflite")
    with open(tflite_path, 'wb') as f:
        f.write(tflite_model)
    
    return tflite_model, tflite_path

def generate_c_array(
    tflite_model: bytes,
    variable_name: str = 'g_model',
    output_dir: str = 'converted_models',
    model_name: str = 'model'
) -> Tuple[str, str]:
    """
    Generate C array from TensorFlow Lite model.
    
    Args:
        tflite_model: TensorFlow Lite model as bytes
        variable_name: Name for the C array variable
        output_dir: Directory to save generated files
        model_name: Name for the generated files
    
    Returns:
        c_code: Generated C code
        c_path: Path to saved C file
    """
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Generate C code
    c_code = f"// Auto-generated from TensorFlow Lite model\n\n"
    c_code += f"#include <stdint.h>\n\n"
    c_code += f"const unsigned char {variable_name}[] = {{\n"
    
    # Add model bytes
    for i, byte in enumerate(tflite_model):
        if i % 12 == 0:
            c_code += "  "
        c_code += f"0x{byte:02x},"
        if i % 12 == 11 or i == len(tflite_model) - 1:
            c_code += "\n"
        else:
            c_code += " "
    
    c_code += "};\n\n"
    c_code += f"const unsigned int {variable_name}_len = {len(tflite_model)};\n"
    
    # Save C file
    c_path = os.path.join(output_dir, f"{model_name}.c")
    with open(c_path, 'w') as f:
        f.write(c_code)
    
    # Generate header file
    h_code = f"// Auto-generated from TensorFlow Lite model\n\n"
    h_code += f"#ifndef {variable_name.upper()}_H\n"
    h_code += f"#define {variable_name.upper()}_H\n\n"
    h_code += f"#include <stdint.h>\n\n"
    h_code += f"extern const unsigned char {variable_name}[];\n"
    h_code += f"extern const unsigned int {variable_name}_len;\n\n"
    h_code += f"#endif // {variable_name.upper()}_H\n"
    
    h_path = os.path.join(output_dir, f"{model_name}.h")
    with open(h_path, 'w') as f:
        f.write(h_code)
    
    return c_code, c_path

def analyze_model(
    model_path: str,
    tflite_path: str,
    output_dir: str = 'analysis',
    model_name: str = 'model'
) -> Dict[str, Any]:
    """
    Analyze model before and after conversion.
    
    Args:
        model_path: Path to original model
        tflite_path: Path to converted TFLite model
        output_dir: Directory to save analysis results
        model_name: Name for the analysis files
    
    Returns:
        Dictionary containing analysis results
    """
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Load original model
    model = tf.keras.models.load_model(model_path)
    
    # Get model size information
    model_size = {
        'original_size': os.path.getsize(model_path),
        'tflite_size': os.path.getsize(tflite_path),
        'size_reduction': 1 - (os.path.getsize(tflite_path) / os.path.getsize(model_path))
    }
    
    # Get model summary
    import io
    buffer = io.StringIO()
    model.summary(print_fn=lambda x: buffer.write(x + '\n'))
    model_summary = buffer.getvalue()
    buffer.close()
    
    # Save model summary
    summary_path = os.path.join(output_dir, f"{model_name}_summary.txt")
    with open(summary_path, 'w') as f:
        f.write(model_summary)
    
    # Benchmark TFLite model
    interpreter = tf.lite.Interpreter(model_path=tflite_path)
    interpreter.allocate_tensors()
    
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    
    # Create random input
    input_shape = input_details[0]['shape']
    input_data = np.array(np.random.random_sample(input_shape), dtype=np.float32)
    
    # Benchmark inference
    import time
    times = []
    for _ in range(100):
        start_time = time.time()
        interpreter.set_tensor(input_details[0]['index'], input_data)
        interpreter.invoke()
        output_data = interpreter.get_tensor(output_details[0]['index'])
        times.append(time.time() - start_time)
    
    benchmark = {
        'average_time_ms': np.mean(times) * 1000,
        'min_time_ms': np.min(times) * 1000,
        'max_time_ms': np.max(times) * 1000,
        'inference_per_second': 1 / np.mean(times),
        'input_details': input_details,
        'output_details': output_details
    }
    
    # Save benchmark results
    benchmark_path = os.path.join(output_dir, f"{model_name}_benchmark.json")
    with open(benchmark_path, 'w') as f:
        import json
        json.dump(benchmark, f, indent=2)
    
    # Prepare results
    results = {
        'model_size': model_size,
        'benchmark': benchmark,
        'summary_path': summary_path,
        'benchmark_path': benchmark_path
    }
    
    return results

def convert_model(
    model_path: str,
    config: Dict[str, Any],
    representative_dataset: Optional[Any] = None,
    output_dir: str = 'converted_models',
    model_name: str = 'model'
) -> Dict[str, Any]:
    """
    Convert model based on configuration.
    
    Args:
        model_path: Path to original model
        config: Conversion configuration dictionary
        representative_dataset: Representative dataset for quantization
        output_dir: Directory to save converted models
        model_name: Name for the converted models
    
    Returns:
        Dictionary containing conversion results
    """
    # Get conversion parameters
    formats = config.get('formats', ['tflite'])
    quantize = config.get('quantization', 'none')
    optimize = config.get('optimization', True)
    variable_name = config.get('variable_name', 'g_model')
    analyze = config.get('analyze', True)
    
    results = {}
    
    # Convert to TFLite
    if 'tflite' in formats:
        tflite_model, tflite_path = convert_to_tflite(
            model_path,
            quantize=quantize,
            optimize=optimize,
            representative_dataset=representative_dataset,
            output_dir=output_dir,
            model_name=model_name
        )
        results['tflite'] = {
            'path': tflite_path,
            'size': os.path.getsize(tflite_path)
        }
    
    # Generate C array
    if 'c_array' in formats and 'tflite' in formats:
        c_code, c_path = generate_c_array(
            tflite_model,
            variable_name=variable_name,
            output_dir=output_dir,
            model_name=model_name
        )
        results['c_array'] = {
            'path': c_path,
            'header_path': os.path.join(output_dir, f"{model_name}.h"),
            'size': os.path.getsize(c_path)
        }
    
    # Analyze model
    if analyze and 'tflite' in formats:
        analysis = analyze_model(
            model_path,
            tflite_path,
            output_dir=os.path.join(output_dir, 'analysis'),
            model_name=model_name
        )
        results['analysis'] = analysis
    
    return results

def get_representative_dataset(
    data: np.ndarray,
    num_samples: int = 100,
    input_type: type = np.float32
) -> Any:
    """
    Create representative dataset for quantization.
    
    Args:
        data: Input data (n_samples, window_length, n_features)
        num_samples: Number of samples to use
        input_type: Input data type
    
    Returns:
        Representative dataset generator
    """
    # Select random samples
    indices = np.random.choice(len(data), min(num_samples, len(data)), replace=False)
    samples = data[indices].astype(input_type)
    
    def representative_dataset():
        for sample in samples:
            yield [sample]
    
    return representative_dataset

if __name__ == "__main__":
    # Example usage
    print("Model conversion utilities loaded.")
    print("Use the provided functions to convert models to TensorFlow Lite format.")