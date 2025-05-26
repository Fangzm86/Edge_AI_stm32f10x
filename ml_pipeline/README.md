# ML Pipeline for Embedded Systems

A comprehensive machine learning pipeline for time series classification on embedded systems, specifically designed for STM32 microcontrollers. This pipeline provides tools for data processing, model training, evaluation, and deployment to resource-constrained devices.

## Features

- **Data Processing**: Window creation, normalization, and augmentation for time series data
- **Model Training**: LSTM and CNN architectures optimized for time series classification
- **Model Evaluation**: Comprehensive metrics and visualizations
- **Model Conversion**: TensorFlow Lite conversion with quantization options
- **Embedded Deployment**: C code generation for STM32 microcontrollers

## Installation

### Prerequisites

- Python 3.7 or higher
- pip package manager

### Install from Source

```bash
# Clone the repository
git clone https://github.com/yourusername/ml_pipeline.git
cd ml_pipeline

# Install the package
pip install -e .

# Install development dependencies (optional)
pip install -e ".[dev]"
```

## Project Structure

```
ml_pipeline/
├── data_processing/       # Data processing tools
│   ├── __init__.py
│   ├── data_processing.py # Core data processing functions
│   ├── config_utils.py    # Configuration utilities
│   ├── process_data.py    # Command-line tool for data processing
│   └── README.md          # Data processing documentation
│
├── models/                # Model training and deployment
│   ├── __init__.py
│   ├── model.py           # Model architecture definitions
│   ├── train.py           # Model training script
│   ├── evaluate.py        # Model evaluation script
│   ├── convert_model.py   # Model conversion for embedded deployment
│   └── README.md          # Model training documentation
│
├── examples/              # Example notebooks and scripts
│   ├── data_processing_example.ipynb
│   ├── model_training_example.ipynb
│   └── deployment_example.ipynb
│
├── setup.py               # Package installation script
└── README.md              # Main documentation
```

## Quick Start

### 1. Process Data

```bash
# Process CSV data with default settings
process-data --input your_data.csv --output processed_data.npz

# Process with custom configuration
process-data --input your_data.csv --output processed_data.npz --config your_config.yaml --visualize
```

### 2. Train Model

```bash
# Train a model with default settings
train-model --data processed_data.npz --output-dir models/

# Train with custom configuration
train-model --data processed_data.npz --config model_config.yaml --output-dir models/ --name my_model
```

### 3. Evaluate Model

```bash
# Evaluate model on test data
evaluate-model --model models/my_model.h5 --data test_data.csv --output-dir evaluation/
```

### 4. Convert for Embedded Deployment

```bash
# Convert model to TFLite with quantization
convert-model --model models/my_model.h5 --quantize float16 --format both --output-dir embedded/
```

## Data Processing

The data processing module provides tools for:

- Creating fixed-length windows from time series data
- Normalizing data at feature and window levels
- Augmenting data with jittering, scaling, and time warping
- Visualizing processed data

For detailed documentation, see [Data Processing README](data_processing/README.md).

## Model Training

The model training module provides:

- LSTM and CNN architectures optimized for time series classification
- Training with early stopping and learning rate scheduling
- Class weighting for imbalanced datasets
- Comprehensive evaluation metrics and visualizations

For detailed documentation, see [Models README](models/README.md).

## Embedded Deployment

The pipeline supports deployment to STM32 microcontrollers through:

- TensorFlow Lite conversion with various quantization options
- C array generation for direct inclusion in embedded projects
- Memory usage analysis and optimization

## Configuration

Both data processing and model training use YAML configuration files:

### Data Processing Configuration

```yaml
data:
  feature_columns: ['accel_x', 'accel_y', 'accel_z']
  label_column: 'activity'

window:
  length: 128
  overlap: 0.5

normalization:
  method: 'standard'
  window_normalization: 'zscore'
```

### Model Configuration

```yaml
model:
  architecture: 'lstm'
  layer_sizes: [64, 32]
  dropout_rate: 0.3

training:
  batch_size: 32
  epochs: 100
  learning_rate: 0.001
```

## Examples

The `examples/` directory contains Jupyter notebooks demonstrating:

1. Data processing workflow
2. Model training and evaluation
3. Model conversion and deployment

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add some amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- TensorFlow team for TensorFlow Lite
- STMicroelectronics for STM32 microcontrollers
- The open-source community for various tools and libraries