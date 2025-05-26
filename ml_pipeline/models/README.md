# Time Series Classification Models

This module provides tools for training, evaluating, and deploying time series classification models for embedded systems. It is specifically designed to work with STM32 microcontrollers and includes tools for model optimization and conversion.

## Table of Contents

- [Installation](#installation)
- [Dependencies](#dependencies)
- [Usage](#usage)
- [Workflow](#workflow)
- [Model Architectures](#model-architectures)
- [Configuration](#configuration)
- [Deployment](#deployment)
- [FAQ](#faq)

## Installation

```bash
# From the project root directory
pip install -e ml_pipeline/
```

## Dependencies

- TensorFlow 2.x
- NumPy
- Pandas
- Scikit-learn
- Matplotlib
- Seaborn
- PyYAML

## Usage

### 1. Training a Model

```bash
# Basic training with default configuration
python train.py --data path/to/data.csv

# Training with custom configuration
python train.py --data path/to/data.csv \
                --config model_config.yaml \
                --data-config ../data_processing/config.yaml \
                --output-dir saved_models \
                --name my_model \
                --gpu 0
```

### 2. Evaluating a Model

```bash
# Basic evaluation
python evaluate.py --model saved_models/my_model.h5 \
                  --data path/to/test_data.csv

# Detailed evaluation with custom configuration
python evaluate.py --model saved_models/my_model.h5 \
                  --data path/to/test_data.csv \
                  --config ../data_processing/config.yaml \
                  --scaler saved_models/scaler.joblib \
                  --output-dir evaluation_results
```

### 3. Converting for Embedded Deployment

```bash
# Convert to TFLite with float16 quantization
python convert_model.py --model saved_models/my_model.h5 \
                       --quantize float16 \
                       --format both \
                       --optimize \
                       --analyze

# Convert with int8 quantization (requires calibration data)
python convert_model.py --model saved_models/my_model.h5 \
                       --quantize int8 \
                       --calibration-data path/to/calibration_data.csv \
                       --format both \
                       --optimize \
                       --analyze
```

## Workflow

1. **Data Preparation**
   - Ensure your data is properly formatted (CSV)
   - Configure data processing parameters in `config.yaml`

2. **Model Training**
   - Choose model architecture (LSTM or CNN)
   - Configure model parameters in `model_config.yaml`
   - Train the model using `train.py`
   - Monitor training progress and metrics

3. **Model Evaluation**
   - Evaluate model performance on test data
   - Analyze metrics and visualizations
   - Iterate on model architecture if needed

4. **Model Deployment**
   - Convert model to TFLite format
   - Apply quantization for size reduction
   - Generate C array for embedded implementation
   - Verify model size and performance

## Model Architectures

### LSTM Model
- Suitable for sequential patterns
- Better for long-term dependencies
- Configuration options:
  ```yaml
  model:
    architecture: "lstm"
    layer_sizes: [64, 32]
    dropout_rate: 0.3
    activation: "tanh"
    use_batch_norm: true
  ```

### CNN Model
- Suitable for local patterns
- More efficient computation
- Configuration options:
  ```yaml
  model:
    architecture: "cnn"
    filters: [64, 128]
    kernel_size: 3
    pool_size: 2
    dropout_rate: 0.25
  ```

## Configuration

### Model Configuration (`model_config.yaml`)
```yaml
model:
  architecture: "lstm"
  layer_sizes: [64, 32]
  dropout_rate: 0.3
  activation: "tanh"
  use_batch_norm: true

training:
  batch_size: 32
  epochs: 100
  learning_rate: 0.001
  validation_split: 0.2
  early_stopping_patience: 10
  use_class_weights: true

augmentation:
  enabled: false
  jitter_sigma: 0.03
  scaling_sigma: 0.1
```

## Deployment

### Model Optimization
1. **Quantization**
   - Float16: 2x size reduction, minimal accuracy impact
   - Int8: 4x size reduction, may impact accuracy
   - Full Int8: Maximum size reduction, requires calibration

2. **Memory Requirements**
   - Check model analysis report for detailed memory usage
   - Ensure compatibility with target device

3. **STM32 Integration**
   - Use generated C array in your STM32 project
   - Follow memory alignment requirements
   - Consider using CMSIS-NN for optimized inference

## FAQ

### Q: How do I choose between LSTM and CNN?
A: LSTM is better for capturing long-term patterns but requires more computation. CNN is more efficient and works well for local patterns. Test both on your data to determine the best fit.

### Q: What quantization should I use?
A: Start with float16 quantization. If you need further size reduction and have calibration data available, try int8 quantization. Always verify accuracy after quantization.

### Q: How can I reduce model size?
1. Reduce the number of layers/units
2. Apply quantization
3. Use the `--optimize` flag during conversion
4. Consider using a simpler architecture

### Q: How do I handle model accuracy degradation after quantization?
1. Use representative calibration data
2. Try different quantization schemes
3. Retrain the model with quantization awareness
4. Adjust model architecture to be more robust

### Q: What are the STM32 memory constraints?
- Flash: Typically 128KB-512KB
- RAM: Usually 32KB-128KB
- Consider these limits when designing your model

## Contributing

Feel free to submit issues and enhancement requests!

## License

This project is licensed under the MIT License - see the LICENSE file for details.