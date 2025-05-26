"""
Model architecture definitions for ML Pipeline.
"""

import os
import json
import numpy as np
import tensorflow as tf
from typing import Dict, List, Tuple, Optional, Union, Any, Callable

def create_model(
    config: Dict[str, Any],
    input_shape: Tuple[int, int],
    num_classes: int
) -> tf.keras.Model:
    """
    Create a model based on the configuration.
    
    Args:
        config: Model configuration dictionary
        input_shape: Shape of input data (window_length, n_features)
        num_classes: Number of output classes
    
    Returns:
        Compiled Keras model
    """
    # Get architecture type
    architecture = config.get('architecture', 'lstm').lower()
    
    # Create model based on architecture
    if architecture == 'lstm':
        model = create_lstm_model(config, input_shape, num_classes)
    elif architecture == 'cnn':
        model = create_cnn_model(config, input_shape, num_classes)
    elif architecture == 'hybrid':
        model = create_hybrid_model(config, input_shape, num_classes)
    elif architecture == 'tcn':
        model = create_tcn_model(config, input_shape, num_classes)
    elif architecture == 'transformer':
        model = create_transformer_model(config, input_shape, num_classes)
    elif architecture == 'resnet':
        model = create_resnet_model(config, input_shape, num_classes)
    else:
        raise ValueError(f"Unknown architecture: {architecture}")
    
    return model

def create_lstm_model(
    config: Dict[str, Any],
    input_shape: Tuple[int, int],
    num_classes: int
) -> tf.keras.Model:
    """
    Create an LSTM model.
    
    Args:
        config: Model configuration dictionary
        input_shape: Shape of input data (window_length, n_features)
        num_classes: Number of output classes
    
    Returns:
        Compiled Keras model
    """
    # Get parameters
    layer_sizes = config.get('layer_sizes', [64, 32])
    dropout_rate = config.get('dropout_rate', 0.3)
    recurrent_dropout_rate = config.get('recurrent_dropout_rate', 0.0)
    activation = config.get('activation', 'tanh')
    use_batch_norm = config.get('use_batch_norm', True)
    use_layer_norm = config.get('use_layer_norm', False)
    bidirectional = config.get('bidirectional', False)
    return_sequences = config.get('return_sequences', False)
    attention = config.get('attention', False)
    
    # Create regularizers
    kernel_regularizer = None
    if config.get('kernel_regularizer', {}).get('type'):
        reg_type = config['kernel_regularizer']['type']
        factor = config['kernel_regularizer'].get('factor', 0.001)
        
        if reg_type == 'l1':
            kernel_regularizer = tf.keras.regularizers.l1(factor)
        elif reg_type == 'l2':
            kernel_regularizer = tf.keras.regularizers.l2(factor)
        elif reg_type == 'l1_l2':
            kernel_regularizer = tf.keras.regularizers.l1_l2(l1=factor, l2=factor)
    
    # Create model
    inputs = tf.keras.layers.Input(shape=input_shape)
    x = inputs
    
    # Add LSTM layers
    for i, units in enumerate(layer_sizes):
        return_seq = return_sequences if i == len(layer_sizes) - 1 else True
        
        if bidirectional:
            lstm_layer = tf.keras.layers.Bidirectional(
                tf.keras.layers.LSTM(
                    units,
                    activation=activation,
                    return_sequences=return_seq,
                    dropout=dropout_rate,
                    recurrent_dropout=recurrent_dropout_rate,
                    kernel_regularizer=kernel_regularizer
                )
            )
        else:
            lstm_layer = tf.keras.layers.LSTM(
                units,
                activation=activation,
                return_sequences=return_seq,
                dropout=dropout_rate,
                recurrent_dropout=recurrent_dropout_rate,
                kernel_regularizer=kernel_regularizer
            )
        
        x = lstm_layer(x)
        
        if use_batch_norm:
            x = tf.keras.layers.BatchNormalization()(x)
        
        if use_layer_norm:
            x = tf.keras.layers.LayerNormalization()(x)
    
    # Add attention if requested
    if attention and return_sequences:
        x = tf.keras.layers.Attention()([x, x])
        x = tf.keras.layers.GlobalAveragePooling1D()(x)
    elif return_sequences:
        x = tf.keras.layers.GlobalAveragePooling1D()(x)
    
    # Add dropout
    if dropout_rate > 0:
        x = tf.keras.layers.Dropout(dropout_rate)(x)
    
    # Add output layer
    if num_classes > 2:
        outputs = tf.keras.layers.Dense(num_classes, activation='softmax')(x)
    else:
        outputs = tf.keras.layers.Dense(1, activation='sigmoid')(x)
    
    # Create model
    model = tf.keras.Model(inputs=inputs, outputs=outputs)
    
    # Compile model
    if num_classes > 2:
        model.compile(
            optimizer='adam',
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy']
        )
    else:
        model.compile(
            optimizer='adam',
            loss='binary_crossentropy',
            metrics=['accuracy']
        )
    
    return model

def create_cnn_model(
    config: Dict[str, Any],
    input_shape: Tuple[int, int],
    num_classes: int
) -> tf.keras.Model:
    """
    Create a CNN model.
    
    Args:
        config: Model configuration dictionary
        input_shape: Shape of input data (window_length, n_features)
        num_classes: Number of output classes
    
    Returns:
        Compiled Keras model
    """
    # Get parameters
    cnn_config = config.get('cnn', {})
    filters = cnn_config.get('filters', [64, 128, 256])
    kernel_size = cnn_config.get('kernel_size', 3)
    pool_size = cnn_config.get('pool_size', 2)
    pool_type = cnn_config.get('pool_type', 'max')
    use_separable_conv = cnn_config.get('use_separable_conv', False)
    dilation_rate = cnn_config.get('dilation_rate', 1)
    
    dropout_rate = config.get('dropout_rate', 0.3)
    activation = config.get('activation', 'relu')
    use_batch_norm = config.get('use_batch_norm', True)
    layer_sizes = config.get('layer_sizes', [64, 32])
    
    # Create regularizers
    kernel_regularizer = None
    if config.get('kernel_regularizer', {}).get('type'):
        reg_type = config['kernel_regularizer']['type']
        factor = config['kernel_regularizer'].get('factor', 0.001)
        
        if reg_type == 'l1':
            kernel_regularizer = tf.keras.regularizers.l1(factor)
        elif reg_type == 'l2':
            kernel_regularizer = tf.keras.regularizers.l2(factor)
        elif reg_type == 'l1_l2':
            kernel_regularizer = tf.keras.regularizers.l1_l2(l1=factor, l2=factor)
    
    # Create model
    inputs = tf.keras.layers.Input(shape=input_shape)
    
    # Reshape input for 1D convolution if needed
    if len(input_shape) == 2:
        x = tf.keras.layers.Reshape((input_shape[0], input_shape[1], 1))(inputs)
    else:
        x = inputs
    
    # Add CNN layers
    for i, filter_size in enumerate(filters):
        # Add convolutional layer
        if use_separable_conv:
            x = tf.keras.layers.SeparableConv1D(
                filters=filter_size,
                kernel_size=kernel_size,
                activation=activation,
                padding='same',
                dilation_rate=dilation_rate,
                kernel_regularizer=kernel_regularizer
            )(x)
        else:
            x = tf.keras.layers.Conv1D(
                filters=filter_size,
                kernel_size=kernel_size,
                activation=activation,
                padding='same',
                dilation_rate=dilation_rate,
                kernel_regularizer=kernel_regularizer
            )(x)
        
        # Add batch normalization
        if use_batch_norm:
            x = tf.keras.layers.BatchNormalization()(x)
        
        # Add pooling layer
        if pool_type == 'max':
            x = tf.keras.layers.MaxPooling1D(pool_size=pool_size)(x)
        else:
            x = tf.keras.layers.AveragePooling1D(pool_size=pool_size)(x)
        
        # Add dropout
        if dropout_rate > 0:
            x = tf.keras.layers.Dropout(dropout_rate)(x)
    
    # Flatten
    x = tf.keras.layers.Flatten()(x)
    
    # Add dense layers
    for units in layer_sizes:
        x = tf.keras.layers.Dense(units, activation=activation)(x)
        
        if use_batch_norm:
            x = tf.keras.layers.BatchNormalization()(x)
        
        if dropout_rate > 0:
            x = tf.keras.layers.Dropout(dropout_rate)(x)
    
    # Add output layer
    if num_classes > 2:
        outputs = tf.keras.layers.Dense(num_classes, activation='softmax')(x)
    else:
        outputs = tf.keras.layers.Dense(1, activation='sigmoid')(x)
    
    # Create model
    model = tf.keras.Model(inputs=inputs, outputs=outputs)
    
    # Compile model
    if num_classes > 2:
        model.compile(
            optimizer='adam',
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy']
        )
    else:
        model.compile(
            optimizer='adam',
            loss='binary_crossentropy',
            metrics=['accuracy']
        )
    
    return model

def create_hybrid_model(
    config: Dict[str, Any],
    input_shape: Tuple[int, int],
    num_classes: int
) -> tf.keras.Model:
    """
    Create a hybrid CNN-LSTM model.
    
    Args:
        config: Model configuration dictionary
        input_shape: Shape of input data (window_length, n_features)
        num_classes: Number of output classes
    
    Returns:
        Compiled Keras model
    """
    # Get parameters
    cnn_config = config.get('cnn', {})
    filters = cnn_config.get('filters', [64, 128])
    kernel_size = cnn_config.get('kernel_size', 3)
    pool_size = cnn_config.get('pool_size', 2)
    
    layer_sizes = config.get('layer_sizes', [64, 32])
    dropout_rate = config.get('dropout_rate', 0.3)
    recurrent_dropout_rate = config.get('recurrent_dropout_rate', 0.0)
    activation = config.get('activation', 'relu')
    use_batch_norm = config.get('use_batch_norm', True)
    bidirectional = config.get('bidirectional', False)
    
    # Create regularizers
    kernel_regularizer = None
    if config.get('kernel_regularizer', {}).get('type'):
        reg_type = config['kernel_regularizer']['type']
        factor = config['kernel_regularizer'].get('factor', 0.001)
        
        if reg_type == 'l1':
            kernel_regularizer = tf.keras.regularizers.l1(factor)
        elif reg_type == 'l2':
            kernel_regularizer = tf.keras.regularizers.l2(factor)
        elif reg_type == 'l1_l2':
            kernel_regularizer = tf.keras.regularizers.l1_l2(l1=factor, l2=factor)
    
    # Create model
    inputs = tf.keras.layers.Input(shape=input_shape)
    x = inputs
    
    # Add CNN layers
    for i, filter_size in enumerate(filters):
        x = tf.keras.layers.Conv1D(
            filters=filter_size,
            kernel_size=kernel_size,
            activation=activation,
            padding='same',
            kernel_regularizer=kernel_regularizer
        )(x)
        
        if use_batch_norm:
            x = tf.keras.layers.BatchNormalization()(x)
        
        x = tf.keras.layers.MaxPooling1D(pool_size=pool_size)(x)
        
        if dropout_rate > 0:
            x = tf.keras.layers.Dropout(dropout_rate)(x)
    
    # Add LSTM layers
    for i, units in enumerate(layer_sizes):
        return_seq = i < len(layer_sizes) - 1
        
        if bidirectional:
            lstm_layer = tf.keras.layers.Bidirectional(
                tf.keras.layers.LSTM(
                    units,
                    activation='tanh',
                    return_sequences=return_seq,
                    dropout=dropout_rate,
                    recurrent_dropout=recurrent_dropout_rate,
                    kernel_regularizer=kernel_regularizer
                )
            )
        else:
            lstm_layer = tf.keras.layers.LSTM(
                units,
                activation='tanh',
                return_sequences=return_seq,
                dropout=dropout_rate,
                recurrent_dropout=recurrent_dropout_rate,
                kernel_regularizer=kernel_regularizer
            )
        
        x = lstm_layer(x)
        
        if use_batch_norm:
            x = tf.keras.layers.BatchNormalization()(x)
    
    # Add dropout
    if dropout_rate > 0:
        x = tf.keras.layers.Dropout(dropout_rate)(x)
    
    # Add output layer
    if num_classes > 2:
        outputs = tf.keras.layers.Dense(num_classes, activation='softmax')(x)
    else:
        outputs = tf.keras.layers.Dense(1, activation='sigmoid')(x)
    
    # Create model
    model = tf.keras.Model(inputs=inputs, outputs=outputs)
    
    # Compile model
    if num_classes > 2:
        model.compile(
            optimizer='adam',
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy']
        )
    else:
        model.compile(
            optimizer='adam',
            loss='binary_crossentropy',
            metrics=['accuracy']
        )
    
    return model

def create_tcn_model(
    config: Dict[str, Any],
    input_shape: Tuple[int, int],
    num_classes: int
) -> tf.keras.Model:
    """
    Create a Temporal Convolutional Network (TCN) model.
    
    Args:
        config: Model configuration dictionary
        input_shape: Shape of input data (window_length, n_features)
        num_classes: Number of output classes
    
    Returns:
        Compiled Keras model
    """
    # Import TCN layer
    try:
        from tcn import TCN
    except ImportError:
        raise ImportError("TCN package not found. Install it with: pip install keras-tcn")
    
    # Get parameters
    tcn_config = config.get('tcn', {})
    nb_filters = tcn_config.get('nb_filters', 64)
    kernel_size = tcn_config.get('kernel_size', 3)
    nb_stacks = tcn_config.get('nb_stacks', 1)
    dilations = tcn_config.get('dilations', [1, 2, 4, 8, 16])
    padding = tcn_config.get('padding', 'causal')
    use_skip_connections = tcn_config.get('use_skip_connections', True)
    dropout_rate = tcn_config.get('dropout_rate', 0.3)
    
    layer_sizes = config.get('layer_sizes', [64, 32])
    use_batch_norm = config.get('use_batch_norm', True)
    activation = config.get('activation', 'relu')
    
    # Create model
    inputs = tf.keras.layers.Input(shape=input_shape)
    
    # Add TCN layer
    x = TCN(
        nb_filters=nb_filters,
        kernel_size=kernel_size,
        nb_stacks=nb_stacks,
        dilations=dilations,
        padding=padding,
        use_skip_connections=use_skip_connections,
        dropout_rate=dropout_rate,
        return_sequences=False
    )(inputs)
    
    # Add dense layers
    for units in layer_sizes:
        x = tf.keras.layers.Dense(units, activation=activation)(x)
        
        if use_batch_norm:
            x = tf.keras.layers.BatchNormalization()(x)
        
        if dropout_rate > 0:
            x = tf.keras.layers.Dropout(dropout_rate)(x)
    
    # Add output layer
    if num_classes > 2:
        outputs = tf.keras.layers.Dense(num_classes, activation='softmax')(x)
    else:
        outputs = tf.keras.layers.Dense(1, activation='sigmoid')(x)
    
    # Create model
    model = tf.keras.Model(inputs=inputs, outputs=outputs)
    
    # Compile model
    if num_classes > 2:
        model.compile(
            optimizer='adam',
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy']
        )
    else:
        model.compile(
            optimizer='adam',
            loss='binary_crossentropy',
            metrics=['accuracy']
        )
    
    return model

def create_transformer_model(
    config: Dict[str, Any],
    input_shape: Tuple[int, int],
    num_classes: int
) -> tf.keras.Model:
    """
    Create a Transformer model.
    
    Args:
        config: Model configuration dictionary
        input_shape: Shape of input data (window_length, n_features)
        num_classes: Number of output classes
    
    Returns:
        Compiled Keras model
    """
    # Get parameters
    transformer_config = config.get('transformer', {})
    num_heads = transformer_config.get('num_heads', 4)
    ff_dim = transformer_config.get('ff_dim', 128)
    num_transformer_blocks = transformer_config.get('num_transformer_blocks', 2)
    mlp_units = transformer_config.get('mlp_units', [64, 32])
    mlp_dropout = transformer_config.get('mlp_dropout', 0.3)
    attention_dropout = transformer_config.get('attention_dropout', 0.2)
    
    dropout_rate = config.get('dropout_rate', 0.3)
    activation = config.get('activation', 'relu')
    use_batch_norm = config.get('use_batch_norm', True)
    
    # Create model
    inputs = tf.keras.layers.Input(shape=input_shape)
    x = inputs
    
    # Add transformer blocks
    for _ in range(num_transformer_blocks):
        # Multi-head attention
        attention_output = tf.keras.layers.MultiHeadAttention(
            num_heads=num_heads,
            key_dim=input_shape[1],
            dropout=attention_dropout
        )(x, x)
        
        # Skip connection 1
        x = tf.keras.layers.Add()([attention_output, x])
        
        # Layer normalization 1
        x = tf.keras.layers.LayerNormalization(epsilon=1e-6)(x)
        
        # Feed-forward network
        ffn = tf.keras.Sequential([
            tf.keras.layers.Dense(ff_dim, activation=activation),
            tf.keras.layers.Dense(input_shape[1])
        ])
        
        # Skip connection 2
        x = tf.keras.layers.Add()([x, ffn(x)])
        
        # Layer normalization 2
        x = tf.keras.layers.LayerNormalization(epsilon=1e-6)(x)
    
    # Global pooling
    x = tf.keras.layers.GlobalAveragePooling1D()(x)
    
    # Add MLP layers
    for units in mlp_units:
        x = tf.keras.layers.Dense(units, activation=activation)(x)
        
        if use_batch_norm:
            x = tf.keras.layers.BatchNormalization()(x)
        
        if mlp_dropout > 0:
            x = tf.keras.layers.Dropout(mlp_dropout)(x)
    
    # Add output layer
    if num_classes > 2:
        outputs = tf.keras.layers.Dense(num_classes, activation='softmax')(x)
    else:
        outputs = tf.keras.layers.Dense(1, activation='sigmoid')(x)
    
    # Create model
    model = tf.keras.Model(inputs=inputs, outputs=outputs)
    
    # Compile model
    if num_classes > 2:
        model.compile(
            optimizer='adam',
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy']
        )
    else:
        model.compile(
            optimizer='adam',
            loss='binary_crossentropy',
            metrics=['accuracy']
        )
    
    return model

def create_resnet_model(
    config: Dict[str, Any],
    input_shape: Tuple[int, int],
    num_classes: int
) -> tf.keras.Model:
    """
    Create a ResNet model for time series.
    
    Args:
        config: Model configuration dictionary
        input_shape: Shape of input data (window_length, n_features)
        num_classes: Number of output classes
    
    Returns:
        Compiled Keras model
    """
    # Get parameters
    resnet_config = config.get('resnet', {})
    filters = resnet_config.get('filters', [64, 128, 256])
    kernel_size = resnet_config.get('kernel_size', 3)
    conv_shortcut = resnet_config.get('conv_shortcut', True)
    use_bias = resnet_config.get('use_bias', True)
    blocks_per_stack = resnet_config.get('blocks_per_stack', 2)
    
    dropout_rate = config.get('dropout_rate', 0.3)
    activation = config.get('activation', 'relu')
    use_batch_norm = config.get('use_batch_norm', True)
    layer_sizes = config.get('layer_sizes', [64, 32])
    
    # Create model
    inputs = tf.keras.layers.Input(shape=input_shape)
    x = inputs
    
    # Initial convolution
    x = tf.keras.layers.Conv1D(
        filters=filters[0],
        kernel_size=kernel_size,
        padding='same',
        use_bias=use_bias
    )(x)
    
    if use_batch_norm:
        x = tf.keras.layers.BatchNormalization()(x)
    
    x = tf.keras.layers.Activation(activation)(x)
    
    # Residual blocks
    for i, filter_size in enumerate(filters):
        # Add blocks for this filter size
        for j in range(blocks_per_stack):
            # Residual block
            residual = x
            
            # First convolution
            x = tf.keras.layers.Conv1D(
                filters=filter_size,
                kernel_size=kernel_size,
                padding='same',
                use_bias=use_bias
            )(x)
            
            if use_batch_norm:
                x = tf.keras.layers.BatchNormalization()(x)
            
            x = tf.keras.layers.Activation(activation)(x)
            
            # Second convolution
            x = tf.keras.layers.Conv1D(
                filters=filter_size,
                kernel_size=kernel_size,
                padding='same',
                use_bias=use_bias
            )(x)
            
            if use_batch_norm:
                x = tf.keras.layers.BatchNormalization()(x)
            
            # Shortcut connection
            if j == 0 and i > 0:
                # Downsample if this is the first block of a new filter size
                if conv_shortcut:
                    residual = tf.keras.layers.Conv1D(
                        filters=filter_size,
                        kernel_size=1,
                        padding='same',
                        use_bias=use_bias
                    )(residual)
                    
                    if use_batch_norm:
                        residual = tf.keras.layers.BatchNormalization()(residual)
                else:
                    residual = tf.keras.layers.MaxPooling1D(pool_size=2)(residual)
                    residual = tf.keras.layers.ZeroPadding1D(
                        padding=((0, 0), (0, filter_size - filters[i-1]))
                    )(residual)
            
            # Add residual connection
            x = tf.keras.layers.Add()([x, residual])
            x = tf.keras.layers.Activation(activation)(x)
        
        # Add pooling after each stack
        if i < len(filters) - 1:
            x = tf.keras.layers.MaxPooling1D(pool_size=2)(x)
    
    # Global pooling
    x = tf.keras.layers.GlobalAveragePooling1D()(x)
    
    # Add dense layers
    for units in layer_sizes:
        x = tf.keras.layers.Dense(units, activation=activation)(x)
        
        if use_batch_norm:
            x = tf.keras.layers.BatchNormalization()(x)
        
        if dropout_rate > 0:
            x = tf.keras.layers.Dropout(dropout_rate)(x)
    
    # Add output layer
    if num_classes > 2:
        outputs = tf.keras.layers.Dense(num_classes, activation='softmax')(x)
    else:
        outputs = tf.keras.layers.Dense(1, activation='sigmoid')(x)
    
    # Create model
    model = tf.keras.Model(inputs=inputs, outputs=outputs)
    
    # Compile model
    if num_classes > 2:
        model.compile(
            optimizer='adam',
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy']
        )
    else:
        model.compile(
            optimizer='adam',
            loss='binary_crossentropy',
            metrics=['accuracy']
        )
    
    return model

def get_optimizer(
    optimizer_name: str,
    learning_rate: float,
    optimizer_params: Dict[str, Any]
) -> tf.keras.optimizers.Optimizer:
    """
    Get optimizer based on name and parameters.
    
    Args:
        optimizer_name: Name of the optimizer
        learning_rate: Learning rate
        optimizer_params: Additional optimizer parameters
    
    Returns:
        Keras optimizer
    """
    optimizer_name = optimizer_name.lower()
    params = optimizer_params.get(optimizer_name, {})
    
    if optimizer_name == 'adam':
        return tf.keras.optimizers.Adam(
            learning_rate=learning_rate,
            beta_1=params.get('beta_1', 0.9),
            beta_2=params.get('beta_2', 0.999),
            epsilon=params.get('epsilon', 1e-7),
            amsgrad=params.get('amsgrad', False)
        )
    elif optimizer_name == 'sgd':
        return tf.keras.optimizers.SGD(
            learning_rate=learning_rate,
            momentum=params.get('momentum', 0.0),
            nesterov=params.get('nesterov', False)
        )
    elif optimizer_name == 'rmsprop':
        return tf.keras.optimizers.RMSprop(
            learning_rate=learning_rate,
            rho=params.get('rho', 0.9),
            momentum=params.get('momentum', 0.0),
            epsilon=params.get('epsilon', 1e-7)
        )
    elif optimizer_name == 'adagrad':
        return tf.keras.optimizers.Adagrad(
            learning_rate=learning_rate,
            initial_accumulator_value=params.get('initial_accumulator_value', 0.1),
            epsilon=params.get('epsilon', 1e-7)
        )
    elif optimizer_name == 'adadelta':
        return tf.keras.optimizers.Adadelta(
            learning_rate=learning_rate,
            rho=params.get('rho', 0.95),
            epsilon=params.get('epsilon', 1e-7)
        )
    elif optimizer_name == 'adamax':
        return tf.keras.optimizers.Adamax(
            learning_rate=learning_rate,
            beta_1=params.get('beta_1', 0.9),
            beta_2=params.get('beta_2', 0.999),
            epsilon=params.get('epsilon', 1e-7)
        )
    elif optimizer_name == 'nadam':
        return tf.keras.optimizers.Nadam(
            learning_rate=learning_rate,
            beta_1=params.get('beta_1', 0.9),
            beta_2=params.get('beta_2', 0.999),
            epsilon=params.get('epsilon', 1e-7)
        )
    else:
        raise ValueError(f"Unknown optimizer: {optimizer_name}")

def get_learning_rate_scheduler(
    config: Dict[str, Any]
) -> Optional[tf.keras.callbacks.Callback]:
    """
    Get learning rate scheduler based on configuration.
    
    Args:
        config: Training configuration dictionary
    
    Returns:
        Learning rate scheduler callback or None
    """
    lr_config = config.get('lr_scheduler', {})
    scheduler_type = lr_config.get('type')
    
    if not scheduler_type:
        return None
    
    if scheduler_type == 'reduce_on_plateau':
        return tf.keras.callbacks.ReduceLROnPlateau(
            monitor=lr_config.get('monitor', 'val_loss'),
            factor=lr_config.get('factor', 0.5),
            patience=lr_config.get('patience', 5),
            min_lr=lr_config.get('min_lr', 0.00001),
            cooldown=lr_config.get('cooldown', 0),
            verbose=lr_config.get('verbose', 1)
        )
    elif scheduler_type == 'cosine_decay':
        initial_lr = config.get('learning_rate', 0.001)
        decay_steps = lr_config.get('decay_steps', 1000)
        
        return tf.keras.callbacks.LearningRateScheduler(
            lambda epoch: initial_lr * (1 + np.cos(np.pi * epoch / decay_steps)) / 2
        )
    elif scheduler_type == 'exponential_decay':
        initial_lr = config.get('learning_rate', 0.001)
        decay_rate = lr_config.get('decay_rate', 0.9)
        decay_steps = lr_config.get('decay_steps', 1000)
        
        return tf.keras.callbacks.LearningRateScheduler(
            lambda epoch: initial_lr * decay_rate ** (epoch / decay_steps)
        )
    elif scheduler_type == 'step_decay':
        initial_lr = config.get('learning_rate', 0.001)
        drop_rate = lr_config.get('drop_rate', 0.5)
        epochs_drop = lr_config.get('epochs_drop', 10)
        
        return tf.keras.callbacks.LearningRateScheduler(
            lambda epoch: initial_lr * (drop_rate ** np.floor(epoch / epochs_drop))
        )
    elif scheduler_type == 'custom':
        # Custom learning rate schedule should be defined in the configuration
        schedule_fn = lr_config.get('schedule_fn')
        if schedule_fn is None:
            raise ValueError("Custom learning rate scheduler requires 'schedule_fn'")
        
        return tf.keras.callbacks.LearningRateScheduler(schedule_fn)
    else:
        raise ValueError(f"Unknown scheduler type: {scheduler_type}")

def save_model_with_metadata(
    model: tf.keras.Model,
    output_dir: str,
    model_name: str,
    config: Dict[str, Any],
    input_shape: Tuple[int, int],
    num_classes: int,
    class_names: List[str] = None,
    feature_names: List[str] = None
) -> str:
    """
    Save model with metadata.
    
    Args:
        model: Trained Keras model
        output_dir: Directory to save model
        model_name: Name of the model
        config: Model configuration dictionary
        input_shape: Shape of input data
        num_classes: Number of output classes
        class_names: List of class names
        feature_names: List of feature names
    
    Returns:
        Path to saved model
    """
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Save model
    model_path = os.path.join(output_dir, f"{model_name}.h5")
    model.save(model_path)
    
    # Prepare metadata
    metadata = {
        'architecture': config.get('architecture', 'unknown'),
        'input_shape': input_shape,
        'num_classes': num_classes,
        'class_names': class_names if class_names else [f"class_{i}" for i in range(num_classes)],
        'feature_names': feature_names if feature_names else [f"feature_{i}" for i in range(input_shape[1])],
        'config': config,
        'creation_date': pd.Timestamp.now().isoformat()
    }
    
    # Save metadata
    metadata_path = os.path.join(output_dir, f"{model_name}_metadata.json")
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)
    
    return model_path

def load_model_with_metadata(
    model_path: str
) -> Tuple[tf.keras.Model, Dict[str, Any]]:
    """
    Load model and its metadata.
    
    Args:
        model_path: Path to saved model
    
    Returns:
        model: Loaded Keras model
        metadata: Model metadata
    """
    # Load model
    model = tf.keras.models.load_model(model_path)
    
    # Load metadata
    metadata_path = os.path.splitext(model_path)[0] + '_metadata.json'
    if os.path.exists(metadata_path):
        with open(metadata_path, 'r') as f:
            metadata = json.load(f)
    else:
        # Create basic metadata from model
        metadata = {
            'architecture': 'unknown',
            'input_shape': model.input_shape[1:],
            'num_classes': model.output_shape[1] if model.output_shape[1] > 1 else 2,
            'class_names': None,
            'feature_names': None,
            'config': {},
            'creation_date': None
        }
    
    return model, metadata

def get_model_summary(model: tf.keras.Model) -> str:
    """
    Get model summary as a string.
    
    Args:
        model: Keras model
    
    Returns:
        Model summary string
    """
    # Create string buffer
    import io
    buffer = io.StringIO()
    
    # Write model summary to buffer
    model.summary(print_fn=lambda x: buffer.write(x + '\n'))
    
    # Get string value
    summary = buffer.getvalue()
    buffer.close()
    
    return summary

def get_model_size(model: tf.keras.Model) -> Dict[str, int]:
    """
    Get model size information.
    
    Args:
        model: Keras model
    
    Returns:
        Dictionary containing model size information
    """
    # Get total parameters
    total_params = model.count_params()
    
    # Get trainable parameters
    trainable_params = sum([tf.keras.backend.count_params(w) for w in model.trainable_weights])
    
    # Get non-trainable parameters
    non_trainable_params = sum([tf.keras.backend.count_params(w) for w in model.non_trainable_weights])
    
    # Get model size in bytes (approximate)
    import sys
    param_size = 4  # Assuming float32
    model_size = total_params * param_size
    
    return {
        'total_parameters': total_params,
        'trainable_parameters': trainable_params,
        'non_trainable_parameters': non_trainable_params,
        'model_size_bytes': model_size
    }