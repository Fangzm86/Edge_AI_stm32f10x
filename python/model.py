import tensorflow as tf
import pandas as pd
from keras import layers, models
import keras
import os
import numpy as np
import matplotlib.pyplot as plt

class MyCNNModel(keras.Model):
    def __init__(self, input_shape, num_classes):
        super(MyCNNModel, self).__init__()
        self.conv1 = layers.Conv1D(filters=64, kernel_size=3, activation='relu', input_shape=input_shape)
        self.pool1 = layers.MaxPooling1D(pool_size=2)
        self.conv2 = layers.Conv1D(filters=128, kernel_size=3, activation='relu')
        self.pool2 = layers.MaxPooling1D(pool_size=2)
        self.flatten = layers.Flatten()
        self.dense1 = layers.Dense(128, activation='relu')
        self.dropout = layers.Dropout(0.3)
        self.output_layer = layers.Dense(num_classes, activation='softmax')

    def call(self, inputs):
        """
        Processes input through the model's layers sequentially:
        1. Conv1 -> Pool1 -> Conv2 -> Pool2
        2. Flatten -> Dense1 -> Dropout
        3. Output layer
        
        Args:
            inputs: Input tensor to process through the network
        
        Returns:
            Processed output tensor after passing through all layers
        """
        x = self.conv1(inputs)
        x = self.pool1(x)
        x = self.conv2(x)
        x = self.pool2(x)
        x = self.flatten(x)
        x = self.dense1(x)
        x = self.dropout(x)
        return self.output_layer(x)

    def compile_model(self, 
                      loss='categorical_crossentropy', 
                      learning_rate=0.001,
                      metrics=['accuracy']):
        self.compile(optimizer=keras.optimizers.Adam(learning_rate=learning_rate),
                     loss=loss,
                     metrics=metrics)
    
    def train(self, 
              x_train, 
              y_train, 
              validation_split,
              epochs=10, 
              batch_size=32, 
              callbacks=None, 
              class_weights=None, 
              save_dir=None):
        """
        Train the model on the provided data.
        
        Args:
            x_train: Training data
            y_train: Training labels
            x_val: Validation data (optional)
            y_val: Validation labels (optional)
            epochs: Number of training epochs
            batch_size: Batch size for training
            callbacks: List of Keras callbacks (optional)
            class_weights: Class weights for imbalanced data (optional)
            save_dir: Directory to save model and training results (optional)
            
        Returns:
            Training history
        """
        # Create validation data tuple if provided

        # Create default callbacks if none provided
        if callbacks is None:
            callbacks = []
            
            # Add early stopping
            callbacks.append(keras.callbacks.EarlyStopping(
                monitor='val_loss' if validation_split else 'loss',
                patience=5,
                restore_best_weights=True
            ))
        
        # Train the model
        history = self.fit(
            x_train, y_train,
            epochs=epochs,
            batch_size=batch_size,
            validation_split=validation_split,
            callbacks=callbacks,
            class_weight=class_weights,
            verbose=1
        )
        
        # Save model if save_dir is provided
        if save_dir:
            os.makedirs(save_dir, exist_ok=True)
            self.save(os.path.join(save_dir, 'model.h5'))
            
            # Plot and save training history
            self._plot_training_history(history, save_dir)
        
        return history
    
    def _plot_training_history(self, history, save_dir):
        """
        Plot and save training history.
        
        Args:
            history: Training history
            save_dir: Directory to save plots
        """
        # Plot training & validation accuracy
        plt.figure(figsize=(12, 4))
        
        plt.subplot(1, 2, 1)
        plt.plot(history.history['accuracy'])
        if 'val_accuracy' in history.history:
            plt.plot(history.history['val_accuracy'])
        plt.title('Model Accuracy')
        plt.ylabel('Accuracy')
        plt.xlabel('Epoch')
        plt.legend(['Train', 'Validation'], loc='upper left')
        
        plt.subplot(1, 2, 2)
        plt.plot(history.history['loss'])
        if 'val_loss' in history.history:
            plt.plot(history.history['val_loss'])
        plt.title('Model Loss')
        plt.ylabel('Loss')
        plt.xlabel('Epoch')
        plt.legend(['Train', 'Validation'], loc='upper left')
        
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, 'training_history.png'))
        plt.close()



# import tensorflow as tf
# from tensorflow.keras import layers
# import os
# import matplotlib.pyplot as plt

# class MyCNNModel(tf.keras.Model):
#     def __init__(self, input_shape, num_classes):
#         super(MyCNNModel, self).__init__()
#         # 定义网络层列表，不在层构造中指明 input_shape
#         self.conv1 = layers.Conv1D(64, 3, activation='relu')
#         self.pool1 = layers.MaxPooling1D(pool_size=2)
#         self.conv2 = layers.Conv1D(128, 3, activation='relu')
#         self.pool2 = layers.MaxPooling1D(pool_size=2)
#         self.flatten = layers.Flatten()
#         self.dense1 = layers.Dense(128, activation='relu')
#         self.dropout = layers.Dropout(0.3)
#         self.output_layer = layers.Dense(num_classes, activation='softmax')

#         # 使用 build() 方法明确指定输入形状，确保序列化兼容性
#         self.build((None, *input_shape))

#     def call(self, inputs, training=False):
#         x = self.conv1(inputs)
#         x = self.pool1(x)
#         x = self.conv2(x)
#         x = self.pool2(x)
#         x = self.flatten(x)
#         x = self.dense1(x)
#         x = self.dropout(x, training=training)
#         return self.output_layer(x)

#     def compile_model(self, learning_rate=0.001):
#         self.compile(
#             optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
#             loss='sparse_categorical_crossentropy',
#             metrics=['accuracy']
#         )

#     def train(self, x_train, y_train,
#               x_val=None, y_val=None,
#               epochs=10, batch_size=32,
#               callbacks=None, class_weights=None,
#               save_dir=None):
#         # 准备验证集
#         validation_data = (x_val, y_val) if x_val is not None and y_val is not None else None

#         # 默认回调
#         if callbacks is None:
#             callbacks = [
#                 tf.keras.callbacks.EarlyStopping(
#                     monitor='val_loss' if validation_data else 'loss',
#                     patience=5,
#                     restore_best_weights=True
#                 )
#             ]

#         # 训练模型
#         history = self.fit(
#             x_train, y_train,
#             validation_data=validation_data,
#             epochs=epochs,
#             batch_size=batch_size,
#             callbacks=callbacks,
#             class_weight=class_weights,
#             verbose=1
#         )

#         # 保存模型并绘制历史
#         if save_dir:
#             os.makedirs(save_dir, exist_ok=True)
#             # 推荐使用 SavedModel 格式，便于 TFLite 转换
#             tf.keras.models.save_model(
#                 self,
#                 os.path.join(save_dir, 'saved_model'),
#                 save_format='tf'
#             )
#             self._plot_training_history(history, save_dir)

#         return history

#     def _plot_training_history(self, history, save_dir):
#         plt.figure(figsize=(12, 4))

#         plt.subplot(1, 2, 1)
#         plt.plot(history.history['accuracy'], label='Train')
#         if 'val_accuracy' in history.history:
#             plt.plot(history.history['val_accuracy'], label='Validation')
#         plt.title('Model Accuracy')
#         plt.xlabel('Epoch')
#         plt.ylabel('Accuracy')
#         plt.legend()

#         plt.subplot(1, 2, 2)
#         plt.plot(history.history['loss'], label='Train')
#         if 'val_loss' in history.history:
#             plt.plot(history.history['val_loss'], label='Validation')
#         plt.title('Model Loss')
#         plt.xlabel('Epoch')
#         plt.ylabel('Loss')
#         plt.legend()

#         plt.tight_layout()
#         plt.savefig(os.path.join(save_dir, 'training_history.png'))
#         plt.close()
