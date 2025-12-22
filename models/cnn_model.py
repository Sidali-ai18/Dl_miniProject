"""
CNN model architecture for facial emotion recognition.
Simple CNN designed for learning fundamentals of deep learning.
"""

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import (
    Conv2D, MaxPooling2D, Flatten, Dense, Dropout, 
    BatchNormalization, Activation
)
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
import os


def create_simple_cnn(input_shape=(48, 48, 1), num_classes=7):
    """
    Create a simple CNN model for emotion recognition.
    This is designed for learning CNN fundamentals.
    
    Args:
        input_shape: Input image shape (height, width, channels)
        num_classes: Number of emotion classes
        
    Returns:
        Compiled Keras model
    """
    model = Sequential([
        # First Convolutional Block
        Conv2D(32, (3, 3), padding='same', input_shape=input_shape),
        BatchNormalization(),
        Activation('relu'),
        Conv2D(32, (3, 3), padding='same'),
        BatchNormalization(),
        Activation('relu'),
        MaxPooling2D(pool_size=(2, 2)),
        Dropout(0.25),
        
        # Second Convolutional Block
        Conv2D(64, (3, 3), padding='same'),
        BatchNormalization(),
        Activation('relu'),
        Conv2D(64, (3, 3), padding='same'),
        BatchNormalization(),
        Activation('relu'),
        MaxPooling2D(pool_size=(2, 2)),
        Dropout(0.25),
        
        # Third Convolutional Block
        Conv2D(128, (3, 3), padding='same'),
        BatchNormalization(),
        Activation('relu'),
        Conv2D(128, (3, 3), padding='same'),
        BatchNormalization(),
        Activation('relu'),
        MaxPooling2D(pool_size=(2, 2)),
        Dropout(0.25),
        
        # Fully Connected Layers
        Flatten(),
        Dense(256, activation='relu'),
        BatchNormalization(),
        Dropout(0.5),
        Dense(128, activation='relu'),
        BatchNormalization(),
        Dropout(0.5),
        Dense(num_classes, activation='softmax')
    ])
    
    return model


def create_basic_cnn(input_shape=(48, 48, 1), num_classes=7):
    """
    Create a very basic CNN model for beginners.
    Simpler architecture to understand CNN fundamentals.
    
    Args:
        input_shape: Input image shape (height, width, channels)
        num_classes: Number of emotion classes
        
    Returns:
        Compiled Keras model
    """
    model = Sequential([
        # First Conv Layer
        Conv2D(32, (3, 3), activation='relu', input_shape=input_shape),
        MaxPooling2D(pool_size=(2, 2)),
        
        # Second Conv Layer
        Conv2D(64, (3, 3), activation='relu'),
        MaxPooling2D(pool_size=(2, 2)),
        
        # Third Conv Layer
        Conv2D(128, (3, 3), activation='relu'),
        MaxPooling2D(pool_size=(2, 2)),
        
        # Fully Connected
        Flatten(),
        Dense(128, activation='relu'),
        Dropout(0.5),
        Dense(num_classes, activation='softmax')
    ])
    
    return model


def compile_model(model, learning_rate=0.001):
    """
    Compile the model with optimizer and loss function.
    
    Args:
        model: Keras model to compile
        learning_rate: Learning rate for optimizer
        
    Returns:
        Compiled model
    """
    model.compile(
        optimizer=Adam(learning_rate=learning_rate),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    
    return model


def get_callbacks(model_save_path='models/best_model.h5', patience=10):
    """
    Create callbacks for model training.
    
    Args:
        model_save_path: Path to save the best model
        patience: Patience for early stopping
        
    Returns:
        List of callbacks
    """
    # Ensure directory exists
    os.makedirs(os.path.dirname(model_save_path), exist_ok=True)
    
    callbacks = [
        # Save the best model
        ModelCheckpoint(
            model_save_path,
            monitor='val_accuracy',
            mode='max',
            save_best_only=True,
            verbose=1
        ),
        
        # Early stopping
        EarlyStopping(
            monitor='val_loss',
            patience=patience,
            restore_best_weights=True,
            verbose=1
        ),
        
        # Reduce learning rate on plateau
        ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=5,
            min_lr=1e-7,
            verbose=1
        )
    ]
    
    return callbacks


def print_model_summary(model):
    """Print model architecture summary."""
    print("\n" + "="*50)
    print("MODEL ARCHITECTURE")
    print("="*50)
    model.summary()
    print("="*50 + "\n")
