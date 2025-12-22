"""
Training script for facial emotion recognition model.
"""

import os
import sys
import argparse
import matplotlib.pyplot as plt
import numpy as np

# Ensure project root is in path for imports
_project_root = os.path.dirname(os.path.abspath(__file__))
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

from models.cnn_model import (
    create_simple_cnn, create_basic_cnn, compile_model, 
    get_callbacks, print_model_summary
)
from utils.preprocessing import prepare_dataset_from_directory


def plot_training_history(history, save_path='results/training_history.png'):
    """
    Plot training history (accuracy and loss).
    
    Args:
        history: Training history object
        save_path: Path to save the plot
    """
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    
    # Plot accuracy
    axes[0].plot(history.history['accuracy'], label='Train Accuracy')
    if 'val_accuracy' in history.history:
        axes[0].plot(history.history['val_accuracy'], label='Val Accuracy')
    axes[0].set_title('Model Accuracy')
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Accuracy')
    axes[0].legend()
    axes[0].grid(True)
    
    # Plot loss
    axes[1].plot(history.history['loss'], label='Train Loss')
    if 'val_loss' in history.history:
        axes[1].plot(history.history['val_loss'], label='Val Loss')
    axes[1].set_title('Model Loss')
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Loss')
    axes[1].legend()
    axes[1].grid(True)
    
    plt.tight_layout()
    plt.savefig(save_path)
    print(f"\nTraining history plot saved to: {save_path}")
    plt.close()


def train_model(
    train_dir,
    test_dir=None,
    model_type='simple',
    epochs=50,
    batch_size=32,
    learning_rate=0.001,
    image_size=48
):
    """
    Train the emotion recognition model.
    
    Args:
        train_dir: Directory containing training images
        test_dir: Directory containing test images (optional)
        model_type: Type of model ('simple' or 'basic')
        epochs: Number of training epochs
        batch_size: Batch size for training
        learning_rate: Learning rate for optimizer
        image_size: Size of input images
    """
    print("\n" + "="*60)
    print("FACIAL EMOTION RECOGNITION - TRAINING")
    print("="*60)
    
    # Check if training directory exists
    if not os.path.exists(train_dir):
        print(f"\nError: Training directory not found: {train_dir}")
        print("\nPlease organize your images in the following structure:")
        print("  data/train/")
        print("    emotion1/")
        print("      image1.jpg")
        print("      image2.jpg")
        print("    emotion2/")
        print("      image1.jpg")
        print("      image2.jpg")
        return
    
    # Prepare data generators
    print("\nPreparing data generators...")
    target_size = (image_size, image_size)
    
    train_generator = prepare_dataset_from_directory(
        train_dir,
        target_size=target_size,
        batch_size=batch_size,
        augment=True
    )
    
    num_classes = len(train_generator.class_indices)
    class_names = list(train_generator.class_indices.keys())
    
    print(f"\nNumber of classes: {num_classes}")
    print(f"Class names: {class_names}")
    print(f"Number of training samples: {train_generator.samples}")
    
    # Prepare test generator if test directory provided
    test_generator = None
    if test_dir and os.path.exists(test_dir):
        test_generator = prepare_dataset_from_directory(
            test_dir,
            target_size=target_size,
            batch_size=batch_size,
            augment=False
        )
        print(f"Number of test samples: {test_generator.samples}")
    
    # Create model
    print(f"\nCreating {model_type} CNN model...")
    input_shape = (image_size, image_size, 1)
    
    if model_type == 'basic':
        model = create_basic_cnn(input_shape=input_shape, num_classes=num_classes)
    else:
        model = create_simple_cnn(input_shape=input_shape, num_classes=num_classes)
    
    model = compile_model(model, learning_rate=learning_rate)
    print_model_summary(model)
    
    # Get callbacks
    callbacks = get_callbacks(
        model_save_path='models/emotion_recognition_model.h5',
        patience=10
    )
    
    # Train model
    print("\nStarting training...")
    print(f"Epochs: {epochs}")
    print(f"Batch size: {batch_size}")
    print(f"Learning rate: {learning_rate}\n")
    
    history = model.fit(
        train_generator,
        epochs=epochs,
        validation_data=test_generator,
        callbacks=callbacks,
        verbose=1
    )
    
    # Plot training history
    plot_training_history(history)
    
    # Final evaluation
    print("\n" + "="*60)
    print("TRAINING COMPLETED")
    print("="*60)
    
    if test_generator:
        test_loss, test_acc = model.evaluate(test_generator, verbose=0)
        print(f"\nFinal Test Accuracy: {test_acc*100:.2f}%")
        print(f"Final Test Loss: {test_loss:.4f}")
    
    final_train_acc = history.history['accuracy'][-1]
    final_train_loss = history.history['loss'][-1]
    print(f"\nFinal Train Accuracy: {final_train_acc*100:.2f}%")
    print(f"Final Train Loss: {final_train_loss:.4f}")
    
    print(f"\nModel saved to: models/emotion_recognition_model.h5")
    print("="*60 + "\n")


def main():
    parser = argparse.ArgumentParser(description='Train facial emotion recognition model')
    parser.add_argument('--train_dir', type=str, default='data/train',
                        help='Directory containing training images')
    parser.add_argument('--test_dir', type=str, default='data/test',
                        help='Directory containing test images')
    parser.add_argument('--model_type', type=str, default='simple',
                        choices=['simple', 'basic'],
                        help='Model architecture type')
    parser.add_argument('--epochs', type=int, default=50,
                        help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=32,
                        help='Batch size for training')
    parser.add_argument('--learning_rate', type=float, default=0.001,
                        help='Learning rate')
    parser.add_argument('--image_size', type=int, default=48,
                        help='Input image size')
    
    args = parser.parse_args()
    
    train_model(
        train_dir=args.train_dir,
        test_dir=args.test_dir,
        model_type=args.model_type,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        image_size=args.image_size
    )


if __name__ == '__main__':
    main()
