#!/usr/bin/env python3
"""
Quick training script specifically for Happy and Neutral emotion recognition.
This is a simplified version optimized for 2-class emotion recognition.
"""

import os
import sys

# Ensure project root is in path
_project_root = os.path.dirname(os.path.abspath(__file__))
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

from train import train_model

def main():
    """Train model for Happy and Neutral emotion recognition with optimal settings."""
    
    print("\n" + "="*70)
    print("   HAPPY & NEUTRAL EMOTION RECOGNITION - TRAINING")
    print("="*70)
    print("\nThis script will train a model to recognize Happy and Neutral emotions.")
    print("Make sure you have added images to:")
    print("  - data/train/happy/")
    print("  - data/train/neutral/")
    print("  - data/test/happy/")
    print("  - data/test/neutral/")
    print("\n" + "="*70 + "\n")
    
    # Check if data exists
    train_dir = 'data/train'
    test_dir = 'data/test'
    
    if not os.path.exists(os.path.join(train_dir, 'happy')) or \
       not os.path.exists(os.path.join(train_dir, 'neutral')):
        print("ERROR: Please create the data directories first!")
        print("Run: python prepare_data.py create --emotions happy neutral")
        return
    
    # Count images
    happy_train = len([f for f in os.listdir(os.path.join(train_dir, 'happy')) 
                       if f.lower().endswith(('.jpg', '.jpeg', '.png'))])
    neutral_train = len([f for f in os.listdir(os.path.join(train_dir, 'neutral')) 
                         if f.lower().endswith(('.jpg', '.jpeg', '.png'))])
    
    if happy_train == 0 or neutral_train == 0:
        print("ERROR: No images found in training directories!")
        print(f"  Happy images: {happy_train}")
        print(f"  Neutral images: {neutral_train}")
        print("\nPlease add your images to data/train/happy/ and data/train/neutral/")
        return
    
    print(f"Found training images:")
    print(f"  Happy: {happy_train} images")
    print(f"  Neutral: {neutral_train} images")
    print(f"  Total: {happy_train + neutral_train} images\n")
    
    if happy_train < 20 or neutral_train < 20:
        print("WARNING: You have less than 20 images per emotion.")
        print("Recommended: At least 30-40 images per emotion for good results.\n")
    
    # Optimal settings for 2-class problem
    print("Training with optimized settings for 2 emotions:\n")
    print("  Model type: basic (simpler, faster, good for 2 classes)")
    print("  Epochs: 30")
    print("  Batch size: 16")
    print("  Learning rate: 0.001")
    print("  Data augmentation: enabled")
    print("\n" + "="*70 + "\n")
    
    # Train model with optimal settings for 2 classes
    train_model(
        train_dir=train_dir,
        test_dir=test_dir,
        model_type='basic',  # Basic model works well for 2 classes
        epochs=30,           # 30 epochs is usually sufficient
        batch_size=16,       # Smaller batch size for small datasets
        learning_rate=0.001,
        image_size=48
    )
    
    print("\n" + "="*70)
    print("TRAINING COMPLETE!")
    print("="*70)
    print("\nYour model has been saved to: models/emotion_recognition_model.h5")
    print("\nNext steps:")
    print("  1. Test on single image:")
    print("     python predict.py --image data/test/happy/photo.jpg --labels happy neutral")
    print("\n  2. Test on webcam:")
    print("     python predict.py --webcam --labels happy neutral")
    print("\n  3. Batch test:")
    print("     python predict.py --batch_dir data/test/ --labels happy neutral")
    print("\n" + "="*70 + "\n")


if __name__ == '__main__':
    main()
