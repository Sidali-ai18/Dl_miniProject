"""
Helper script to prepare data for emotion recognition.
Creates directory structure and provides utilities for data organization.
"""

import os
import shutil
import argparse


def create_data_structure(base_dir='data', emotions=None):
    """
    Create directory structure for organizing emotion images.
    
    Args:
        base_dir: Base directory for data
        emotions: List of emotion names
    """
    if emotions is None:
        emotions = ['happy', 'sad', 'neutral', 'angry', 'surprised', 'fear', 'disgust']
    
    # Create train and test directories
    for split in ['train', 'test']:
        split_dir = os.path.join(base_dir, split)
        
        for emotion in emotions:
            emotion_dir = os.path.join(split_dir, emotion)
            os.makedirs(emotion_dir, exist_ok=True)
            
            # Create a README file in each emotion directory
            readme_path = os.path.join(emotion_dir, 'README.txt')
            with open(readme_path, 'w') as f:
                f.write(f"Place {emotion} emotion images here.\n")
                f.write(f"Examples: person1_{emotion}_1.jpg, person2_{emotion}_1.jpg\n")
    
    print(f"\n✓ Created directory structure in '{base_dir}'")
    print(f"\nEmotion categories created: {', '.join(emotions)}")
    print(f"\nNext steps:")
    print(f"1. Take photos of different emotions")
    print(f"2. Place 70-80% of images in data/train/<emotion>/")
    print(f"3. Place 20-30% of images in data/test/<emotion>/")
    print(f"4. Run: python train.py\n")


def split_data(source_dir, train_dir, test_dir, split_ratio=0.8):
    """
    Split data from source directory into train and test sets.
    
    Args:
        source_dir: Directory containing subdirectories of emotions with images
        train_dir: Directory to store training images
        test_dir: Directory to store test images
        split_ratio: Ratio of training data (default: 0.8)
    """
    import random
    
    if not os.path.exists(source_dir):
        print(f"Error: Source directory not found: {source_dir}")
        return
    
    # Get all emotion subdirectories
    emotions = [d for d in os.listdir(source_dir) 
                if os.path.isdir(os.path.join(source_dir, d))]
    
    if not emotions:
        print(f"Error: No emotion subdirectories found in {source_dir}")
        return
    
    print(f"\nSplitting data from '{source_dir}'")
    print(f"Train ratio: {split_ratio*100:.0f}%")
    print(f"Test ratio: {(1-split_ratio)*100:.0f}%\n")
    
    for emotion in emotions:
        emotion_source = os.path.join(source_dir, emotion)
        
        # Get all images in this emotion directory
        images = [f for f in os.listdir(emotion_source)
                 if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp'))]
        
        if not images:
            print(f"⚠ No images found in {emotion_source}")
            continue
        
        # Shuffle images
        random.shuffle(images)
        
        # Split into train and test
        split_idx = int(len(images) * split_ratio)
        train_images = images[:split_idx]
        test_images = images[split_idx:]
        
        # Create emotion directories
        train_emotion_dir = os.path.join(train_dir, emotion)
        test_emotion_dir = os.path.join(test_dir, emotion)
        os.makedirs(train_emotion_dir, exist_ok=True)
        os.makedirs(test_emotion_dir, exist_ok=True)
        
        # Copy train images
        for img in train_images:
            src = os.path.join(emotion_source, img)
            dst = os.path.join(train_emotion_dir, img)
            shutil.copy2(src, dst)
        
        # Copy test images
        for img in test_images:
            src = os.path.join(emotion_source, img)
            dst = os.path.join(test_emotion_dir, img)
            shutil.copy2(src, dst)
        
        print(f"✓ {emotion:12s}: {len(train_images)} train, {len(test_images)} test")
    
    print(f"\n✓ Data split complete!")
    print(f"  Train: {train_dir}")
    print(f"  Test:  {test_dir}\n")


def count_images(data_dir):
    """
    Count images in each emotion category.
    
    Args:
        data_dir: Directory containing emotion subdirectories
    """
    if not os.path.exists(data_dir):
        print(f"Error: Directory not found: {data_dir}")
        return
    
    print(f"\nImage count in '{data_dir}':\n")
    
    total = 0
    emotions = [d for d in os.listdir(data_dir)
                if os.path.isdir(os.path.join(data_dir, d))]
    
    for emotion in sorted(emotions):
        emotion_dir = os.path.join(data_dir, emotion)
        images = [f for f in os.listdir(emotion_dir)
                 if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp'))]
        count = len(images)
        total += count
        print(f"  {emotion:12s}: {count:3d} images")
    
    print(f"\n  Total: {total} images\n")


def main():
    parser = argparse.ArgumentParser(description='Data preparation utilities')
    parser.add_argument('command', choices=['create', 'split', 'count'],
                       help='Command to run')
    parser.add_argument('--base_dir', default='data',
                       help='Base directory for data')
    parser.add_argument('--source_dir', help='Source directory for splitting')
    parser.add_argument('--train_dir', default='data/train',
                       help='Training directory')
    parser.add_argument('--test_dir', default='data/test',
                       help='Test directory')
    parser.add_argument('--split_ratio', type=float, default=0.8,
                       help='Train/test split ratio')
    parser.add_argument('--emotions', nargs='+',
                       help='List of emotion names')
    
    args = parser.parse_args()
    
    if args.command == 'create':
        create_data_structure(args.base_dir, args.emotions)
    elif args.command == 'split':
        if not args.source_dir:
            print("Error: --source_dir required for split command")
            return
        split_data(args.source_dir, args.train_dir, args.test_dir, args.split_ratio)
    elif args.command == 'count':
        count_images(args.train_dir)
        count_images(args.test_dir)


if __name__ == '__main__':
    main()
