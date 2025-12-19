# Facial Emotion Recognition - Deep Learning Mini Project

A simple CNN-based facial emotion recognition system designed to help learn deep learning and CNN fundamentals.

## Project Overview

This mini-assessment project demonstrates building a facial emotion recognition model using Convolutional Neural Networks (CNNs). The project is designed for learning purposes and uses photos of you and your friend to classify different facial emotions.

## Features

- 🎯 **Face Detection**: Automatic face detection using OpenCV Haar Cascades
- 🧠 **CNN Models**: Two model architectures (basic and advanced) for learning
- 📊 **Data Augmentation**: Built-in augmentation to improve model generalization
- 🎥 **Real-time Prediction**: Webcam support for live emotion recognition
- 📓 **Interactive Notebook**: Jupyter notebook with detailed explanations
- 📈 **Training Visualization**: Automatic plotting of training metrics

## Project Structure

```
Dl_miniProject/
├── data/
│   ├── train/          # Training images organized by emotion
│   │   ├── happy/
│   │   ├── sad/
│   │   ├── neutral/
│   │   └── ...
│   └── test/           # Test images organized by emotion
├── models/
│   ├── cnn_model.py    # CNN model architectures
│   └── *.h5            # Saved trained models
├── utils/
│   └── preprocessing.py # Image preprocessing utilities
├── notebooks/
│   └── emotion_recognition_demo.ipynb  # Demo notebook
├── train.py            # Training script
├── predict.py          # Prediction script
├── requirements.txt    # Python dependencies
└── README.md           # This file
```

## Installation

1. **Clone the repository**:
```bash
git clone https://github.com/Sidali-ai18/Dl_miniProject.git
cd Dl_miniProject
```

2. **Install dependencies**:
```bash
pip install -r requirements.txt
```

## Data Preparation

### 1. Collect Images

Take photos of you and your friend showing different emotions:
- Happy
- Sad
- Neutral
- Angry
- Surprised
- (Add more emotions as needed)

### 2. Organize Images

Organize your images in the following directory structure:

```
data/
├── train/
│   ├── happy/
│   │   ├── person1_happy_1.jpg
│   │   ├── person1_happy_2.jpg
│   │   ├── person2_happy_1.jpg
│   │   └── ...
│   ├── sad/
│   │   ├── person1_sad_1.jpg
│   │   └── ...
│   └── neutral/
│       └── ...
└── test/
    ├── happy/
    ├── sad/
    └── neutral/
```

**Tips**:
- Use 70-80% of images for training, 20-30% for testing
- Take at least 20-30 images per person per emotion
- Vary lighting, angles, and backgrounds for better generalization

### 3. Extract Faces (Optional but Recommended)

If your images contain more than just faces, extract faces first:

```python
from utils.preprocessing import extract_faces_from_images

extract_faces_from_images('data/raw_images', 'data/train')
```

## Usage

### Training

Train the model using the training script:

```bash
# Basic training
python train.py

# With custom parameters
python train.py --train_dir data/train \
                --test_dir data/test \
                --model_type basic \
                --epochs 50 \
                --batch_size 32 \
                --learning_rate 0.001
```

**Parameters**:
- `--train_dir`: Directory containing training images
- `--test_dir`: Directory containing test images
- `--model_type`: Model architecture (`basic` or `simple`)
- `--epochs`: Number of training epochs
- `--batch_size`: Batch size for training
- `--learning_rate`: Learning rate for optimizer
- `--image_size`: Input image size (default: 48)

### Prediction

#### Single Image Prediction

```bash
python predict.py --model models/emotion_recognition_model.h5 \
                  --image path/to/image.jpg
```

#### Batch Prediction

Predict emotions for all images in a directory:

```bash
python predict.py --model models/emotion_recognition_model.h5 \
                  --batch_dir path/to/images/
```

#### Real-time Webcam Prediction

```bash
python predict.py --model models/emotion_recognition_model.h5 \
                  --webcam
```

Press 'q' to quit webcam mode.

### Using the Jupyter Notebook

Open the interactive notebook for a guided walkthrough:

```bash
jupyter notebook notebooks/emotion_recognition_demo.ipynb
```

The notebook includes:
- Detailed explanations of CNN concepts
- Step-by-step code with visualizations
- Understanding model architecture
- Training and evaluation examples

## Model Architectures

### Basic CNN (For Beginners)

A simple 3-layer CNN architecture:
- 3 Convolutional layers (32, 64, 128 filters)
- MaxPooling after each conv layer
- 2 Fully connected layers
- Dropout for regularization

### Simple CNN (Advanced)

A deeper architecture with:
- 6 Convolutional layers
- Batch normalization
- Multiple dropout layers
- Better generalization capabilities

## Learning Objectives

This project helps you learn:

1. **CNN Fundamentals**:
   - Convolutional layers and feature extraction
   - Pooling layers and dimensionality reduction
   - Fully connected layers for classification

2. **Deep Learning Workflow**:
   - Data preprocessing and augmentation
   - Model building and compilation
   - Training with validation
   - Model evaluation and prediction

3. **Practical Skills**:
   - Working with image data
   - Using TensorFlow/Keras
   - Face detection with OpenCV
   - Real-time inference

## Key Concepts Explained

### Convolutional Layers
- Extract features from images
- Learn patterns like edges, textures, and facial features
- Use filters that slide over the image

### Pooling Layers
- Reduce spatial dimensions
- Keep important features
- Make model robust to variations

### Data Augmentation
- Creates variations of training images
- Helps model generalize better
- Prevents overfitting

### Dropout
- Randomly deactivates neurons during training
- Prevents overfitting
- Makes model more robust

## Tips for Better Results

1. **More Data**: Collect more images (50+ per person per emotion)
2. **Consistent Lighting**: Take photos in similar lighting conditions
3. **Data Augmentation**: Enabled by default in training
4. **Multiple Angles**: Capture faces from different angles
5. **Experiment**: Try different model architectures and hyperparameters
6. **Monitor Training**: Watch for overfitting (training acc >> validation acc)

## Troubleshooting

### No face detected
- Ensure face is clearly visible in the image
- Check lighting conditions
- Try adjusting face detector parameters

### Low accuracy
- Collect more training data
- Increase number of epochs
- Try data augmentation
- Experiment with learning rate

### Model overfitting
- Reduce model complexity
- Increase dropout rates
- Use more data augmentation
- Collect more training data

## Requirements

- Python 3.7+
- TensorFlow 2.x
- OpenCV
- NumPy
- Matplotlib
- Scikit-learn

See `requirements.txt` for complete list.

## Examples

### Training Output
```
FACIAL EMOTION RECOGNITION - TRAINING
============================================================

Preparing data generators...
Number of classes: 3
Class names: ['happy', 'neutral', 'sad']
Number of training samples: 180

Creating simple CNN model...
Training...
```

### Prediction Output
```
FACIAL EMOTION RECOGNITION - PREDICTION
============================================================

Image: test_image.jpg
Predicted Emotion: HAPPY
Confidence: 95.32%

All Predictions:
  happy       : 95.32%
  neutral     :  3.21%
  sad         :  1.47%
```

## Future Enhancements

- Add more emotion classes
- Implement transfer learning
- Deploy as web application
- Add video emotion tracking
- Multi-face detection and tracking

## License

This project is for educational purposes.

## Authors

Mini-assessment project for Deep Learning course

## Acknowledgments

- TensorFlow/Keras documentation
- OpenCV for face detection
- Deep learning community tutorials
