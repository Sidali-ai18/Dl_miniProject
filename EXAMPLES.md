# Usage Examples

This file contains example commands and code snippets for using the facial emotion recognition system.

## Installation

```bash
# Clone the repository
git clone https://github.com/Sidali-ai18/Dl_miniProject.git
cd Dl_miniProject

# Install dependencies
pip install -r requirements.txt
```

## Data Preparation Examples

### Create Directory Structure

```bash
# Create organized folders for your emotions
python prepare_data.py create

# Create with custom emotions
python prepare_data.py create --emotions happy sad angry neutral
```

### Split Existing Data

```bash
# Split images from source directory (80% train, 20% test)
python prepare_data.py split --source_dir my_photos/ --train_dir data/train --test_dir data/test

# Custom split ratio (70% train, 30% test)
python prepare_data.py split --source_dir my_photos/ --split_ratio 0.7
```

### Check Data Count

```bash
# Count images in train and test directories
python prepare_data.py count
```

## Training Examples

### Basic Training

```bash
# Train with default settings
python train.py
```

### Custom Training Options

```bash
# Train with basic model for 30 epochs
python train.py --model_type basic --epochs 30

# Train with custom batch size and learning rate
python train.py --batch_size 16 --learning_rate 0.0001

# Train with all custom parameters
python train.py \
    --train_dir data/train \
    --test_dir data/test \
    --model_type simple \
    --epochs 50 \
    --batch_size 32 \
    --learning_rate 0.001 \
    --image_size 48
```

### Training Output Example

```
============================================================
FACIAL EMOTION RECOGNITION - TRAINING
============================================================

Preparing data generators...
Found 180 images belonging to 3 classes.
Found 45 images belonging to 3 classes.

Number of classes: 3
Class names: ['happy', 'neutral', 'sad']
Number of training samples: 180
Number of test samples: 45

Creating simple CNN model...

==================================================
MODEL ARCHITECTURE
==================================================
Model: "sequential"
_________________________________________________________________
Layer (type)                Output Shape              Param #   
=================================================================
conv2d (Conv2D)             (None, 48, 48, 32)        320       
batch_normalization         (None, 48, 48, 32)        128       
activation (Activation)     (None, 48, 48, 32)        0         
...
=================================================================
Total params: 234,563
Trainable params: 233,475
Non-trainable params: 1,088
==================================================

Starting training...
Epochs: 50
Batch size: 32
Learning rate: 0.001

Epoch 1/50
6/6 [==============================] - 2s 333ms/step - loss: 1.0986 - accuracy: 0.3389 - val_loss: 1.0912 - val_accuracy: 0.3556
Epoch 2/50
6/6 [==============================] - 1s 167ms/step - loss: 1.0523 - accuracy: 0.4611 - val_loss: 1.0234 - val_accuracy: 0.5333
...
Epoch 50/50
6/6 [==============================] - 1s 167ms/step - loss: 0.1234 - accuracy: 0.9556 - val_loss: 0.2456 - val_accuracy: 0.9111

============================================================
TRAINING COMPLETED
============================================================

Final Test Accuracy: 91.11%
Final Test Loss: 0.2456
Final Train Accuracy: 95.56%
Final Train Loss: 0.1234

Model saved to: models/emotion_recognition_model.h5
Training history plot saved to: results/training_history.png
============================================================
```

## Prediction Examples

### Single Image Prediction

```bash
# Predict emotion in one image
python predict.py --image data/test/happy/photo1.jpg

# Use custom model
python predict.py --model models/my_model.h5 --image test_photo.jpg

# Use custom emotion labels
python predict.py --image photo.jpg --labels happy sad angry neutral
```

### Prediction Output Example

```
============================================================
FACIAL EMOTION RECOGNITION - PREDICTION
============================================================

Image: data/test/happy/photo1.jpg
Predicted Emotion: HAPPY
Confidence: 95.32%

All Predictions:
  happy       : 95.32%
  neutral     :  3.21%
  sad         :  1.47%
============================================================
```

### Batch Prediction

```bash
# Predict all images in a directory
python predict.py --batch_dir data/test/happy/

# With custom model
python predict.py --model models/my_model.h5 --batch_dir test_images/
```

### Batch Prediction Output Example

```
============================================================
FACIAL EMOTION RECOGNITION - BATCH PREDICTION
============================================================

Processing 15 images...

photo1.jpg                     -> happy      (95.3%)
photo2.jpg                     -> happy      (87.6%)
photo3.jpg                     -> neutral    (78.4%)
photo4.jpg                     -> No face detected
photo5.jpg                     -> sad        (92.1%)
...

============================================================
```

### Real-time Webcam Prediction

```bash
# Use default webcam (camera 0)
python predict.py --webcam

# Use specific camera
python predict.py --webcam --camera_id 1

# With custom model and labels
python predict.py --webcam --model models/my_model.h5 --labels happy sad angry
```

## Python Code Examples

### Using the Face Detector

```python
from utils.preprocessing import FaceDetector
import cv2

# Initialize detector
detector = FaceDetector()

# Load image
image = cv2.imread('photo.jpg')

# Detect face
face = detector.detect_face(image)

if face is not None:
    cv2.imwrite('detected_face.jpg', face)
    print("Face detected and saved!")
else:
    print("No face detected")
```

### Using the Preprocessor

```python
from utils.preprocessing import preprocess_image, load_and_preprocess_image
import cv2

# Preprocess an already loaded image
image = cv2.imread('photo.jpg')
processed = preprocess_image(image, target_size=(48, 48))

# Load and preprocess in one step
processed = load_and_preprocess_image('photo.jpg', target_size=(48, 48), detect_face=True)
```

### Creating and Training a Model

```python
from models.cnn_model import create_basic_cnn, compile_model, get_callbacks
from utils.preprocessing import prepare_dataset_from_directory

# Create model
model = create_basic_cnn(input_shape=(48, 48, 1), num_classes=3)
model = compile_model(model, learning_rate=0.001)

# Prepare data
train_gen = prepare_dataset_from_directory('data/train', batch_size=32, augment=True)
test_gen = prepare_dataset_from_directory('data/test', batch_size=32, augment=False)

# Train model
callbacks = get_callbacks(model_save_path='models/my_model.h5')
history = model.fit(
    train_gen,
    epochs=30,
    validation_data=test_gen,
    callbacks=callbacks
)
```

### Making Predictions

```python
from predict import EmotionPredictor
import cv2

# Initialize predictor
predictor = EmotionPredictor(
    model_path='models/emotion_recognition_model.h5',
    emotion_labels=['happy', 'sad', 'neutral']
)

# Predict from file
emotion, confidence, probs = predictor.predict_from_file('test_image.jpg')
print(f"Emotion: {emotion}, Confidence: {confidence*100:.1f}%")

# Predict from loaded image
image = cv2.imread('test_image.jpg')
emotion, confidence, probs = predictor.predict_emotion(image)

# Real-time webcam prediction
predictor.predict_from_webcam(camera_id=0)
```

### Extract Faces from Images

```python
from utils.preprocessing import extract_faces_from_images

# Extract faces from all images in a directory
extract_faces_from_images(
    input_dir='raw_photos/',
    output_dir='faces_only/'
)
```

## Jupyter Notebook Examples

```python
# In Jupyter Notebook
import matplotlib.pyplot as plt
from predict import EmotionPredictor
import cv2

# Load predictor
predictor = EmotionPredictor('models/emotion_recognition_model.h5')

# Predict and visualize
image = cv2.imread('test_image.jpg')
emotion, confidence, probs = predictor.predict_emotion(image)

# Display image with prediction
plt.figure(figsize=(10, 5))
plt.subplot(1, 2, 1)
plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
plt.title(f'Predicted: {emotion} ({confidence*100:.1f}%)')
plt.axis('off')

# Display probabilities
plt.subplot(1, 2, 2)
plt.bar(predictor.emotion_labels, probs)
plt.title('Emotion Probabilities')
plt.ylabel('Probability')
plt.xlabel('Emotion')
plt.ylim([0, 1])
plt.show()
```

## Advanced Usage

### Custom Data Augmentation

```python
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# Create custom augmentation
datagen = ImageDataGenerator(
    rotation_range=30,
    width_shift_range=0.3,
    height_shift_range=0.3,
    horizontal_flip=True,
    vertical_flip=False,
    zoom_range=0.3,
    shear_range=0.3,
    fill_mode='nearest',
    rescale=1./255
)

# Use with your data
generator = datagen.flow_from_directory(
    'data/train',
    target_size=(48, 48),
    color_mode='grayscale',
    batch_size=32,
    class_mode='categorical'
)
```

### Custom Model Architecture

```python
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dense, Flatten, Dropout

# Create your own architecture
model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(48, 48, 1)),
    MaxPooling2D((2, 2)),
    
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D((2, 2)),
    
    Flatten(),
    Dense(128, activation='relu'),
    Dropout(0.5),
    Dense(3, activation='softmax')  # 3 emotions
])

model.compile(
    optimizer='adam',
    loss='categorical_crossentropy',
    metrics=['accuracy']
)
```

### Transfer Learning Example

```python
from tensorflow.keras.applications import VGG16
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D

# Load pre-trained VGG16
base_model = VGG16(weights='imagenet', include_top=False, input_shape=(48, 48, 3))

# Freeze base layers
for layer in base_model.layers:
    layer.trainable = False

# Add custom layers
x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dense(256, activation='relu')(x)
predictions = Dense(3, activation='softmax')(x)

# Create model
model = Model(inputs=base_model.input, outputs=predictions)
```

## Tips and Tricks

### 1. Debugging Low Accuracy
```python
# Check if data is loading correctly
train_gen = prepare_dataset_from_directory('data/train')
images, labels = next(train_gen)
print(f"Batch shape: {images.shape}")
print(f"Labels shape: {labels.shape}")
print(f"Classes: {train_gen.class_indices}")

# Visualize augmented images
import matplotlib.pyplot as plt
fig, axes = plt.subplots(2, 4, figsize=(12, 6))
for i, ax in enumerate(axes.ravel()):
    ax.imshow(images[i].squeeze(), cmap='gray')
    ax.axis('off')
plt.show()
```

### 2. Monitoring Training
```python
# Use verbose=1 for detailed progress
history = model.fit(train_gen, epochs=50, validation_data=test_gen, verbose=1)

# Plot training curves
plt.plot(history.history['accuracy'], label='Train')
plt.plot(history.history['val_accuracy'], label='Val')
plt.legend()
plt.show()
```

### 3. Saving and Loading Models
```python
# Save model
model.save('my_model.h5')

# Load model
from tensorflow.keras.models import load_model
model = load_model('my_model.h5')

# Continue training
model.fit(train_gen, epochs=10)
```

## Common Workflows

### Workflow 1: Quick Start (Minimal Data)
```bash
# 1. Create structure
python prepare_data.py create --emotions happy sad neutral

# 2. Add 20+ images per emotion to data/train/ and data/test/

# 3. Train with basic model
python train.py --model_type basic --epochs 30

# 4. Test
python predict.py --image test.jpg
```

### Workflow 2: Full Project (More Data)
```bash
# 1. Collect 50+ images per emotion
# 2. Split data automatically
python prepare_data.py split --source_dir raw_images/

# 3. Verify data
python prepare_data.py count

# 4. Train with advanced model
python train.py --model_type simple --epochs 50 --batch_size 32

# 5. Test on batch
python predict.py --batch_dir data/test/

# 6. Try real-time
python predict.py --webcam
```

### Workflow 3: Iterative Improvement
```bash
# Round 1: Basic training
python train.py --epochs 20 --model_type basic

# Round 2: Collect more data and retrain
python train.py --epochs 30 --model_type simple

# Round 3: Fine-tune hyperparameters
python train.py --epochs 50 --learning_rate 0.0001 --batch_size 16

# Round 4: Try different model
python train.py --epochs 50 --model_type simple
```

## Conclusion

These examples cover the most common use cases. For more details, see:
- README.md - Full documentation
- QUICKSTART.md - Getting started guide
- DOCUMENTATION.md - Detailed explanations
- notebooks/emotion_recognition_demo.ipynb - Interactive tutorial
