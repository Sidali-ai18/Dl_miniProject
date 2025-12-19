"""
Utilities for preprocessing facial images for emotion recognition.
Includes face detection, image preprocessing, and data augmentation.
"""

import cv2
import numpy as np
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import os


class FaceDetector:
    """Face detection using OpenCV's Haar Cascade classifier."""
    
    def __init__(self):
        # Load the pre-trained Haar Cascade for face detection
        cascade_path = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
        self.face_cascade = cv2.CascadeClassifier(cascade_path)
    
    def detect_face(self, image):
        """
        Detect face in an image and return the cropped face region.
        
        Args:
            image: Input image (BGR format)
            
        Returns:
            Cropped face image or None if no face detected
        """
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        faces = self.face_cascade.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(30, 30)
        )
        
        if len(faces) > 0:
            # Get the largest face
            x, y, w, h = max(faces, key=lambda rect: rect[2] * rect[3])
            face = image[y:y+h, x:x+w]
            return face
        return None


def preprocess_image(image, target_size=(48, 48)):
    """
    Preprocess image for model input.
    
    Args:
        image: Input image
        target_size: Target size for resizing (width, height)
        
    Returns:
        Preprocessed image array
    """
    # Resize image
    image = cv2.resize(image, target_size)
    
    # Convert to grayscale if not already
    if len(image.shape) == 3:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Normalize pixel values to [0, 1]
    image = image.astype('float32') / 255.0
    
    # Expand dimensions for model input
    image = np.expand_dims(image, axis=-1)
    
    return image


def load_and_preprocess_image(image_path, target_size=(48, 48), detect_face=True, detector=None):
    """
    Load image from path and preprocess it.
    
    Args:
        image_path: Path to image file
        target_size: Target size for resizing
        detect_face: Whether to detect and crop face
        detector: Optional FaceDetector instance to reuse (more efficient for batch processing)
        
    Returns:
        Preprocessed image array or None if face not detected
    """
    # Load image
    image = cv2.imread(image_path)
    if image is None:
        print(f"Error loading image: {image_path}")
        return None
    
    # Detect face if requested
    if detect_face:
        if detector is None:
            detector = FaceDetector()
        face = detector.detect_face(image)
        if face is None:
            print(f"No face detected in: {image_path}")
            return None
        image = face
    
    # Preprocess
    return preprocess_image(image, target_size)


def create_data_generator(augment=True):
    """
    Create ImageDataGenerator for training data augmentation.
    
    Args:
        augment: Whether to apply data augmentation
        
    Returns:
        ImageDataGenerator instance
    """
    if augment:
        return ImageDataGenerator(
            rotation_range=20,
            width_shift_range=0.2,
            height_shift_range=0.2,
            horizontal_flip=True,
            zoom_range=0.2,
            shear_range=0.2,
            fill_mode='nearest',
            rescale=1./255
        )
    else:
        return ImageDataGenerator(rescale=1./255)


def prepare_dataset_from_directory(data_dir, target_size=(48, 48), batch_size=32, augment=True):
    """
    Prepare dataset from directory structure.
    Expected structure: data_dir/emotion_name/image.jpg
    
    Args:
        data_dir: Path to data directory
        target_size: Target image size
        batch_size: Batch size for training
        augment: Whether to apply augmentation
        
    Returns:
        Data generator
    """
    datagen = create_data_generator(augment=augment)
    
    generator = datagen.flow_from_directory(
        data_dir,
        target_size=target_size,
        color_mode='grayscale',
        batch_size=batch_size,
        class_mode='categorical',
        shuffle=True
    )
    
    return generator


def extract_faces_from_images(input_dir, output_dir):
    """
    Extract faces from all images in input directory and save to output directory.
    Maintains the same subdirectory structure.
    
    Args:
        input_dir: Directory containing original images
        output_dir: Directory to save extracted faces
    """
    detector = FaceDetector()
    
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # Walk through directory structure
    for root, dirs, files in os.walk(input_dir):
        for filename in files:
            if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
                input_path = os.path.join(root, filename)
                
                # Maintain directory structure
                rel_path = os.path.relpath(root, input_dir)
                output_subdir = os.path.join(output_dir, rel_path)
                os.makedirs(output_subdir, exist_ok=True)
                
                # Read and process image
                image = cv2.imread(input_path)
                if image is not None:
                    face = detector.detect_face(image)
                    if face is not None:
                        output_path = os.path.join(output_subdir, filename)
                        cv2.imwrite(output_path, face)
                        print(f"Saved face: {output_path}")
                    else:
                        print(f"No face detected: {input_path}")
