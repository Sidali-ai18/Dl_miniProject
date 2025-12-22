"""
Prediction script for facial emotion recognition.
Supports single image, webcam, and batch prediction.
"""

import os
import sys
import argparse
import cv2
import numpy as np
from tensorflow.keras.models import load_model

# Ensure project root is in path for imports
_project_root = os.path.dirname(os.path.abspath(__file__))
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

from utils.preprocessing import FaceDetector, preprocess_image


class EmotionPredictor:
    """Emotion predictor class for making predictions on images."""
    
    def __init__(self, model_path, emotion_labels=None):
        """
        Initialize predictor.
        
        Args:
            model_path: Path to trained model
            emotion_labels: List of emotion labels (in order)
        """
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model not found: {model_path}")
        
        self.model = load_model(model_path)
        self.face_detector = FaceDetector()
        
        # Default emotion labels (can be customized)
        if emotion_labels is None:
            self.emotion_labels = ['angry', 'disgust', 'fear', 'happy', 
                                   'neutral', 'sad', 'surprise']
        else:
            self.emotion_labels = emotion_labels
    
    def predict_emotion(self, image, detect_face=True):
        """
        Predict emotion from image.
        
        Args:
            image: Input image (BGR format)
            detect_face: Whether to detect face first
            
        Returns:
            Tuple of (predicted_emotion, confidence, all_probabilities)
        """
        # Detect face if requested
        if detect_face:
            face = self.face_detector.detect_face(image)
            if face is None:
                return None, 0, None
            image = face
        
        # Preprocess image
        processed = preprocess_image(image)
        processed = np.expand_dims(processed, axis=0)
        
        # Make prediction
        predictions = self.model.predict(processed, verbose=0)[0]
        
        # Get predicted class
        predicted_idx = np.argmax(predictions)
        predicted_emotion = self.emotion_labels[predicted_idx]
        confidence = predictions[predicted_idx]
        
        return predicted_emotion, confidence, predictions
    
    def predict_from_file(self, image_path):
        """
        Predict emotion from image file.
        
        Args:
            image_path: Path to image file
            
        Returns:
            Tuple of (predicted_emotion, confidence, all_probabilities)
        """
        image = cv2.imread(image_path)
        if image is None:
            print(f"Error: Could not load image: {image_path}")
            return None, 0, None
        
        return self.predict_emotion(image)
    
    def predict_from_webcam(self, camera_id=0):
        """
        Real-time emotion prediction from webcam.
        
        Args:
            camera_id: Camera device ID
        """
        cap = cv2.VideoCapture(camera_id)
        
        if not cap.isOpened():
            print("Error: Could not open webcam")
            return
        
        print("\nStarting webcam emotion recognition...")
        print("Press 'q' to quit\n")
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            # Make prediction
            emotion, confidence, probabilities = self.predict_emotion(frame.copy())
            
            # Draw results on frame
            if emotion is not None:
                # Detect face for visualization
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                faces = self.face_detector.face_cascade.detectMultiScale(
                    gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30)
                )
                
                # Draw rectangle around face
                for (x, y, w, h) in faces:
                    cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
                    
                    # Draw emotion label
                    label = f"{emotion}: {confidence*100:.1f}%"
                    cv2.putText(frame, label, (x, y-10),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
            
            # Display frame
            cv2.imshow('Emotion Recognition', frame)
            
            # Check for quit
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        
        cap.release()
        cv2.destroyAllWindows()


def predict_single_image(model_path, image_path, emotion_labels=None):
    """
    Predict emotion from a single image.
    
    Args:
        model_path: Path to trained model
        image_path: Path to image file
        emotion_labels: List of emotion labels
    """
    print("\n" + "="*60)
    print("FACIAL EMOTION RECOGNITION - PREDICTION")
    print("="*60)
    
    predictor = EmotionPredictor(model_path, emotion_labels)
    
    emotion, confidence, probabilities = predictor.predict_from_file(image_path)
    
    if emotion is None:
        print(f"\nError: No face detected in image: {image_path}")
        return
    
    print(f"\nImage: {image_path}")
    print(f"Predicted Emotion: {emotion.upper()}")
    print(f"Confidence: {confidence*100:.2f}%")
    
    print("\nAll Predictions:")
    for label, prob in zip(predictor.emotion_labels, probabilities):
        print(f"  {label:12s}: {prob*100:5.2f}%")
    
    print("="*60 + "\n")


def predict_batch(model_path, image_dir, emotion_labels=None):
    """
    Predict emotions for all images in a directory.
    
    Args:
        model_path: Path to trained model
        image_dir: Directory containing images
        emotion_labels: List of emotion labels
    """
    print("\n" + "="*60)
    print("FACIAL EMOTION RECOGNITION - BATCH PREDICTION")
    print("="*60)
    
    predictor = EmotionPredictor(model_path, emotion_labels)
    
    # Get all image files
    image_extensions = ('.jpg', '.jpeg', '.png', '.bmp')
    image_files = [f for f in os.listdir(image_dir) 
                   if f.lower().endswith(image_extensions)]
    
    if not image_files:
        print(f"\nNo images found in: {image_dir}")
        return
    
    print(f"\nProcessing {len(image_files)} images...\n")
    
    results = []
    for image_file in image_files:
        image_path = os.path.join(image_dir, image_file)
        emotion, confidence, _ = predictor.predict_from_file(image_path)
        
        if emotion is not None:
            results.append((image_file, emotion, confidence))
            print(f"{image_file:30s} -> {emotion:10s} ({confidence*100:.1f}%)")
        else:
            print(f"{image_file:30s} -> No face detected")
    
    print("\n" + "="*60 + "\n")


def main():
    parser = argparse.ArgumentParser(description='Predict emotions from images')
    parser.add_argument('--model', type=str, default='models/emotion_recognition_model.h5',
                        help='Path to trained model')
    parser.add_argument('--image', type=str,
                        help='Path to single image file')
    parser.add_argument('--batch_dir', type=str,
                        help='Directory containing images for batch prediction')
    parser.add_argument('--webcam', action='store_true',
                        help='Use webcam for real-time prediction')
    parser.add_argument('--camera_id', type=int, default=0,
                        help='Camera device ID for webcam mode')
    parser.add_argument('--labels', type=str, nargs='+',
                        help='Emotion labels (space-separated)')
    
    args = parser.parse_args()
    
    # Check if model exists
    if not os.path.exists(args.model):
        print(f"\nError: Model not found: {args.model}")
        print("Please train a model first using train.py")
        return
    
    # Run appropriate prediction mode
    if args.webcam:
        predictor = EmotionPredictor(args.model, args.labels)
        predictor.predict_from_webcam(args.camera_id)
    elif args.image:
        predict_single_image(args.model, args.image, args.labels)
    elif args.batch_dir:
        predict_batch(args.model, args.batch_dir, args.labels)
    else:
        print("\nError: Please specify --image, --batch_dir, or --webcam")
        parser.print_help()


if __name__ == '__main__':
    main()
