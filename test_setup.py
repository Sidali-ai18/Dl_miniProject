"""
Test script to verify the setup and imports work correctly.
"""

import sys
import os

def test_imports():
    """Test if all required packages can be imported."""
    print("\n" + "="*60)
    print("Testing Package Imports")
    print("="*60 + "\n")
    
    packages = [
        ('numpy', 'NumPy'),
        ('cv2', 'OpenCV'),
        ('tensorflow', 'TensorFlow'),
        ('keras', 'Keras'),
        ('matplotlib', 'Matplotlib'),
        ('sklearn', 'Scikit-learn'),
        ('pandas', 'Pandas'),
        ('PIL', 'Pillow'),
    ]
    
    failed = []
    for package, name in packages:
        try:
            __import__(package)
            print(f"✓ {name:15s} - OK")
        except ImportError as e:
            print(f"✗ {name:15s} - FAILED")
            failed.append(name)
    
    if failed:
        print(f"\n⚠ Failed to import: {', '.join(failed)}")
        print("Please run: pip install -r requirements.txt")
        return False
    else:
        print("\n✓ All packages imported successfully!")
        return True


def test_project_structure():
    """Test if project structure is set up correctly."""
    print("\n" + "="*60)
    print("Testing Project Structure")
    print("="*60 + "\n")
    
    required_dirs = [
        'data',
        'data/train',
        'data/test',
        'models',
        'utils',
        'notebooks'
    ]
    
    required_files = [
        'train.py',
        'predict.py',
        'prepare_data.py',
        'utils/preprocessing.py',
        'utils/__init__.py',
        'models/cnn_model.py',
        'models/__init__.py',
        'requirements.txt',
        'README.md',
        'QUICKSTART.md'
    ]
    
    missing_dirs = []
    for dir_path in required_dirs:
        if os.path.exists(dir_path):
            print(f"✓ Directory: {dir_path}")
        else:
            print(f"✗ Missing directory: {dir_path}")
            missing_dirs.append(dir_path)
    
    print()
    
    missing_files = []
    for file_path in required_files:
        if os.path.exists(file_path):
            print(f"✓ File: {file_path}")
        else:
            print(f"✗ Missing file: {file_path}")
            missing_files.append(file_path)
    
    if missing_dirs or missing_files:
        print("\n⚠ Some files or directories are missing!")
        return False
    else:
        print("\n✓ All required files and directories exist!")
        return True


def test_modules():
    """Test if project modules can be imported."""
    print("\n" + "="*60)
    print("Testing Project Modules")
    print("="*60 + "\n")
    
    try:
        from models.cnn_model import create_basic_cnn, create_simple_cnn
        print("✓ models.cnn_model imported successfully")
    except Exception as e:
        print(f"✗ Failed to import models.cnn_model: {e}")
        return False
    
    try:
        from utils.preprocessing import FaceDetector, preprocess_image
        print("✓ utils.preprocessing imported successfully")
    except Exception as e:
        print(f"✗ Failed to import utils.preprocessing: {e}")
        return False
    
    print("\n✓ All project modules imported successfully!")
    return True


def test_face_detector():
    """Test if face detector can be initialized."""
    print("\n" + "="*60)
    print("Testing Face Detector")
    print("="*60 + "\n")
    
    try:
        from utils.preprocessing import FaceDetector
        detector = FaceDetector()
        print("✓ Face detector initialized successfully")
        print(f"✓ Haar cascade loaded: {detector.face_cascade is not None}")
        return True
    except Exception as e:
        print(f"✗ Failed to initialize face detector: {e}")
        return False


def test_model_creation():
    """Test if models can be created."""
    print("\n" + "="*60)
    print("Testing Model Creation")
    print("="*60 + "\n")
    
    try:
        from models.cnn_model import create_basic_cnn, compile_model
        
        # Create basic model
        model = create_basic_cnn(input_shape=(48, 48, 1), num_classes=3)
        model = compile_model(model)
        
        print("✓ Basic CNN model created successfully")
        print(f"✓ Model has {model.count_params():,} parameters")
        
        return True
    except Exception as e:
        print(f"✗ Failed to create model: {e}")
        return False


def main():
    """Run all tests."""
    print("\n" + "="*60)
    print("FACIAL EMOTION RECOGNITION - SETUP TEST")
    print("="*60)
    
    results = []
    
    # Run tests
    results.append(("Package Imports", test_imports()))
    results.append(("Project Structure", test_project_structure()))
    results.append(("Project Modules", test_modules()))
    results.append(("Face Detector", test_face_detector()))
    results.append(("Model Creation", test_model_creation()))
    
    # Summary
    print("\n" + "="*60)
    print("TEST SUMMARY")
    print("="*60 + "\n")
    
    all_passed = True
    for test_name, passed in results:
        status = "✓ PASSED" if passed else "✗ FAILED"
        print(f"{test_name:20s}: {status}")
        if not passed:
            all_passed = False
    
    print("\n" + "="*60)
    if all_passed:
        print("✓ ALL TESTS PASSED!")
        print("\nYou're ready to start!")
        print("Next steps:")
        print("1. Add your images to data/train/ and data/test/")
        print("2. Run: python train.py")
        print("3. Run: python predict.py --image path/to/image.jpg")
    else:
        print("✗ SOME TESTS FAILED")
        print("\nPlease fix the issues above before proceeding.")
    print("="*60 + "\n")
    
    return 0 if all_passed else 1


if __name__ == '__main__':
    sys.exit(main())
