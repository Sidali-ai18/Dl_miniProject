# Quick Reference - Happy & Neutral Emotion Recognition

## 📁 Folder Structure
```
data/
├── train/
│   ├── happy/      ← 30-50 happy photos here
│   └── neutral/    ← 30-50 neutral photos here
└── test/
    ├── happy/      ← 10-15 happy photos here
    └── neutral/    ← 10-15 neutral photos here
```

## 🚀 Commands

### Training
```bash
# Optimized for Happy & Neutral (recommended)
python train_happy_neutral.py

# Or use general script with custom labels
python train.py --epochs 30 --model_type basic --batch_size 16
```

### Testing
```bash
# Single image
python predict.py --image photo.jpg --labels happy neutral

# Webcam (press 'q' to quit)
python predict.py --webcam --labels happy neutral

# Batch test
python predict.py --batch_dir data/test/ --labels happy neutral
```

### Data Management
```bash
# Count your images
python prepare_data.py count

# Split data (if all images in one folder)
python prepare_data.py split --source_dir raw_photos/ --split_ratio 0.8
```

## 📊 Expected Accuracy
- 30-40 images/emotion: **75-85%**
- 50-60 images/emotion: **85-92%**
- 80+ images/emotion: **90-95%**

## 💡 Tips
✅ **Happy**: Clear smile, natural expression, vary intensity
✅ **Neutral**: Relaxed face, no emotion, calm expression
✅ **Lighting**: Keep consistent across photos
✅ **Variety**: Different angles and backgrounds
✅ **Balance**: Same number of photos for each emotion

## 🔧 Troubleshooting
- **Low accuracy?** → Collect more images (50+ per emotion)
- **Overfitting?** → Use more data augmentation (already enabled)
- **No face detected?** → Ensure face is clearly visible and well-lit

## 📖 Full Documentation
- `HAPPY_NEUTRAL_SETUP.md` - Complete guide
- `README.md` - Full project documentation
- `GETTING_STARTED.md` - General quick start
