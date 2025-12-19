# Quick Start Guide

Get started with facial emotion recognition in minutes!

## Step 1: Install Dependencies

```bash
pip install -r requirements.txt
```

## Step 2: Prepare Your Data

### Option A: Create Directory Structure

```bash
python prepare_data.py create
```

This creates the directory structure. Then manually add your images:
- Place training images in `data/train/<emotion>/`
- Place test images in `data/test/<emotion>/`

### Option B: Split Existing Data

If you have all images in one directory:

```bash
python prepare_data.py split --source_dir path/to/your/images
```

This automatically splits images into train (80%) and test (20%) sets.

## Step 3: Take Photos

Take photos of you and your friend showing different emotions:

**Recommended:**
- At least 30 images per person per emotion
- Clear, well-lit faces
- Different angles and backgrounds
- Various lighting conditions

**Example emotions:**
- happy
- sad
- neutral

You can start with just 2-3 emotions!

## Step 4: Check Your Data

```bash
python prepare_data.py count
```

This shows how many images you have in each category.

## Step 5: Train Your Model

### Basic Training
```bash
python train.py
```

### Custom Training
```bash
python train.py --epochs 30 --model_type basic --batch_size 16
```

Training will:
- Load and augment your images
- Train the CNN model
- Save the best model to `models/emotion_recognition_model.h5`
- Generate training plots in `results/`

## Step 6: Test Your Model

### Predict on Single Image
```bash
python predict.py --image path/to/test/image.jpg
```

### Batch Prediction
```bash
python predict.py --batch_dir data/test/happy
```

### Real-time Webcam
```bash
python predict.py --webcam
```

Press 'q' to quit webcam mode.

## Step 7: Explore the Notebook

```bash
jupyter notebook notebooks/emotion_recognition_demo.ipynb
```

The notebook includes:
- Detailed CNN explanations
- Code with visualizations
- Interactive examples
- Learning resources

## Troubleshooting

### "No training data found"
- Make sure images are in `data/train/<emotion>/` subdirectories
- Each emotion should have its own folder
- Check that image files are .jpg, .jpeg, or .png

### "No face detected"
- Ensure faces are clearly visible
- Check image quality and lighting
- Try adjusting the image

### Model accuracy is low
- Collect more training images (aim for 50+ per emotion)
- Train for more epochs
- Try the 'simple' model type instead of 'basic'

## Example Directory After Setup

```
Dl_miniProject/
├── data/
│   ├── train/
│   │   ├── happy/
│   │   │   ├── person1_happy_1.jpg
│   │   │   ├── person1_happy_2.jpg
│   │   │   ├── person2_happy_1.jpg
│   │   │   └── ... (20+ more images)
│   │   ├── sad/
│   │   │   └── ... (20+ images)
│   │   └── neutral/
│   │       └── ... (20+ images)
│   └── test/
│       ├── happy/
│       │   └── ... (5-10 images)
│       ├── sad/
│       └── neutral/
└── models/
    └── emotion_recognition_model.h5  (created after training)
```

## Tips for Best Results

1. **Quality over Quantity**: Clear, well-lit photos are better than many poor-quality ones
2. **Variation**: Capture different angles, lighting, and backgrounds
3. **Consistency**: Use similar image quality across all emotions
4. **Balance**: Try to have similar number of images for each emotion
5. **Testing**: Keep some images aside for testing that the model never sees during training

## Next Steps

Once your basic model works:
- Add more emotions
- Collect more data
- Experiment with hyperparameters
- Try the advanced model architecture
- Deploy your model

## Need Help?

Check the main [README.md](README.md) for detailed documentation.

Happy learning! 🎓
