# Setup for Happy and Neutral Emotion Recognition

This guide is specifically for training a model to recognize **Happy** and **Neutral** emotions.

## Quick Setup

The folder structure has been created for you:

```
data/
├── train/
│   ├── happy/      <- Add your happy emotion photos here
│   └── neutral/    <- Add your neutral emotion photos here
└── test/
    ├── happy/      <- Add test happy photos here
    └── neutral/    <- Add test neutral photos here
```

## Data Collection Tips

### For Happy Emotion:
- Smile naturally (showing teeth or not)
- Take photos with different expressions of happiness
- Vary the intensity (slight smile to big grin)
- Try different angles and lighting
- Aim for **30-50 images** from each person

### For Neutral Emotion:
- Relaxed face, no strong emotion
- Looking straight at camera
- Natural, calm expression
- Mouth closed, not smiling
- Aim for **30-50 images** from each person

## Recommended Data Split

- **Training**: 70-80% of your images
  - Example: 35-40 happy images in `data/train/happy/`
  - Example: 35-40 neutral images in `data/train/neutral/`

- **Testing**: 20-30% of your images
  - Example: 10-15 happy images in `data/test/happy/`
  - Example: 10-15 neutral images in `data/test/neutral/`

## Training Commands

### Basic Training (Recommended for 2 emotions)
```bash
python train.py --epochs 30 --model_type basic --batch_size 16
```

### Advanced Training
```bash
python train.py --epochs 50 --model_type simple --batch_size 16 --learning_rate 0.001
```

## Testing Your Model

### Test on a single image
```bash
# Test a happy image
python predict.py --image data/test/happy/photo1.jpg --labels happy neutral

# Test a neutral image
python predict.py --image data/test/neutral/photo1.jpg --labels happy neutral
```

### Batch test all images
```bash
python predict.py --batch_dir data/test/ --labels happy neutral
```

### Real-time webcam test
```bash
python predict.py --webcam --labels happy neutral
```

## Expected Results

With **2 emotions** (Happy and Neutral), you can expect:
- **30-40 images/emotion**: 75-85% accuracy
- **50-60 images/emotion**: 85-92% accuracy
- **80+ images/emotion**: 90-95% accuracy

Binary classification (2 classes) is easier than multi-class, so you should get good results!

## Tips for Best Results

1. **Clear Expressions**: Make sure happy photos show clear smiles, and neutral photos show relaxed faces
2. **Consistent Lighting**: Try to keep similar lighting across all photos
3. **Vary Angles**: Take some photos from slightly different angles
4. **Multiple Sessions**: Take photos on different days with different lighting/backgrounds
5. **Balance Data**: Try to have similar number of images for both emotions

## Example Workflow

```bash
# 1. Check your data
python prepare_data.py count

# Should show something like:
#   happy       : 40 images
#   neutral     : 40 images

# 2. Train the model
python train.py --epochs 30 --model_type basic

# 3. Test predictions
python predict.py --image data/test/happy/test1.jpg --labels happy neutral

# 4. Try webcam
python predict.py --webcam --labels happy neutral
```

## Troubleshooting

**If accuracy is low (<70%)**:
- Collect more images (aim for 50+ per emotion)
- Ensure expressions are clear and distinct
- Check that lighting is consistent
- Make sure faces are clearly visible
- Train for more epochs (try --epochs 50)

**If model overfits (train accuracy >> test accuracy)**:
- Collect more diverse training data
- Use data augmentation (enabled by default)
- Try the basic model instead of simple

## Next Steps After Training

Once you have a working model:
1. Test it on new photos of you and your friend
2. Try the real-time webcam mode
3. Experiment with different facial expressions
4. Add more emotion classes if desired (sad, surprised, etc.)

Good luck with your mini-assessment project! 🎓
