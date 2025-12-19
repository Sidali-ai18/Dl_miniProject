# Getting Started in 5 Minutes

This guide will get you up and running with facial emotion recognition in just 5 minutes!

## Prerequisites

- Python 3.7 or higher
- A webcam (optional, for real-time prediction)
- Photos of you and your friend

## Step 1: Install (1 minute)

```bash
# Clone the repository
git clone https://github.com/Sidali-ai18/Dl_miniProject.git
cd Dl_miniProject

# Install dependencies
pip install -r requirements.txt
```

## Step 2: Prepare Data (2 minutes)

### Create folder structure:
```bash
python prepare_data.py create --emotions happy sad neutral
```

### Add your photos:
```
data/
  train/
    happy/     <- Add 20+ happy photos here
    sad/       <- Add 20+ sad photos here
    neutral/   <- Add 20+ neutral photos here
  test/
    happy/     <- Add 5+ happy photos here
    sad/       <- Add 5+ sad photos here
    neutral/   <- Add 5+ neutral photos here
```

**Quick tip**: Start with just 2 emotions if you're short on time!

## Step 3: Train (1 minute setup, ~5-10 minutes training)

```bash
# Start training
python train.py --epochs 30 --model_type basic
```

The model will train automatically and save the best version.

## Step 4: Test (1 minute)

```bash
# Test on a single image
python predict.py --image data/test/happy/photo.jpg

# Try real-time webcam (press 'q' to quit)
python predict.py --webcam
```

## That's It! 🎉

You now have a working emotion recognition system!

## Next Steps

- Read [QUICKSTART.md](QUICKSTART.md) for detailed instructions
- Explore [notebooks/emotion_recognition_demo.ipynb](notebooks/emotion_recognition_demo.ipynb) for interactive learning
- Check [EXAMPLES.md](EXAMPLES.md) for more usage examples
- Read [DOCUMENTATION.md](DOCUMENTATION.md) to understand how CNNs work

## Troubleshooting

**"No training data found"**
- Make sure images are in `data/train/<emotion>/` folders

**"No face detected"**
- Ensure faces are clearly visible and well-lit

**Low accuracy**
- Collect more training images (aim for 30+ per emotion)
- Train for more epochs: `--epochs 50`

## Tips for Best Results

1. **Good lighting**: Take photos in well-lit areas
2. **Clear faces**: Ensure face is centered and visible
3. **Variety**: Different angles, backgrounds, lighting
4. **Balance**: Similar number of photos for each emotion

Happy learning! 🚀
