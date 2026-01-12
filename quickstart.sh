#!/bin/bash

# Quick Start Script for RADNet
# This script demonstrates how to set up and use RADNet

echo "=== RADNet Quick Start ==="
echo ""

# Step 1: Install dependencies
echo "Step 1: Installing dependencies..."
echo "Run: pip install -r requirements.txt"
echo ""

# Step 2: Prepare data
echo "Step 2: Prepare your dataset"
echo "Create the following directory structure:"
echo "  data/"
echo "  ├── train/"
echo "  │   ├── hazy/    # Hazy training images"
echo "  │   └── clear/   # Clear (ground truth) training images"
echo "  ├── val/"
echo "  │   ├── hazy/    # Hazy validation images"
echo "  │   └── clear/   # Clear validation images"
echo "  └── test/"
echo "      ├── hazy/    # Hazy test images"
echo "      └── clear/   # Clear test images (optional)"
echo ""

# Step 3: Training
echo "Step 3: Train the model"
echo "Run: python train.py --train_hazy_dir ./data/train/hazy --train_clear_dir ./data/train/clear --val_hazy_dir ./data/val/hazy --val_clear_dir ./data/val/clear"
echo ""

# Step 4: Testing
echo "Step 4: Test the model"
echo "Run: python test.py --test_hazy_dir ./data/test/hazy --test_clear_dir ./data/test/clear --checkpoint ./checkpoints/best_model.pth"
echo ""

# Step 5: Demo
echo "Step 5: Dehaze a single image"
echo "Run: python demo.py --input path/to/hazy_image.jpg --output dehazed_image.png --checkpoint ./checkpoints/best_model.pth"
echo ""

echo "For more details, see README.md"
