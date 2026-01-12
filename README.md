# RADNet
RADNet: Image dehazing network based on retinal neuromorphic inspiration

## Overview

RADNet is a deep learning-based image dehazing network inspired by retinal processing mechanisms. The network uses adaptive layers and retinal-inspired blocks to effectively remove haze from images while preserving important details.

## Features

- **Retinal-inspired architecture**: Mimics biological retinal processing for adaptive dehazing
- **Multi-scale processing**: Handles various haze densities and image conditions
- **Efficient design**: Optimized for both quality and computational efficiency
- **Easy to use**: Simple training and inference scripts

## Installation

### Requirements

- Python 3.7+
- PyTorch 1.8+
- CUDA (optional, for GPU acceleration)

### Setup

1. Clone the repository:
```bash
git clone https://github.com/xin-web-create/RADNet.git
cd RADNet
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

### Data Preparation

Organize your dataset in the following structure:
```
data/
├── train/
│   ├── hazy/    # Hazy training images
│   └── clear/   # Clear (ground truth) training images
├── val/
│   ├── hazy/    # Hazy validation images
│   └── clear/   # Clear validation images
└── test/
    ├── hazy/    # Hazy test images
    └── clear/   # Clear test images (optional)
```

### Training

Train the model with default parameters:
```bash
python train.py \
    --train_hazy_dir ./data/train/hazy \
    --train_clear_dir ./data/train/clear \
    --val_hazy_dir ./data/val/hazy \
    --val_clear_dir ./data/val/clear
```

Train with custom parameters:
```bash
python train.py \
    --train_hazy_dir ./data/train/hazy \
    --train_clear_dir ./data/train/clear \
    --batch_size 16 \
    --epochs 100 \
    --lr 1e-4 \
    --base_channels 64 \
    --num_blocks 6
```

### Testing

Evaluate the model on test data:
```bash
python test.py \
    --test_hazy_dir ./data/test/hazy \
    --test_clear_dir ./data/test/clear \
    --checkpoint ./checkpoints/best_model.pth \
    --result_dir ./results
```

### Demo / Inference

Dehaze a single image:
```bash
python demo.py \
    --input path/to/hazy_image.jpg \
    --output path/to/dehazed_image.png \
    --checkpoint ./checkpoints/best_model.pth
```

Dehaze all images in a directory:
```bash
python demo.py \
    --input path/to/hazy_images/ \
    --output path/to/dehazed_results/ \
    --checkpoint ./checkpoints/best_model.pth
```

## Model Architecture

RADNet consists of:
- **Retinal Blocks**: Process features with residual connections
- **Adaptive Layers**: Channel-wise attention inspired by retinal adaptation
- **Feature Fusion**: Multi-scale feature integration
- **Residual Connection**: Preserves input information

## Configuration

Edit `config.py` to modify default settings:
- Model parameters (channels, blocks)
- Training parameters (batch size, learning rate, epochs)
- Data paths and image sizes
- Device settings

## Results

The model outputs:
- Dehazed images in the results directory
- Training logs in TensorBoard format (logs directory)
- Model checkpoints (checkpoints directory)

View training progress with TensorBoard:
```bash
tensorboard --logdir logs
```

## Citation

If you use this code in your research, please cite:
```
@misc{radnet,
  title={RADNet: Image dehazing network based on retinal neuromorphic inspiration},
  author={Your Name},
  year={2026}
}
```

## License

This project is open source and available under the MIT License.

## Acknowledgments

This work is inspired by biological retinal processing mechanisms and recent advances in deep learning for image restoration.
