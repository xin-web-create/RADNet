"""
Demo script for RADNet - Dehaze a single image or directory of images
"""

import torch
import argparse
import os
from PIL import Image
import torchvision.transforms as transforms

from model import build_model
from utils import save_image, tensor_to_image
import matplotlib.pyplot as plt


def load_image(image_path, img_size=(256, 256)):
    """Load and preprocess a single image"""
    image = Image.open(image_path).convert('RGB')
    
    transform = transforms.Compose([
        transforms.Resize(img_size),
        transforms.ToTensor(),
    ])
    
    return transform(image).unsqueeze(0)


def dehaze_image(model, image_tensor, device):
    """Dehaze a single image tensor"""
    model.eval()
    with torch.no_grad():
        image_tensor = image_tensor.to(device)
        dehazed = model(image_tensor)
    return dehazed


def main(args):
    """Main demo function"""
    # Set device
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')
    
    # Build model
    print('Loading model...')
    model = build_model(
        in_channels=3,
        out_channels=3,
        base_channels=args.base_channels,
        num_blocks=args.num_blocks
    )
    model = model.to(device)
    
    # Load checkpoint
    if args.checkpoint:
        checkpoint = torch.load(args.checkpoint, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        print(f"Loaded checkpoint from epoch {checkpoint.get('epoch', 'unknown')}")
    else:
        print('Warning: No checkpoint provided, using randomly initialized model')
    
    # Process single image or directory
    if os.path.isfile(args.input):
        # Single image
        print(f'Processing image: {args.input}')
        
        # Load image
        hazy_img = load_image(args.input, tuple(args.img_size))
        
        # Dehaze
        dehazed_img = dehaze_image(model, hazy_img, device)
        
        # Save result
        output_path = args.output if args.output else 'dehazed_output.png'
        save_image(dehazed_img, output_path)
        print(f'Dehazed image saved to: {output_path}')
        
        # Display if requested
        if args.display:
            fig, axes = plt.subplots(1, 2, figsize=(12, 6))
            axes[0].imshow(tensor_to_image(hazy_img))
            axes[0].set_title('Hazy Input')
            axes[0].axis('off')
            axes[1].imshow(tensor_to_image(dehazed_img))
            axes[1].set_title('Dehazed Output')
            axes[1].axis('off')
            plt.tight_layout()
            plt.show()
            
    elif os.path.isdir(args.input):
        # Directory of images
        print(f'Processing directory: {args.input}')
        
        # Create output directory
        output_dir = args.output if args.output else './dehazed_results'
        os.makedirs(output_dir, exist_ok=True)
        
        # Get all image files
        image_files = [f for f in os.listdir(args.input) 
                      if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp'))]
        
        print(f'Found {len(image_files)} images')
        
        # Process each image
        for img_file in image_files:
            img_path = os.path.join(args.input, img_file)
            print(f'Processing: {img_file}')
            
            # Load and dehaze
            hazy_img = load_image(img_path, tuple(args.img_size))
            dehazed_img = dehaze_image(model, hazy_img, device)
            
            # Save result
            output_path = os.path.join(output_dir, f'dehazed_{img_file}')
            save_image(dehazed_img, output_path)
        
        print(f'All dehazed images saved to: {output_dir}')
    else:
        print(f'Error: {args.input} is not a valid file or directory')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='RADNet Demo - Dehaze images')
    
    # Input/Output
    parser.add_argument('--input', type=str, required=True,
                        help='Input hazy image or directory')
    parser.add_argument('--output', type=str, default=None,
                        help='Output path for dehazed image(s)')
    
    # Model parameters
    parser.add_argument('--checkpoint', type=str, default=None,
                        help='Path to model checkpoint')
    parser.add_argument('--base_channels', type=int, default=64,
                        help='Base number of channels')
    parser.add_argument('--num_blocks', type=int, default=6,
                        help='Number of retinal blocks')
    
    # Other parameters
    parser.add_argument('--img_size', type=int, nargs=2, default=[256, 256],
                        help='Image size (height width)')
    parser.add_argument('--device', type=str, default='cuda',
                        help='Device to use (cuda or cpu)')
    parser.add_argument('--display', action='store_true',
                        help='Display results (only for single image)')
    
    args = parser.parse_args()
    main(args)
