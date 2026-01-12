"""
Testing/Evaluation script for RADNet
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
import argparse
import os
import numpy as np

from model import build_model
from dataloader import DehazingDataset, get_val_transform
from utils import load_checkpoint, calculate_psnr, save_image, visualize_results, AverageMeter
from config import Config


def test(model, test_loader, device, save_results=True, result_dir='./results'):
    """Test the model"""
    model.eval()
    psnr_meter = AverageMeter()
    
    if save_results:
        os.makedirs(result_dir, exist_ok=True)
    
    with torch.no_grad():
        for idx, data in enumerate(tqdm(test_loader, desc='Testing')):
            if isinstance(data, tuple):
                hazy, clear = data
                hazy = hazy.to(device)
                clear = clear.to(device)
                has_gt = True
            else:
                hazy = data.to(device)
                clear = None
                has_gt = False
            
            # Forward pass
            dehazed = model(hazy)
            
            # Calculate PSNR if ground truth is available
            if has_gt:
                psnr = calculate_psnr(dehazed, clear)
                psnr_meter.update(psnr.item(), hazy.size(0))
            
            # Save results
            if save_results:
                for i in range(hazy.size(0)):
                    img_idx = idx * test_loader.batch_size + i
                    
                    # Save dehazed image
                    dehazed_path = os.path.join(result_dir, f'dehazed_{img_idx:04d}.png')
                    save_image(dehazed[i:i+1], dehazed_path)
                    
                    # Save visualization
                    viz_path = os.path.join(result_dir, f'comparison_{img_idx:04d}.png')
                    visualize_results(
                        hazy[i:i+1],
                        clear[i:i+1] if has_gt else None,
                        dehazed[i:i+1],
                        save_path=viz_path
                    )
    
    if has_gt:
        print(f'Average PSNR: {psnr_meter.avg:.2f} dB')
    else:
        print('Testing completed (no ground truth available)')


def main(args):
    """Main testing function"""
    # Set device
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')
    
    # Build model
    print('Building model...')
    model = build_model(
        in_channels=3,
        out_channels=3,
        base_channels=args.base_channels,
        num_blocks=args.num_blocks
    )
    model = model.to(device)
    
    # Load checkpoint
    if args.checkpoint:
        print(f'Loading checkpoint from {args.checkpoint}...')
        checkpoint = torch.load(args.checkpoint, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        print(f"Loaded checkpoint from epoch {checkpoint.get('epoch', 'unknown')}")
    else:
        print('Warning: No checkpoint provided, using randomly initialized model')
    
    # Create test dataset
    print('Loading test data...')
    test_dataset = DehazingDataset(
        hazy_dir=args.test_hazy_dir,
        clear_dir=args.test_clear_dir if args.test_clear_dir and os.path.exists(args.test_clear_dir) else None,
        transform=get_val_transform(tuple(args.img_size)),
        img_size=tuple(args.img_size)
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True
    )
    
    # Test
    print('Testing...')
    test(model, test_loader, device, 
         save_results=args.save_results,
         result_dir=args.result_dir)
    
    print('Testing completed!')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Test RADNet')
    
    # Data parameters
    parser.add_argument('--test_hazy_dir', type=str, required=True,
                        help='Directory with test hazy images')
    parser.add_argument('--test_clear_dir', type=str, default='',
                        help='Directory with test clear images (optional)')
    
    # Model parameters
    parser.add_argument('--checkpoint', type=str, required=True,
                        help='Path to model checkpoint')
    parser.add_argument('--base_channels', type=int, default=64,
                        help='Base number of channels')
    parser.add_argument('--num_blocks', type=int, default=6,
                        help='Number of retinal blocks')
    
    # Testing parameters
    parser.add_argument('--batch_size', type=int, default=1,
                        help='Batch size')
    parser.add_argument('--num_workers', type=int, default=4,
                        help='Number of data loading workers')
    parser.add_argument('--img_size', type=int, nargs=2, default=[256, 256],
                        help='Image size (height width)')
    
    # Output parameters
    parser.add_argument('--save_results', action='store_true', default=True,
                        help='Save dehazed results')
    parser.add_argument('--result_dir', type=str, default='./results',
                        help='Directory to save results')
    parser.add_argument('--device', type=str, default='cuda',
                        help='Device to use (cuda or cpu)')
    
    args = parser.parse_args()
    main(args)
