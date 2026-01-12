"""
Training script for RADNet
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
import argparse
import os

from model import build_model
from dataloader import create_dataloaders
from utils import save_checkpoint, calculate_psnr, AverageMeter, visualize_results
from config import Config


def train_one_epoch(model, train_loader, criterion, optimizer, device, epoch):
    """Train for one epoch"""
    model.train()
    losses = AverageMeter()
    psnr_meter = AverageMeter()
    
    pbar = tqdm(train_loader, desc=f'Epoch {epoch}')
    for batch_idx, (hazy, clear) in enumerate(pbar):
        hazy = hazy.to(device)
        clear = clear.to(device)
        
        # Forward pass
        dehazed = model(hazy)
        loss = criterion(dehazed, clear)
        
        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        # Calculate PSNR
        psnr = calculate_psnr(dehazed, clear)
        
        # Update metrics
        losses.update(loss.item(), hazy.size(0))
        psnr_meter.update(psnr.item(), hazy.size(0))
        
        # Update progress bar
        pbar.set_postfix({
            'loss': f'{losses.avg:.4f}',
            'psnr': f'{psnr_meter.avg:.2f}dB'
        })
    
    return losses.avg, psnr_meter.avg


def validate(model, val_loader, criterion, device):
    """Validate the model"""
    model.eval()
    losses = AverageMeter()
    psnr_meter = AverageMeter()
    
    with torch.no_grad():
        for hazy, clear in tqdm(val_loader, desc='Validation'):
            hazy = hazy.to(device)
            clear = clear.to(device)
            
            # Forward pass
            dehazed = model(hazy)
            loss = criterion(dehazed, clear)
            
            # Calculate PSNR
            psnr = calculate_psnr(dehazed, clear)
            
            # Update metrics
            losses.update(loss.item(), hazy.size(0))
            psnr_meter.update(psnr.item(), hazy.size(0))
    
    return losses.avg, psnr_meter.avg


def main(args):
    """Main training function"""
    # Create directories
    Config.create_dirs()
    
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
    
    # Print model info
    num_params = sum(p.numel() for p in model.parameters())
    print(f'Model parameters: {num_params:,}')
    
    # Define loss and optimizer
    criterion = nn.L1Loss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=args.scheduler_step, gamma=args.scheduler_gamma)
    
    # Create dataloaders
    print('Loading data...')
    train_loader, val_loader = create_dataloaders(
        train_hazy_dir=args.train_hazy_dir,
        train_clear_dir=args.train_clear_dir,
        val_hazy_dir=args.val_hazy_dir,
        val_clear_dir=args.val_clear_dir,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        img_size=tuple(args.img_size)
    )
    
    # Tensorboard writer
    writer = SummaryWriter(Config.LOG_DIR)
    
    # Training loop
    best_psnr = 0.0
    for epoch in range(1, args.epochs + 1):
        # Train
        train_loss, train_psnr = train_one_epoch(
            model, train_loader, criterion, optimizer, device, epoch
        )
        
        # Log training metrics
        writer.add_scalar('Train/Loss', train_loss, epoch)
        writer.add_scalar('Train/PSNR', train_psnr, epoch)
        writer.add_scalar('Train/LR', optimizer.param_groups[0]['lr'], epoch)
        
        print(f'Epoch {epoch}/{args.epochs} - Train Loss: {train_loss:.4f}, Train PSNR: {train_psnr:.2f}dB')
        
        # Validate
        if val_loader is not None and epoch % args.val_interval == 0:
            val_loss, val_psnr = validate(model, val_loader, criterion, device)
            writer.add_scalar('Val/Loss', val_loss, epoch)
            writer.add_scalar('Val/PSNR', val_psnr, epoch)
            print(f'Epoch {epoch}/{args.epochs} - Val Loss: {val_loss:.4f}, Val PSNR: {val_psnr:.2f}dB')
            
            # Save best model
            if val_psnr > best_psnr:
                best_psnr = val_psnr
                save_path = os.path.join(Config.CHECKPOINT_DIR, 'best_model.pth')
                save_checkpoint(model, optimizer, epoch, val_loss, save_path)
        
        # Save checkpoint
        if epoch % args.save_interval == 0:
            save_path = os.path.join(Config.CHECKPOINT_DIR, f'checkpoint_epoch_{epoch}.pth')
            save_checkpoint(model, optimizer, epoch, train_loss, save_path)
        
        # Update learning rate
        scheduler.step()
    
    # Save final model
    final_path = os.path.join(Config.CHECKPOINT_DIR, 'final_model.pth')
    save_checkpoint(model, optimizer, args.epochs, train_loss, final_path)
    
    writer.close()
    print('Training completed!')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train RADNet')
    
    # Data parameters
    parser.add_argument('--train_hazy_dir', type=str, default='./data/train/hazy',
                        help='Directory with training hazy images')
    parser.add_argument('--train_clear_dir', type=str, default='./data/train/clear',
                        help='Directory with training clear images')
    parser.add_argument('--val_hazy_dir', type=str, default='./data/val/hazy',
                        help='Directory with validation hazy images')
    parser.add_argument('--val_clear_dir', type=str, default='./data/val/clear',
                        help='Directory with validation clear images')
    
    # Model parameters
    parser.add_argument('--base_channels', type=int, default=64,
                        help='Base number of channels')
    parser.add_argument('--num_blocks', type=int, default=6,
                        help='Number of retinal blocks')
    
    # Training parameters
    parser.add_argument('--batch_size', type=int, default=16,
                        help='Batch size')
    parser.add_argument('--epochs', type=int, default=100,
                        help='Number of epochs')
    parser.add_argument('--lr', type=float, default=1e-4,
                        help='Learning rate')
    parser.add_argument('--weight_decay', type=float, default=1e-4,
                        help='Weight decay')
    parser.add_argument('--scheduler_step', type=int, default=30,
                        help='Learning rate scheduler step size')
    parser.add_argument('--scheduler_gamma', type=float, default=0.5,
                        help='Learning rate scheduler gamma')
    parser.add_argument('--num_workers', type=int, default=4,
                        help='Number of data loading workers')
    parser.add_argument('--img_size', type=int, nargs=2, default=[256, 256],
                        help='Image size (height width)')
    
    # Other parameters
    parser.add_argument('--device', type=str, default='cuda',
                        help='Device to use (cuda or cpu)')
    parser.add_argument('--save_interval', type=int, default=10,
                        help='Save checkpoint every N epochs')
    parser.add_argument('--val_interval', type=int, default=5,
                        help='Validate every N epochs')
    
    args = parser.parse_args()
    main(args)
