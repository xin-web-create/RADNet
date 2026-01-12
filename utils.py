"""
Utility functions for RADNet
"""

import torch
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import os


def save_checkpoint(model, optimizer, epoch, loss, filepath):
    """Save model checkpoint"""
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss,
    }
    torch.save(checkpoint, filepath)
    print(f"Checkpoint saved to {filepath}")


def load_checkpoint(model, optimizer, filepath, device='cpu'):
    """Load model checkpoint"""
    checkpoint = torch.load(filepath, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    if optimizer is not None:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    epoch = checkpoint['epoch']
    loss = checkpoint['loss']
    print(f"Checkpoint loaded from {filepath} (Epoch: {epoch}, Loss: {loss:.4f})")
    return epoch, loss


def calculate_psnr(img1, img2):
    """
    Calculate Peak Signal-to-Noise Ratio (PSNR)
    
    Args:
        img1: First image tensor
        img2: Second image tensor
        
    Returns:
        PSNR value in dB
    """
    mse = torch.mean((img1 - img2) ** 2)
    if mse == 0:
        return float('inf')
    return 20 * torch.log10(1.0 / torch.sqrt(mse))


def calculate_ssim(img1, img2):
    """
    Calculate Structural Similarity Index (SSIM)
    Simple implementation for single channel
    """
    C1 = (0.01) ** 2
    C2 = (0.03) ** 2
    
    mu1 = torch.mean(img1)
    mu2 = torch.mean(img2)
    
    sigma1 = torch.var(img1)
    sigma2 = torch.var(img2)
    sigma12 = torch.mean((img1 - mu1) * (img2 - mu2))
    
    ssim = ((2 * mu1 * mu2 + C1) * (2 * sigma12 + C2)) / \
           ((mu1 ** 2 + mu2 ** 2 + C1) * (sigma1 + sigma2 + C2))
    
    return ssim


def tensor_to_image(tensor):
    """Convert tensor to numpy image"""
    image = tensor.cpu().detach().numpy()
    if len(image.shape) == 4:
        image = image[0]
    image = np.transpose(image, (1, 2, 0))
    image = np.clip(image, 0, 1)
    return image


def save_image(tensor, filepath):
    """Save tensor as image"""
    image = tensor_to_image(tensor)
    image = (image * 255).astype(np.uint8)
    Image.fromarray(image).save(filepath)


def visualize_results(hazy, clear, dehazed, save_path=None):
    """
    Visualize dehazing results
    
    Args:
        hazy: Hazy input image tensor
        clear: Clear ground truth image tensor (can be None)
        dehazed: Dehazed output image tensor
        save_path: Path to save the visualization
    """
    fig, axes = plt.subplots(1, 3 if clear is not None else 2, figsize=(15, 5))
    
    # Hazy image
    axes[0].imshow(tensor_to_image(hazy))
    axes[0].set_title('Hazy Input')
    axes[0].axis('off')
    
    # Dehazed image
    idx = 1
    axes[idx].imshow(tensor_to_image(dehazed))
    axes[idx].set_title('Dehazed Output')
    axes[idx].axis('off')
    
    # Clear ground truth (if available)
    if clear is not None:
        idx = 2
        axes[idx].imshow(tensor_to_image(clear))
        axes[idx].set_title('Clear Ground Truth')
        axes[idx].axis('off')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Visualization saved to {save_path}")
    else:
        plt.show()
    
    plt.close()


class AverageMeter:
    """Compute and store the average and current value"""
    
    def __init__(self):
        self.reset()
        
    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0
        
    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count
