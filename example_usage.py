"""
Example usage of RADNet in Python code
This demonstrates how to use RADNet programmatically
"""

import torch
from model import build_model
from PIL import Image
import torchvision.transforms as transforms


def example_basic_usage():
    """Basic example: Create and use the model"""
    print("Example 1: Basic model usage")
    
    # Build the model
    model = build_model(
        in_channels=3,
        out_channels=3,
        base_channels=64,
        num_blocks=6
    )
    
    # Print model info
    num_params = sum(p.numel() for p in model.parameters())
    print(f"Model created with {num_params:,} parameters")
    
    # Create a random input (simulating a hazy image)
    batch_size = 1
    height, width = 256, 256
    hazy_image = torch.randn(batch_size, 3, height, width)
    
    # Run inference
    model.eval()
    with torch.no_grad():
        dehazed_image = model(hazy_image)
    
    print(f"Input shape: {hazy_image.shape}")
    print(f"Output shape: {dehazed_image.shape}")
    print()


def example_load_and_process_image():
    """Example: Load and process a real image"""
    print("Example 2: Load and process a real image")
    
    # This is a template - replace with your actual image path
    # image_path = 'path/to/your/hazy_image.jpg'
    
    # Transform to convert image to tensor
    transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
    ])
    
    # Load image (commented out - requires actual image file)
    # image = Image.open(image_path).convert('RGB')
    # image_tensor = transform(image).unsqueeze(0)  # Add batch dimension
    
    # Create model
    model = build_model()
    model.eval()
    
    # Process image (commented out - requires actual image)
    # with torch.no_grad():
    #     dehazed = model(image_tensor)
    
    # Save result (commented out - requires actual processing)
    # dehazed_image = dehazed.squeeze(0)  # Remove batch dimension
    # dehazed_image = transforms.ToPILImage()(dehazed_image)
    # dehazed_image.save('dehazed_output.jpg')
    
    print("Template code for loading and processing real images")
    print("Uncomment the code and provide your image path")
    print()


def example_with_gpu():
    """Example: Using GPU acceleration"""
    print("Example 3: Using GPU (if available)")
    
    # Check if GPU is available
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Create model and move to GPU
    model = build_model()
    model = model.to(device)
    
    # Create input and move to GPU
    hazy_image = torch.randn(1, 3, 256, 256).to(device)
    
    # Run inference
    model.eval()
    with torch.no_grad():
        dehazed_image = model(hazy_image)
    
    print(f"Processing on {device} completed successfully")
    print()


def example_batch_processing():
    """Example: Process multiple images in a batch"""
    print("Example 4: Batch processing")
    
    # Create model
    model = build_model()
    model.eval()
    
    # Create a batch of images
    batch_size = 4
    images = torch.randn(batch_size, 3, 256, 256)
    
    # Process batch
    with torch.no_grad():
        dehazed_batch = model(images)
    
    print(f"Processed batch of {batch_size} images")
    print(f"Input batch shape: {images.shape}")
    print(f"Output batch shape: {dehazed_batch.shape}")
    print()


def example_load_pretrained():
    """Example: Load a pretrained model"""
    print("Example 5: Loading pretrained model")
    
    # Create model
    model = build_model()
    
    # Load pretrained weights (template - requires actual checkpoint)
    # checkpoint_path = './checkpoints/best_model.pth'
    # checkpoint = torch.load(checkpoint_path)
    # model.load_state_dict(checkpoint['model_state_dict'])
    
    print("Template code for loading pretrained model")
    print("Provide checkpoint path to load trained weights")
    print()


if __name__ == '__main__':
    print("=" * 60)
    print("RADNet Usage Examples")
    print("=" * 60)
    print()
    
    # Run examples
    example_basic_usage()
    example_load_and_process_image()
    example_with_gpu()
    example_batch_processing()
    example_load_pretrained()
    
    print("=" * 60)
    print("Examples completed!")
    print("For more information, see README.md")
    print("=" * 60)
