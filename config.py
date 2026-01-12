"""
Configuration file for RADNet training and testing
"""

import os

class Config:
    """Configuration class for RADNet"""
    
    # Model parameters
    MODEL_NAME = 'RADNet'
    IN_CHANNELS = 3
    OUT_CHANNELS = 3
    BASE_CHANNELS = 64
    
    # Training parameters
    BATCH_SIZE = 16
    NUM_EPOCHS = 100
    LEARNING_RATE = 1e-4
    WEIGHT_DECAY = 1e-4
    
    # Data parameters
    TRAIN_DIR = './data/train'
    VAL_DIR = './data/val'
    TEST_DIR = './data/test'
    IMG_SIZE = (256, 256)
    
    # Training settings
    NUM_WORKERS = 4
    PIN_MEMORY = True
    SAVE_INTERVAL = 10
    VAL_INTERVAL = 5
    
    # Paths
    CHECKPOINT_DIR = './checkpoints'
    LOG_DIR = './logs'
    RESULT_DIR = './results'
    
    # Device
    DEVICE = 'cuda'
    
    @classmethod
    def create_dirs(cls):
        """Create necessary directories"""
        os.makedirs(cls.CHECKPOINT_DIR, exist_ok=True)
        os.makedirs(cls.LOG_DIR, exist_ok=True)
        os.makedirs(cls.RESULT_DIR, exist_ok=True)
