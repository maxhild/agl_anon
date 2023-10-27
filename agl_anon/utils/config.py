"""
Configuration settings for the CRNN OCR project.

This file centralizes the configuration settings, making it easier to modify hyperparameters, paths, and other settings.
"""

# ========================
# Dataset Configurations
# ========================

# Paths to training, validation, and test datasets
TRAIN_DATASET_PATH = "path/to/train_dataset"
VAL_DATASET_PATH = "path/to/val_dataset"
TEST_DATASET_PATH = "path/to/test_dataset"

# Image dimensions after preprocessing
IMAGE_WIDTH = 128
IMAGE_HEIGHT = 32

# ========================
# Model Configurations
# ========================

# Number of classes (characters) in the dataset. Add 1 for CTC 'blank' character.
NUM_CLASSES = 27

# LSTM configurations
LSTM_HIDDEN_SIZE = 128
LSTM_NUM_LAYERS = 2

# ========================
# Training Configurations
# ========================

# Number of epochs for training
EPOCHS = 10

# Learning rate
LEARNING_RATE = 0.001

# Batch size for training and evaluation
BATCH_SIZE = 32

# Path to save the trained model
MODEL_SAVE_PATH = "path/to/save_model"

# ========================
# Evaluation Configurations
# ========================

# Path to the trained model for evaluation
MODEL_PATH_FOR_EVAL = "path/to/trained_model"

# ========================
# Miscellaneous Configurations
# ========================

# Seed for reproducibility
SEED = 42

# Whether to use GPU (if available)
USE_GPU = True

