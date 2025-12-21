"""
Configuration settings for sequence-based deep learning model
"""

# Data paths
DATA_PATH = "../data/processed/temperature_cleaned.csv"
OUTPUT_DIR = "outputs"

# Sequence parameters
SEQUENCE_LENGTH = 15  # Number of timestamps per sequence
FEATURE_COLUMN = "temperature"

# Model parameters
MODEL_TYPE = "LSTM"  # Options: "LSTM", "GRU", "RNN"
LSTM_UNITS = 128
DROPOUT_RATE = 0.3
DENSE_UNITS = 64

# Training parameters
BATCH_SIZE = 32
EPOCHS = 30
VALIDATION_SPLIT = 0.2
RANDOM_STATE = 42

# Label thresholds (same as Deliverable 3)
COLD_THRESHOLD = 10
NORMAL_THRESHOLD = 25
WARM_THRESHOLD = 35

