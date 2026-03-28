"""
ScanAid: Shared configuration constants.
Import this module instead of hardcoding values across scripts.
"""

import os

# --- Image dimensions (must match MobileNetV2 input) ---
IMG_WIDTH = 224
IMG_HEIGHT = 224
IMG_SHAPE = (IMG_WIDTH, IMG_HEIGHT, 3)

# --- Training ---
BATCH_SIZE = 32

# --- Directory paths (relative to project root) ---
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

DATA_RAW_DIR = os.path.join(PROJECT_ROOT, "syndrome_dataset")
DATA_PROCESSED_DIR = os.path.join(PROJECT_ROOT, "data", "processed")

DATA_DOWN_SYNDROME_DIR = os.path.join(DATA_PROCESSED_DIR, "down_syndrome")
DATA_ANGELMAN_DIR = os.path.join(DATA_PROCESSED_DIR, "angelman")

# --- Model save locations ---
MODELS_DIR = os.path.join(PROJECT_ROOT, "models")
DOWN_SYNDROME_MODEL_PATH = os.path.join(MODELS_DIR, "down_syndrome_detector.h5")
ANGELMAN_MODEL_PATH = os.path.join(MODELS_DIR, "angelman_siamese_model.h5")

# --- Reports ---
REPORTS_DIR = os.path.join(PROJECT_ROOT, "reports")
