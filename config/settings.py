"""
Configuration settings for the ValueVision project.
"""

import os
from typing import List

# Dataset configuration
DATASET_NAMES: List[str] = [
    "Appliances",
    "Automotive",
    "Electronics",
    "Office_Products",
    "Tools_and_Home_Improvement",
    "Cell_Phones_and_Accessories",
    "Toys_and_Games",
    "Musical_Instruments",
]

# Model configuration
BASE_MODEL = "meta-llama/Meta-Llama-3.1-8B"

# Item processing configuration
MIN_TOKENS = 150
MAX_TOKENS = 160
MIN_CHARS = 300
CEILING_CHARS = MAX_TOKENS * 7

# Data loading configuration
CHUNK_SIZE = 1000
MIN_PRICE = 0.5
MAX_PRICE = 999.49

# Sampling configuration
RANDOM_SEED = 42
MAX_ITEMS_PER_PRICE = 1200
HIGH_PRICE_THRESHOLD = 240
AUTOMOTIVE_WEIGHT = 1
OTHER_CATEGORY_WEIGHT = 5

# Train/test split configuration
TRAIN_SIZE = 400_000
TEST_SIZE = 2_000

# Output paths
OUTPUT_DIR = "output"
TRAIN_FILE = "train.pkl"
TEST_FILE = "test.pkl"

# HuggingFace dataset configuration
HF_DATASET_NAME = "McAuley-Lab/Amazon-Reviews-2023"
