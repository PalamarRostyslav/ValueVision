"""
Dataset creation and export utilities.
"""

import pickle
import os
from typing import List
from datasets import Dataset, DatasetDict

from config.settings import OUTPUT_DIR, TRAIN_FILE, TEST_FILE


class DatasetCreator:
    """Handles dataset creation and export operations."""
    
    def __init__(self, output_dir: str = OUTPUT_DIR):
        """Initialize with output directory."""
        self.output_dir = output_dir
        self._ensure_output_dir()
    
    def _ensure_output_dir(self) -> None:
        """Create output directory if it doesn't exist."""
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)
    
    def create_huggingface_dataset(self, train_items: List, test_items: List) -> DatasetDict:
        """
        Create HuggingFace Dataset from items.
        
        Args:
            train_items: List of training Item objects
            test_items: List of test Item objects
            
        Returns:
            DatasetDict with train and test splits
        """
        # Extract prompts and prices
        train_prompts = [item.prompt for item in train_items]
        train_prices = [item.price for item in train_items]
        test_prompts = [item.test_prompt() for item in test_items]
        test_prices = [item.price for item in test_items]
        
        # Create datasets
        train_dataset = Dataset.from_dict({
            "text": train_prompts, 
            "price": train_prices
        })
        test_dataset = Dataset.from_dict({
            "text": test_prompts, 
            "price": test_prices
        })
        
        return DatasetDict({
            "train": train_dataset,
            "test": test_dataset
        })
    
    def save_items_to_pickle(self, train_items: List, test_items: List) -> None:
        """
        Save Item objects to pickle files.
        
        Args:
            train_items: List of training Item objects
            test_items: List of test Item objects
        """
        train_path = os.path.join(self.output_dir, TRAIN_FILE)
        test_path = os.path.join(self.output_dir, TEST_FILE)
        
        with open(train_path, 'wb') as file:
            pickle.dump(train_items, file)
        
        with open(test_path, 'wb') as file:
            pickle.dump(test_items, file)
        
        print(f"Saved training items to {train_path}")
        print(f"Saved test items to {test_path}")
    
    def save_dataset_to_disk(self, dataset: DatasetDict, path: str = None) -> None:
        """
        Save HuggingFace dataset to disk.
        
        Args:
            dataset: DatasetDict to save
            path: Path to save to (defaults to output_dir/dataset)
        """
        if path is None:
            path = os.path.join(self.output_dir, "dataset")
        
        dataset.save_to_disk(path)
        print(f"Saved dataset to {path}")
