"""
Data sampling and preprocessing utilities.
"""

import random
import numpy as np
from typing import List
from collections import defaultdict

from config.settings import (
    RANDOM_SEED, MAX_ITEMS_PER_PRICE, HIGH_PRICE_THRESHOLD,
    AUTOMOTIVE_WEIGHT, OTHER_CATEGORY_WEIGHT
)

class DataSampler:
    """Handles data sampling and preprocessing operations."""
    
    def __init__(self, seed: int = RANDOM_SEED):
        """Initialize the sampler with a random seed."""
        self.seed = seed
        self._set_seeds()
    
    def _set_seeds(self) -> None:
        """Set random seeds for reproducibility."""
        np.random.seed(self.seed)
        random.seed(self.seed)
    
    def create_price_slots(self, items: List) -> dict:
        """
        Group items by rounded price.
        
        Args:
            items: List of Item objects
            
        Returns:
            Dictionary with price as key and list of items as value
        """
        slots = defaultdict(list)
        for item in items:
            slots[round(item.price)].append(item)
        return slots
    
    def sample_items(self, items: List) -> List:
        """
        Sample items based on price distribution and category weights.
        
        Args:
            items: List of Item objects
            
        Returns:
            List of sampled Item objects
        """
        slots = self.create_price_slots(items)
        sample = []
        
        for price in range(1, 1000):
            slot = slots[price]
            
            if price >= HIGH_PRICE_THRESHOLD:
                sample.extend(slot)
            elif len(slot) <= MAX_ITEMS_PER_PRICE:
                sample.extend(slot)
            else:
                # Apply weighted sampling for large slots
                weights = np.array([
                    AUTOMOTIVE_WEIGHT if item.category == 'Automotive' 
                    else OTHER_CATEGORY_WEIGHT 
                    for item in slot
                ])
                weights = weights / np.sum(weights)
                
                selected_indices = np.random.choice(
                    len(slot), 
                    size=MAX_ITEMS_PER_PRICE, 
                    replace=False, 
                    p=weights
                )
                selected = [slot[i] for i in selected_indices]
                sample.extend(selected)
        
        return sample
    
    def shuffle_and_split(self, items: List, train_size: int, test_size: int) -> tuple:
        """
        Shuffle items and split into train and test sets.
        
        Args:
            items: List of items to split
            train_size: Number of items for training
            test_size: Number of items for testing
            
        Returns:
            Tuple of (train_items, test_items)
        """
        random.seed(self.seed)
        random.shuffle(items)
        
        train = items[:train_size]
        test = items[train_size:train_size + test_size]
        
        return train, test
