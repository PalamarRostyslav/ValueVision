"""
Fine-tuning strategies for optimizing predictors.

This module contains the abstract base class for fine-tuning strategies.
Concrete implementations are located in the strategies/ subdirectory.
"""

from abc import ABC, abstractmethod
from typing import List, Callable

from src.data.models import Item


class FineTuning(ABC):
    """
    Abstract base class for fine-tuning strategies.
    
    All fine-tuning implementations should inherit from this class
    and implement the optimize_predictor method.
    """
    
    def __init__(self, name: str):
        """
        Initialize the fine-tuning strategy.
        
        Args:
            name: Human-readable name for this strategy
        """
        self.name = name
    
    @abstractmethod
    def optimize_predictor(self, training_data: List[Item], validation_data: List[Item] = None) -> Callable:
        """
        Optimize a predictor function using the given training data.
        
        Args:
            training_data: List of Item objects for training
            validation_data: Optional validation data for evaluation
            
        Returns:
            Optimized predictor function
        """
        pass
