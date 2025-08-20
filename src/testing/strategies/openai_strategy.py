"""
OpenAI Fine-tuning Strategy

This module implements a fine-tuning strategy using OpenAI's fine-tuning API.
"""

from typing import List, Callable
from ..fine_tuning import FineTuning
from src.data.models import Item
from src.ai_models import OpenAIFineTuning
from config.settings import OPENAI_MODEL, OPENAI_API_KEY


class FineTuningOpenAI(FineTuning):
    """
    Fine-tuning strategy using OpenAI's API.
    
    This strategy uploads training data to OpenAI, fine-tunes a model,
    and returns a predictor function that uses the fine-tuned model.
    """
    
    def __init__(self, model_name: str = None, api_key: str = None, use_existing: bool = True, model_id: str = None, **kwargs):
        """
        Initialize OpenAI fine-tuning strategy.
        
        Args:
            model_name: OpenAI model to fine-tune (defaults to settings.OPENAI_MODEL)
            api_key: OpenAI API key (defaults to settings.OPENAI_API_KEY)
            use_existing: Whether to reuse existing fine-tuned models (default: True)
            model_id: Specific model ID to use (skips training if provided)
            **kwargs: Additional fine-tuning parameters
        """
        # Use defaults from settings if not provided
        model_name = model_name or OPENAI_MODEL
        api_key = api_key or OPENAI_API_KEY
        
        strategy_name = f"OpenAI Fine-tuning ({model_name})"
        if model_id:
            strategy_name += f" [Model: {model_id}]"
        elif not use_existing:
            strategy_name += " [Force New]"
        
        super().__init__(strategy_name)
        self.model_name = model_name
        self.api_key = api_key
        self.use_existing = use_existing
        self.model_id = model_id
        self.fine_tuning_params = kwargs
        
    def optimize_predictor(self, training_data: List[Item], validation_data: List[Item] = None) -> Callable:
        """
        Optimize predictor using OpenAI fine-tuning.
        
        Args:
            training_data: Training data for fine-tuning
            validation_data: Validation data for evaluation
            
        Returns:
            Fine-tuned predictor function
        """
        print(f"\nStarting {self.name}...")
        print(f"Training samples: {len(training_data)}")
        if validation_data:
            print(f"Validation samples: {len(validation_data)}")
        
        # Initialize OpenAI fine-tuning
        openai_ft = OpenAIFineTuning(
            model_name=self.model_name,
            api_key=self.api_key,
            use_existing=self.use_existing
        )
        
        # Add model_id to parameters if specified
        params = self.fine_tuning_params.copy()
        if self.model_id:
            params['model_id'] = self.model_id
        
        # Perform fine-tuning and get predictor
        predictor = openai_ft.fine_tune_and_create_predictor(
            train_items=training_data,
            validation_items=validation_data,
            **params
        )
        
        return predictor
