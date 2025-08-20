"""
Abstract base class for AI frontier model fine-tuning.

This module defines the interface that all AI frontier models must implement
for fine-tuning on price estimation tasks.
"""

import json
import re
from abc import ABC, abstractmethod
from typing import List, Callable, Dict
from pathlib import Path

from src.data.models import Item


class AIFineTuningModel(ABC):
    """
    Abstract base class for AI frontier model fine-tuning.
    
    This class defines the interface for fine-tuning AI models like OpenAI GPT
    and Anthropic Claude for price estimation tasks.
    """
    
    def __init__(self, model_name: str, provider: str):
        """
        Initialize the AI fine-tuning model.
        
        Args:
            model_name: Name of the base model to fine-tune
            provider: AI provider (e.g., 'openai', 'anthropic')
        """
        self.model_name = model_name
        self.provider = provider
        self.fine_tuned_model_id = None
        self.job_id = None
        
    def messages_for(self, item: Item) -> List[Dict[str, str]]:
        """
        Create training messages for a given item.
        
        Args:
            item: Item to create messages for
            
        Returns:
            List of message dictionaries in chat format
        """
        system_message = "You estimate prices of items. Reply only with the price, no explanation"
        user_prompt = item.test_prompt().replace(" to the nearest dollar", "").replace("\n\nPrice is $", "")
        
        return [
            {"role": "system", "content": system_message},
            {"role": "user", "content": user_prompt},
            {"role": "assistant", "content": f"Price is ${item.price:.2f}"}
        ]
    
    def messages_for_inference(self, item: Item) -> List[Dict[str, str]]:
        """
        Create inference messages for a given item (without the price).
        
        Args:
            item: Item to create messages for
            
        Returns:
            List of message dictionaries for inference
        """
        system_message = "You estimate prices of items. Reply only with the price, no explanation"
        user_prompt = item.test_prompt().replace(" to the nearest dollar", "").replace("\n\nPrice is $", "")
        
        return [
            {"role": "system", "content": system_message},
            {"role": "user", "content": user_prompt},
            {"role": "assistant", "content": "Price is $"}
        ]
    
    def make_jsonl(self, items: List[Item]) -> str:
        """
        Convert items to JSONL format for fine-tuning.
        
        Args:
            items: List of items to convert
            
        Returns:
            JSONL string representation
        """
        result = ""
        for item in items:
            messages = self.messages_for(item)
            messages_str = json.dumps(messages)
            result += '{"messages": ' + messages_str + '}\n'
        return result.strip()
    
    def write_jsonl(self, items: List[Item], filename: str) -> None:
        """
        Write items to JSONL file.
        
        Args:
            items: List of items to write
            filename: Output filename
        """
        filepath = Path(filename)
        filepath.parent.mkdir(parents=True, exist_ok=True)
        
        with open(filename, "w") as f:
            jsonl = self.make_jsonl(items)
            f.write(jsonl)
        
        print(f"Written {len(items)} items to {filename}")
    
    def get_price(self, response_text: str) -> float:
        """
        Extract price from model response text.
        
        Args:
            response_text: Text response from the model
            
        Returns:
            Extracted price as float
        """
        s = response_text.replace('$', '').replace(',', '')
        match = re.search(r"[-+]?\d*\.\d+|\d+", s)
        return float(match.group()) if match else 0.0
    
    @abstractmethod
    def upload_training_files(self, train_items: List[Item], validation_items: List[Item]) -> Dict[str, str]:
        """
        Upload training and validation files to the AI provider.
        
        Args:
            train_items: Training data items
            validation_items: Validation data items
            
        Returns:
            Dictionary containing file IDs
        """
        pass
    
    @abstractmethod
    def start_fine_tuning(self, file_ids: Dict[str, str], **kwargs) -> str:
        """
        Start the fine-tuning job.
        
        Args:
            file_ids: Dictionary containing training and validation file IDs
            **kwargs: Additional fine-tuning parameters
            
        Returns:
            Job ID for the fine-tuning job
        """
        pass
    
    @abstractmethod
    def monitor_fine_tuning(self, job_id: str, check_interval: int = 5) -> str:
        """
        Monitor fine-tuning progress and wait for completion.
        
        Args:
            job_id: Fine-tuning job ID
            check_interval: Time in seconds between status checks
            
        Returns:
            Fine-tuned model ID
        """
        pass
    
    @abstractmethod
    def create_predictor(self, model_id: str) -> Callable:
        """
        Create a predictor function using the fine-tuned model.
        
        Args:
            model_id: Fine-tuned model ID
            
        Returns:
            Predictor function
        """
        pass
    
    def fine_tune_and_create_predictor(self, train_items: List[Item], 
            validation_items: List[Item] = None,
                                     **kwargs) -> Callable:
        """
        Complete fine-tuning pipeline: upload data, fine-tune, and create predictor.
        
        Args:
            train_items: Training data items
            validation_items: Validation data items (optional)
            **kwargs: Additional fine-tuning parameters
            
        Returns:
            Fine-tuned predictor function
        """
        print(f"\nStarting {self.provider} fine-tuning with {self.model_name}")
        print(f"Training items: {len(train_items)}")
        if validation_items:
            print(f"Validation items: {len(validation_items)}")
        
        # Upload files
        print("\n1. Uploading training files...")
        file_ids = self.upload_training_files(train_items, validation_items)
        
        # Start fine-tuning
        print("\n2. Starting fine-tuning job...")
        job_id = self.start_fine_tuning(file_ids, **kwargs)
        self.job_id = job_id
        
        # Monitor progress
        print("\n3. Monitoring fine-tuning progress...")
        model_id = self.monitor_fine_tuning(job_id)
        self.fine_tuned_model_id = model_id
        
        # Create predictor
        print("\n4. Creating predictor function...")
        predictor = self.create_predictor(model_id)
        
        # Add optimization info
        predictor.optimization_info = {
            'strategy': f'{self.provider.title()} Fine-tuning',
            'base_model': self.model_name,
            'fine_tuned_model': model_id,
            'job_id': job_id,
            'training_samples': len(train_items),
            'validation_samples': len(validation_items) if validation_items else 0
        }
        
        print(f"\n✓ Fine-tuning completed! Model ID: {model_id}")
        return predictor
