"""
OpenAI fine-tuning implementation.

This module implements fine-tuning for OpenAI models like GPT-4o-mini
for price estimation tasks.
"""

import time
from typing import List, Callable, Dict, Optional
import openai
from pathlib import Path

from .base import AIFineTuningModel
from .model_manager import ModelManager
from src.data.models import Item
from config.settings import OPENAI_API_KEY, OPENAI_MODEL


class OpenAIFineTuning(AIFineTuningModel):
    """
    OpenAI fine-tuning implementation for price estimation.
    
    Supports fine-tuning GPT models using OpenAI's fine-tuning API.
    """
    
    def __init__(self, model_name: str = None, api_key: str = None, use_existing: bool = True):
        """
        Initialize OpenAI fine-tuning.
        
        Args:
            model_name: OpenAI model to fine-tune (defaults to settings.OPENAI_MODEL)
            api_key: OpenAI API key (defaults to settings.OPENAI_API_KEY)
            use_existing: Whether to reuse existing fine-tuned models (default: True)
        """
        # Use defaults from settings if not provided
        model_name = model_name or OPENAI_MODEL
        api_key = api_key or OPENAI_API_KEY
        
        if not api_key:
            raise ValueError("OpenAI API key not found. Please set OPENAI_API_KEY in your .env file or pass it as a parameter.")
        
        super().__init__(model_name, "openai")
        
        # Initialize OpenAI client with API key
        self.client = openai.OpenAI(api_key=api_key)
        self.use_existing = use_existing
        self.model_manager = ModelManager()
    
    def _get_training_data_info(self, train_items: List[Item], validation_items: List[Item] = None) -> Dict:
        """
        Generate training data information for model matching.
        
        Args:
            train_items: Training data items
            validation_items: Validation data items
            
        Returns:
            Dictionary with training data characteristics
        """
        # Create a summary of the training data for matching purposes
        info = {
            "train_count": len(train_items),
            "validation_count": len(validation_items) if validation_items else 0,
            "base_model": self.model_name,
        }
        
        # Add a sample of training data for better matching
        if train_items:
            # Use first few items to create a signature
            sample_items = train_items[:5]
            sample_signatures = []
            for item in sample_items:
                sig = f"{item.title[:50]}_{item.price}_{len(item.features) if item.features else 0}"
                sample_signatures.append(sig)
            info["sample_signature"] = "_".join(sample_signatures)
        
        return info
    
    def check_existing_model(self, train_items: List[Item], validation_items: List[Item] = None) -> Optional[str]:
        """
        Check if an existing fine-tuned model matches the training data.
        
        Args:
            train_items: Training data items
            validation_items: Validation data items
            
        Returns:
            Existing model ID if found, None otherwise
        """
        if not self.use_existing:
            return None
        
        training_data_info = self._get_training_data_info(train_items, validation_items)
        
        return self.model_manager.find_existing_model(
            provider="openai",
            base_model=self.model_name,
            training_data_info=training_data_info
        )
    
    def upload_training_files(self, train_items: List[Item], validation_items: List[Item] = None) -> Dict[str, str]:
        """
        Upload training and validation files to OpenAI.
        
        Args:
            train_items: Training data items
            validation_items: Validation data items
            
        Returns:
            Dictionary containing file IDs
        """
        # Create output directory
        output_dir = Path("output/fine_tuning")
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Write training file
        train_file_path = output_dir / "fine_tune_train.jsonl"
        self.write_jsonl(train_items, str(train_file_path))
        
        # Upload training file
        with open(train_file_path, "rb") as f:
            train_file = self.client.files.create(file=f, purpose="fine-tune")
        
        file_ids = {"training_file": train_file.id}
        print(f"Training file uploaded: {train_file.id}")
        
        # Upload validation file if provided
        if validation_items:
            validation_file_path = output_dir / "fine_tune_validation.jsonl"
            self.write_jsonl(validation_items, str(validation_file_path))
            
            with open(validation_file_path, "rb") as f:
                validation_file = self.client.files.create(file=f, purpose="fine-tune")
            
            file_ids["validation_file"] = validation_file.id
            print(f"Validation file uploaded: {validation_file.id}")
        
        return file_ids
    
    def start_fine_tuning(self, file_ids: Dict[str, str], **kwargs) -> str:
        """
        Start OpenAI fine-tuning job.
        
        Args:
            file_ids: Dictionary containing training and validation file IDs
            **kwargs: Additional fine-tuning parameters
            
        Returns:
            Job ID for the fine-tuning job
        """
        # Default parameters
        params = {
            "training_file": file_ids["training_file"],
            "model": self.model_name,
            "seed": 42,
            "hyperparameters": {"n_epochs": 1},
            "suffix": "pricer"
        }
        
        # Add validation file if available
        if "validation_file" in file_ids:
            params["validation_file"] = file_ids["validation_file"]
        
        # Override with user parameters
        params.update(kwargs)
        
        # Start fine-tuning job
        job = self.client.fine_tuning.jobs.create(**params)
        
        print(f"Fine-tuning job started: {job.id}")
        print(f"Model: {params['model']}")
        print(f"Epochs: {params['hyperparameters']['n_epochs']}")
        
        return job.id
    
    def monitor_fine_tuning(self, job_id: str, check_interval: int = 5) -> str:
        """
        Monitor OpenAI fine-tuning progress.
        
        Args:
            job_id: Fine-tuning job ID
            check_interval: Time in seconds between status checks
            
        Returns:
            Fine-tuned model ID
        """
        print(f"Monitoring job {job_id}...")
        
        while True:
            job = self.client.fine_tuning.jobs.retrieve(job_id)
            status = job.status
            
            print(f"Status: {status}")
            
            if status == "succeeded":
                model_id = job.fine_tuned_model
                print(f"✓ Fine-tuning completed successfully!")
                print(f"Fine-tuned model: {model_id}")
                return model_id
            
            elif status == "failed":
                print(f"✗ Fine-tuning failed!")
                # Get recent events for debugging
                events = self.client.fine_tuning.jobs.list_events(fine_tuning_job_id=job_id, limit=5)
                print("Recent events:")
                for event in events.data:
                    print(f"  {event.created_at}: {event.message}")
                raise Exception(f"Fine-tuning job {job_id} failed")
            
            elif status in ["cancelled", "expired"]:
                raise Exception(f"Fine-tuning job {job_id} was {status}")
            
            # Show recent events
            try:
                events = self.client.fine_tuning.jobs.list_events(fine_tuning_job_id=job_id, limit=3)
                for event in events.data:
                    if hasattr(event, 'message') and event.message:
                        print(f"  {event.message}")
            except Exception:
                pass  # Continue if events can't be retrieved
            
            print(f"Waiting {check_interval} seconds...")
            time.sleep(check_interval)
    
    def create_predictor(self, model_id: str) -> Callable:
        """
        Create OpenAI predictor function.
        
        Args:
            model_id: Fine-tuned model ID
            
        Returns:
            Predictor function
        """
        def openai_fine_tuned_predictor(item: Item) -> float:
            """
            Predict item price using fine-tuned OpenAI model.
            
            Args:
                item: Item to predict price for
                
            Returns:
                Predicted price
            """
            try:
                messages = self.messages_for_inference(item)
                
                response = self.client.chat.completions.create(
                    model=model_id,
                    messages=messages,
                    seed=42,
                    max_tokens=7,
                    temperature=0
                )
                
                reply = response.choices[0].message.content
                price = self.get_price(reply)
                
                return max(1.0, price)
                
            except Exception as e:
                print(f"Error predicting price for item: {e}")
                return 1.0
        
        openai_fine_tuned_predictor.__name__ = f"openai_finetuned_{model_id.replace(':', '_')}"
        return openai_fine_tuned_predictor
    
    def fine_tune_and_create_predictor(self, train_items: List[Item], validation_items: List[Item] = None, **kwargs) -> Callable:
        """
        Fine-tune model and create predictor with model persistence.
        
        This method overrides the base implementation to add model management capabilities:
        - Checks for existing models to avoid retraining
        - Saves model information for future reuse
        - Supports manual model specification
        
        Args:
            train_items: Training data
            validation_items: Validation data
            model_id: Specific model ID to use (skips training)
            **kwargs: Additional fine-tuning parameters
            
        Returns:
            Predictor function
        """
        # Check if a specific model ID was provided
        manual_model_id = kwargs.pop('model_id', None)
        if manual_model_id:
            print(f"Using manually specified model: {manual_model_id}")
            predictor = self.create_predictor(manual_model_id)
            predictor.optimization_info = {
                'strategy': 'OpenAI Fine-tuning (Manual)',
                'base_model': self.model_name,
                'fine_tuned_model': manual_model_id,
                'training_samples': len(train_items),
                'validation_samples': len(validation_items) if validation_items else 0
            }
            return predictor
        
        # Check for existing model
        existing_model_id = self.check_existing_model(train_items, validation_items)
        if existing_model_id:
            print(f"✓ Reusing existing model: {existing_model_id}")
            predictor = self.create_predictor(existing_model_id)
            predictor.optimization_info = {
                'strategy': 'OpenAI Fine-tuning (Reused)',
                'base_model': self.model_name,
                'fine_tuned_model': existing_model_id,
                'training_samples': len(train_items),
                'validation_samples': len(validation_items) if validation_items else 0
            }
            return predictor
        
        # No existing model found, proceed with fine-tuning
        print("No existing model found. Starting new fine-tuning...")
        
        # Use the parent implementation for actual fine-tuning
        predictor = super().fine_tune_and_create_predictor(train_items, validation_items, **kwargs)
        
        # Save the new model information
        model_id = self.fine_tuned_model_id
        if model_id:
            training_data_info = self._get_training_data_info(train_items, validation_items)
            
            self.model_manager.save_model(
                model_id=model_id,
                provider="openai",
                base_model=self.model_name,
                training_data_info=training_data_info,
                training_params=kwargs,
                validation_metrics={}  # Could be populated with actual metrics
            )
        
        return predictor
