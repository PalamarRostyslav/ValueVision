"""
AI Model Management System

This module provides functionality to persist, reuse, and manage fine-tuned AI models
to avoid unnecessary retraining and reduce costs.
"""

import json
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Any
import hashlib


class ModelManager:
    """
    Manages fine-tuned AI models with persistence and reuse capabilities.
    
    Features:
    - Save model information after training
    - Check for existing models to avoid retraining
    - List available models
    - Clean up old models
    """
    
    def __init__(self, models_dir: str = "output/models"):
        """
        Initialize model manager.
        
        Args:
            models_dir: Directory to store model information
        """
        self.models_dir = Path(models_dir)
        self.models_dir.mkdir(parents=True, exist_ok=True)
        self.models_file = self.models_dir / "models_registry.json"
        
    def _load_registry(self) -> Dict[str, Any]:
        """Load the models registry from disk."""
        if self.models_file.exists():
            try:
                with open(self.models_file, 'r') as f:
                    return json.load(f)
            except (json.JSONDecodeError, IOError):
                return {}
        return {}
    
    def _save_registry(self, registry: Dict[str, Any]) -> None:
        """Save the models registry to disk."""
        with open(self.models_file, 'w') as f:
            json.dump(registry, f, indent=2, default=str)
    
    def _generate_training_hash(self, training_data_info: Dict[str, Any]) -> str:
        """
        Generate a hash for training data characteristics.
        
        Args:
            training_data_info: Information about training data
            
        Returns:
            Hash string representing the training data
        """
        # Create a deterministic string from training data info
        info_str = json.dumps(training_data_info, sort_keys=True)
        return hashlib.md5(info_str.encode()).hexdigest()[:12]
    
    def save_model(self, 
            model_id: str, 
            provider: str,
            base_model: str,
            training_data_info: Dict[str, Any],
            training_params: Dict[str, Any] = None,
            validation_metrics: Dict[str, Any] = None) -> None:
        """
        Save information about a fine-tuned model.
        
        Args:
            model_id: The fine-tuned model ID from the provider
            provider: AI provider (e.g., 'openai', 'anthropic')
            base_model: Base model that was fine-tuned
            training_data_info: Information about training data
            training_params: Parameters used for training
            validation_metrics: Validation results
        """
        registry = self._load_registry()
        
        training_hash = self._generate_training_hash(training_data_info)
        
        model_info = {
            "model_id": model_id,
            "provider": provider,
            "base_model": base_model,
            "training_hash": training_hash,
            "training_data_info": training_data_info,
            "training_params": training_params or {},
            "validation_metrics": validation_metrics or {},
            "created_at": datetime.now().isoformat(),
            "last_used": datetime.now().isoformat()
        }
        
        # Use a composite key for uniqueness
        key = f"{provider}_{base_model}_{training_hash}"
        registry[key] = model_info
        
        self._save_registry(registry)
        print(f"✓ Model saved: {model_id} (Key: {key})")
    
    def find_existing_model(self, 
            provider: str,
            base_model: str,
            training_data_info: Dict[str, Any]) -> Optional[str]:
        """
        Find an existing model for the given training configuration.
        
        Args:
            provider: AI provider
            base_model: Base model name
            training_data_info: Training data information
            
        Returns:
            Model ID if found, None otherwise
        """
        registry = self._load_registry()
        training_hash = self._generate_training_hash(training_data_info)
        key = f"{provider}_{base_model}_{training_hash}"
        
        if key in registry:
            model_info = registry[key]
            # Update last used timestamp
            model_info["last_used"] = datetime.now().isoformat()
            self._save_registry(registry)
            
            print(f"✓ Found existing model: {model_info['model_id']}")
            print(f"  Created: {model_info['created_at']}")
            print(f"  Training samples: {training_data_info.get('train_count', 'unknown')}")
            
            return model_info["model_id"]
        
        return None
    
    def list_models(self, provider: str = None) -> List[Dict[str, Any]]:
        """
        List all saved models.
        
        Args:
            provider: Filter by provider (optional)
            
        Returns:
            List of model information dictionaries
        """
        registry = self._load_registry()
        models = list(registry.values())
        
        if provider:
            models = [m for m in models if m["provider"] == provider]
        
        # Sort by creation date (newest first)
        models.sort(key=lambda x: x["created_at"], reverse=True)
        
        return models
    
    def get_model_by_id(self, model_id: str) -> Optional[Dict[str, Any]]:
        """
        Get model information by model ID.
        
        Args:
            model_id: The model ID to search for
            
        Returns:
            Model information if found, None otherwise
        """
        registry = self._load_registry()
        
        for model_info in registry.values():
            if model_info["model_id"] == model_id:
                model_info["last_used"] = datetime.now().isoformat()
                self._save_registry(registry)
                return model_info
        
        return None
    
    def delete_model(self, model_id: str) -> bool:
        """
        Delete a model from the registry.
        
        Args:
            model_id: Model ID to delete
            
        Returns:
            True if deleted, False if not found
        """
        registry = self._load_registry()
        
        for key, model_info in list(registry.items()):
            if model_info["model_id"] == model_id:
                del registry[key]
                self._save_registry(registry)
                print(f"✓ Deleted model: {model_id}")
                return True
        
        print(f"✗ Model not found: {model_id}")
        return False
    
    def cleanup_old_models(self, days: int = 30, keep_latest: int = 3) -> int:
        """
        Clean up old models from the registry.
        
        Args:
            days: Delete models older than this many days
            keep_latest: Always keep this many latest models per provider
            
        Returns:
            Number of models deleted
        """
        registry = self._load_registry()
        
        if not registry:
            return 0
        
        from datetime import datetime, timedelta
        cutoff_date = datetime.now() - timedelta(days=days)
        
        # Group models by provider
        by_provider = {}
        for key, model_info in registry.items():
            provider = model_info["provider"]
            if provider not in by_provider:
                by_provider[provider] = []
            by_provider[provider].append((key, model_info))
        
        deleted_count = 0
        
        for provider, models in by_provider.items():
            # Sort by creation date (newest first)
            models.sort(key=lambda x: x[1]["created_at"], reverse=True)
            
            # Keep the latest N models regardless of age
            to_check = models[keep_latest:]
            
            for key, model_info in to_check:
                created_at = datetime.fromisoformat(model_info["created_at"])
                if created_at < cutoff_date:
                    del registry[key]
                    deleted_count += 1
                    print(f"✓ Cleaned up old model: {model_info['model_id']}")
        
        self._save_registry(registry)
        return deleted_count
    
    def print_models_summary(self, provider: str = None) -> None:
        """
        Print a summary of all models.
        
        Args:
            provider: Filter by provider (optional)
        """
        models = self.list_models(provider)
        
        if not models:
            print("No models found.")
            return
        
        print(f"\n{'='*80}")
        print(f"{'Model Registry Summary':<30} {'Total Models:':<20} {len(models)}")
        print(f"{'='*80}")
        
        for i, model in enumerate(models, 1):
            print(f"\n{i}. {model['model_id']}")
            print(f"   Provider: {model['provider']}")
            print(f"   Base Model: {model['base_model']}")
            print(f"   Created: {model['created_at'][:19]}")
            print(f"   Last Used: {model['last_used'][:19]}")
            print(f"   Training Samples: {model['training_data_info'].get('train_count', 'unknown')}")
            
            if model.get('validation_metrics'):
                metrics = model['validation_metrics']
                print(f"   Metrics: {metrics}")
        
        print(f"\n{'='*80}")
