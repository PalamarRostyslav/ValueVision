"""
Feature-based optimization strategy for fine-tuning.

This module contains the FineTuningWithFeatures class that optimizes
predictor performance using machine learning on extracted features.
"""

import json
from collections import Counter
from typing import List, Callable, Dict, Any
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

from ..fine_tuning import FineTuning
from src.data.models import Item


class FineTuningWithFeatures(FineTuning):
    """
    Fine-tuning strategy that uses machine learning on extracted features.
    
    This implementation extracts features from item details, trains a 
    linear regression model, and creates an optimized predictor.
    """
    
    def __init__(self, top_features: int = 40, top_brands: int = 40):
        """
        Initialize feature-based fine-tuning.
        
        Args:
            top_features: Number of top features to analyze
            top_brands: Number of top brands to analyze
        """
        super().__init__("Feature-Based Optimization")
        self.top_features = top_features
        self.top_brands = top_brands
        self.model = None
        self.feature_columns = ['weight', 'rank', 'text_length', 'is_top_electronics_brand']
        self.average_weight = None
        self.average_rank = None
        self.top_electronics_brands = ["hp", "dell", "lenovo", "samsung", "asus", "sony", "canon", "apple", "intel"]
    
    def extract_features_from_details(self, training_data: List[Item]) -> None:
        """
        Extract and parse features from item details JSON.
        
        Args:
            training_data: Training data to process
        """
        print("Extracting features from item details...")
        
        # Parse JSON features
        for item in training_data:
            try:
                if hasattr(item, 'details') and item.details:
                    item.features = json.loads(item.details)
                else:
                    item.features = {}
            except (json.JSONDecodeError, TypeError):
                item.features = {}
        
        # Analyze feature frequency
        feature_count = Counter()
        for item in training_data:
            for feature_name in item.features.keys():
                feature_count[feature_name] += 1
        
        # Show most common features
        most_common_features = feature_count.most_common(self.top_features)
        print(f"Most common features (top {self.top_features}):")
        for feature, count in most_common_features[:10]:  # Show top 10
            print(f"  {feature}: {count}")
        
        # Analyze brands
        brands = Counter()
        for item in training_data:
            brand = item.features.get("Brand")
            if brand:
                brands[brand] += 1
        
        most_common_brands = brands.most_common(self.top_brands)
        print(f"\nMost common brands (top 10):")
        for brand, count in most_common_brands[:10]:
            print(f"  {brand}: {count}")
    
    def get_weight(self, item: Item) -> float:
        """
        Extract weight from item features and convert to pounds.
        
        Args:
            item: Item to extract weight from
            
        Returns:
            Weight in pounds, or None if not available
        """
        if not hasattr(item, 'features') or not item.features:
            return None
            
        weight_str = item.features.get('Item Weight')
        if not weight_str:
            return None
        
        try:
            parts = weight_str.split(' ')
            if len(parts) < 2:
                return None
                
            amount = float(parts[0])
            unit = parts[1].lower()
            
            # Convert to pounds
            if unit == "pounds":
                return amount
            elif unit == "ounces":
                return amount / 16
            elif unit == "grams":
                return amount / 453.592
            elif unit == "milligrams":
                return amount / 453592
            elif unit == "kilograms":
                return amount / 0.453592
            elif unit == "hundredths" and len(parts) > 2 and parts[2].lower() == "pounds":
                return amount / 100
            else:
                return None
        except (ValueError, IndexError):
            return None
    
    def get_rank(self, item: Item) -> float:
        """
        Extract best sellers rank from item features.
        
        Args:
            item: Item to extract rank from
            
        Returns:
            Average rank, or None if not available
        """
        if not hasattr(item, 'features') or not item.features:
            return None
            
        rank_dict = item.features.get("Best Sellers Rank")
        if rank_dict and isinstance(rank_dict, dict):
            ranks = [v for v in rank_dict.values() if isinstance(v, (int, float))]
            if ranks:
                return sum(ranks) / len(ranks)
        return None
    
    def is_top_electronics_brand(self, item: Item) -> bool:
        """
        Check if item is from a top electronics brand.
        
        Args:
            item: Item to check
            
        Returns:
            True if from top electronics brand
        """
        if not hasattr(item, 'features') or not item.features:
            return False
            
        brand = item.features.get("Brand")
        return brand and brand.lower() in self.top_electronics_brands
    
    def get_text_length(self, item: Item) -> int:
        """
        Get total text length of item description.
        
        Args:
            item: Item to measure
            
        Returns:
            Total character count
        """
        if hasattr(item, 'test_prompt'):
            return len(item.test_prompt())
        else:
            # Fallback - combine available text fields
            text_parts = [
                getattr(item, 'title', ''),
                getattr(item, 'description', ''),
                str(getattr(item, 'features', {}))
            ]
            return len(''.join(text_parts))
    
    def calculate_defaults(self, training_data: List[Item]) -> None:
        """
        Calculate default values for missing features.
        
        Args:
            training_data: Training data to calculate defaults from
        """
        # Calculate average weight
        weights = [self.get_weight(item) for item in training_data]
        weights = [w for w in weights if w is not None]
        self.average_weight = sum(weights) / len(weights) if weights else 1.0
        
        # Calculate average rank
        ranks = [self.get_rank(item) for item in training_data]
        ranks = [r for r in ranks if r is not None]
        self.average_rank = sum(ranks) / len(ranks) if ranks else 100000.0
        
        print(f"Calculated defaults: weight={self.average_weight:.2f} lbs, rank={self.average_rank:.0f}")
    
    def get_features(self, item: Item) -> Dict[str, Any]:
        """
        Extract all features for an item.
        
        Args:
            item: Item to extract features from
            
        Returns:
            Dictionary of features
        """
        return {
            "weight": self.get_weight(item) or self.average_weight,
            "rank": self.get_rank(item) or self.average_rank,
            "text_length": self.get_text_length(item),
            "is_top_electronics_brand": 1 if self.is_top_electronics_brand(item) else 0
        }
    
    def create_dataframe(self, items: List[Item]) -> pd.DataFrame:
        """
        Convert list of items to DataFrame with features.
        
        Args:
            items: Items to convert
            
        Returns:
            DataFrame with features and prices
        """
        features_list = [self.get_features(item) for item in items]
        df = pd.DataFrame(features_list)
        df['price'] = [item.price for item in items]
        return df
    
    def optimize_predictor(self, training_data: List[Item], validation_data: List[Item] = None) -> Callable:
        """
        Optimize predictor using feature-based machine learning.
        
        Args:
            training_data: Training data for optimization
            validation_data: Validation data for evaluation
            
        Returns:
            Optimized predictor function
        """
        print(f"\nStarting {self.name}...")
        print(f"Using {len(training_data)} training samples")
        
        # Extract features from details
        self.extract_features_from_details(training_data)
        if validation_data:
            self.extract_features_from_details(validation_data)
        
        # Calculate default values
        self.calculate_defaults(training_data)
        
        # Create training DataFrame
        train_df = self.create_dataframe(training_data)
        
        # Prepare features and target
        X_train = train_df[self.feature_columns]
        y_train = train_df['price']
        
        # Train model
        print("Training linear regression model...")
        self.model = LinearRegression()
        self.model.fit(X_train, y_train)
        
        # Show feature coefficients
        print("Feature coefficients:")
        for feature, coef in zip(self.feature_columns, self.model.coef_):
            print(f"  {feature}: {coef:.2f}")
        print(f"  Intercept: {self.model.intercept_:.2f}")
        
        # Evaluate on training data
        y_train_pred = self.model.predict(X_train)
        train_mse = mean_squared_error(y_train, y_train_pred)
        train_r2 = r2_score(y_train, y_train_pred)
        
        print(f"Training MSE: {train_mse:.2f}")
        print(f"Training R²: {train_r2:.3f}")
        
        # Evaluate on validation data if provided
        validation_mse = None
        validation_r2 = None
        if validation_data:
            print("\nEvaluating on validation data...")
            val_df = self.create_dataframe(validation_data)
            X_val = val_df[self.feature_columns]
            y_val = val_df['price']
            
            y_val_pred = self.model.predict(X_val)
            validation_mse = mean_squared_error(y_val, y_val_pred)
            validation_r2 = r2_score(y_val, y_val_pred)
            
            print(f"Validation MSE: {validation_mse:.2f}")
            print(f"Validation R²: {validation_r2:.3f}")
        
        # Create optimized predictor
        def optimized_predictor(item):
            features = self.get_features(item)
            features_df = pd.DataFrame([features])
            prediction = self.model.predict(features_df)[0]
            return max(prediction, 1.0)  # Minimum price of $1
        
        optimized_predictor.__name__ = f"feature_based_predictor_r2_{train_r2:.3f}"
        
        # Store optimization info
        optimized_predictor.optimization_info = {
            'training_mse': train_mse,
            'training_r2': train_r2,
            'validation_mse': validation_mse,
            'validation_r2': validation_r2,
            'feature_columns': self.feature_columns,
            'strategy': self.name
        }
        
        return optimized_predictor
