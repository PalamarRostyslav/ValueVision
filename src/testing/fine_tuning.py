"""
Fine-tuning framework for model optimization and improvement.

This module provides abstract base classes and implementations for various
fine-tuning strategies that can be applied to price prediction models.
"""

from abc import ABC, abstractmethod
from src.testing.tester import ModelTester
from typing import List, Callable, Dict, Optional
import random
from src.data.models import Item


class FineTuning(ABC):
    """
    Abstract base class for fine-tuning strategies.
    
    This class defines the interface that all fine-tuning implementations
    must follow, ensuring consistency and extensibility.
    """
    
    def __init__(self, name: str = None):
        """
        Initialize the fine-tuning strategy.
        
        Args:
            name: Optional name for the strategy (defaults to class name)
        """
        self.name = name or self.__class__.__name__
        self.metrics_history: List[Dict] = []
        self.best_score: Optional[float] = None
        self.best_params: Optional[Dict] = None
    
    @abstractmethod
    def optimize_predictor(self, training_data: List[Item], validation_data: List[Item] = None) -> Callable:
        """
        Optimize and return an optimized predictor.
        
        Args:
            training_data: Data to use for training/optimization
            validation_data: Data to use for validation/evaluation (optional)
            
        Returns:
            Optimized predictor function
        """
        pass
    
    def evaluate_predictor(self, predictor: Callable, data: List[Item]) -> Dict[str, float]:
        """
        Evaluate a predictor on given data.
        
        Args:
            predictor: Function to evaluate
            data: Data to evaluate on
            
        Returns:
            Dictionary containing evaluation metrics
        """
        
        # Use a subset for faster evaluation during optimization
        eval_size = min(100, len(data))
        subset_data = data[:eval_size]
        
        tester = ModelTester(predictor, subset_data, size=eval_size)
        
        # Run tests silently
        tester.guesses.clear()
        tester.truths.clear()
        tester.errors.clear()
        tester.sles.clear()
        tester.colors.clear()
        
        for i in range(eval_size):
            guess, truth, error, sle, color = tester.test_single_item(i)
            tester.guesses.append(guess)
            tester.truths.append(truth)
            tester.errors.append(error)
            tester.sles.append(sle)
            tester.colors.append(color)
        
        return tester.calculate_metrics()
    
    def log_metrics(self, metrics: Dict[str, float], params: Dict = None) -> None:
        """
        Log metrics from an optimization iteration.
        
        Args:
            metrics: Performance metrics
            params: Parameters used for this iteration
        """
        log_entry = {
            'metrics': metrics.copy(),
            'params': params.copy() if params else {},
            'iteration': len(self.metrics_history) + 1
        }
        self.metrics_history.append(log_entry)
        
        # Update best score (using RMSLE as primary metric)
        current_score = metrics.get('rmsle', float('inf'))
        if self.best_score is None or current_score < self.best_score:
            self.best_score = current_score
            self.best_params = params.copy() if params else {}
    
    def get_optimization_summary(self) -> Dict:
        """
        Get a summary of the optimization process.
        
        Returns:
            Dictionary containing optimization summary
        """
        if not self.metrics_history:
            return {}
        
        return {
            'strategy_name': self.name,
            'total_iterations': len(self.metrics_history),
            'best_score': self.best_score,
            'best_params': self.best_params,
            'improvement': self.metrics_history[0]['metrics'].get('rmsle', 0) - self.best_score,
            'final_metrics': self.metrics_history[-1]['metrics']
        }

class FineTuningRandomSeed(FineTuning):
    """
    Fine-tuning strategy that optimizes random seed parameters.
    
    This implementation tests different random seeds to find the best
    performing configuration for stochastic predictors.
    """
    
    def __init__(self, seed_range: tuple = (1, 1000), iterations: int = 50):
        """
        Initialize random seed fine-tuning.
        
        Args:
            seed_range: Tuple of (min_seed, max_seed) for random seed testing
            iterations: Number of different seeds to test
        """
        super().__init__("Random Seed Optimization")
        self.seed_range = seed_range
        self.iterations = iterations
    
    def base_predictor(self, item, random_seed=42):
        """
        Base predictor with random seed parameter for fine-tuning.
        
        Args:
            item: Item to predict price for
            random_seed: Random seed for reproducible randomness
            
        Returns:
            Predicted price
        """
        random.seed(random_seed)
        
        base_price = len(item.title) * 2.5
        if hasattr(item, 'features') and item.features:
            feature_boost = len(item.features) * 10
            base_price += feature_boost
        
        # Add some randomness
        noise = random.uniform(-base_price * 0.1, base_price * 0.1)
        return max(base_price + noise, 10.0)
    
    def optimize_predictor(self, training_data: List[Item], validation_data: List[Item] = None) -> Callable:
        """
        Optimize predictor by testing different random seeds.
        
        Args:
            training_data: Training data (used for optimization)
            validation_data: Validation data (used for final evaluation of best seed)
            
        Returns:
            Optimized predictor with best seed fixed
        """
        print(f"\nStarting {self.name}...")
        print(f"Testing {self.iterations} different seeds from {self.seed_range[0]} to {self.seed_range[1]}")
        print(f"Using {len(training_data)} training samples for optimization")
        
        best_seed = None
        best_score = float('inf')
        
        for i in range(self.iterations):
            test_seed = random.randint(self.seed_range[0], self.seed_range[1])
            
            total_error = 0
            test_size = min(50, len(training_data))
            
            for j in range(test_size):
                item = training_data[j]
                guess = self.base_predictor(item, random_seed=test_seed)
                truth = item.price
                error = abs(guess - truth)
                total_error += error
            
            avg_error = total_error / test_size
            
            # Update best if this is better
            if avg_error < best_score:
                best_score = avg_error
                best_seed = test_seed
            
            if (i + 1) % 5 == 0:
                print(f"  Tested {i + 1}/{self.iterations} seeds. Best so far: {best_seed} (${best_score:.2f})")
        
        print(f"\nOptimization complete!")
        print(f"Best seed: {best_seed} with training error: ${best_score:.2f}")
        
        # If validation data is provided, evaluate the best seed on it
        if validation_data:
            print("\nEvaluating best seed on validation data...")
            validation_error = 0
            val_size = min(100, len(validation_data))
            
            for j in range(val_size):
                item = validation_data[j]
                guess = self.base_predictor(item, random_seed=best_seed)
                truth = item.price
                error = abs(guess - truth)
                validation_error += error
            
            avg_validation_error = validation_error / val_size
            print(f"Validation error: ${avg_validation_error:.2f}")
        
        # Return optimized predictor with best seed
        def optimized_predictor(item):
            return self.base_predictor(item, random_seed=best_seed)
        
        optimized_predictor.__name__ = f"optimized_predictor_seed_{best_seed}"
        
        # Store optimization info as attributes on the predictor
        optimized_predictor.optimization_info = {
            'best_seed': best_seed,
            'training_error': best_score,
            'validation_error': avg_validation_error if validation_data else None,
            'iterations_tested': self.iterations,
            'strategy': self.name
        }
        
        return optimized_predictor


