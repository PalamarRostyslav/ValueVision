"""
Simple testing module for ValueVision price prediction models.

Contains basic testing functionality and a simple fine-tuning framework.
"""

import math
import matplotlib.pyplot as plt
from typing import List, Callable
from abc import ABC, abstractmethod
import random

# Color constants for terminal output
GREEN = "\033[92m"
YELLOW = "\033[93m"
RED = "\033[91m"
RESET = "\033[0m"
COLOR_MAP = {"red": RED, "orange": YELLOW, "green": GREEN}

class Tester:
    """Simple testing class for price prediction models."""
    
    def __init__(self, predictor: Callable, data: List, title: str = None, size: int = 250):
        self.predictor = predictor
        self.data = data
        self.title = title or predictor.__name__.replace("_", " ").title()
        self.size = min(size, len(data))
        
        # Results storage
        self.guesses = []
        self.truths = []
        self.errors = []
        self.sles = []
        self.colors = []
    
    def color_for_error(self, error: float, truth: float) -> str:
        """Determine color based on error magnitude."""
        relative_error = error / truth if truth > 0 else float('inf')
        
        if error < 40 or relative_error < 0.2:
            return "green"
        elif error < 80 or relative_error < 0.4:
            return "orange"
        else:
            return "red"
    
    def run_single_test(self, index: int) -> None:
        """Run test on a single item."""
        datapoint = self.data[index]
        guess = self.predictor(datapoint)
        truth = datapoint.price
        error = abs(guess - truth)
        
        # Calculate Squared Log Error
        log_error = math.log(truth + 1) - math.log(max(guess, 0.01) + 1)
        sle = log_error ** 2
        
        color = self.color_for_error(error, truth)
        
        # Store results
        self.guesses.append(guess)
        self.truths.append(truth)
        self.errors.append(error)
        self.sles.append(sle)
        self.colors.append(color)
        
        # Print result
        title = datapoint.title if len(datapoint.title) <= 40 else datapoint.title[:40] + "..."
        print(f"{COLOR_MAP[color]}{index + 1}: Guess: ${guess:,.2f} Truth: ${truth:,.2f} "
            f"Error: ${error:,.2f} SLE: {sle:,.2f} Item: {title}{RESET}")
    
    def create_scatter_plot(self, title: str) -> None:
        """Create scatter plot comparing predictions to ground truth."""
        plt.figure(figsize=(12, 8))
        
        max_val = max(max(self.truths), max(self.guesses))
        
        # Perfect prediction line
        plt.plot([0, max_val], [0, max_val], color='deepskyblue', lw=2, alpha=0.6, label='Perfect Prediction')
        
        # Scatter plot
        plt.scatter(self.truths, self.guesses, s=3, c=self.colors, alpha=0.7)
        
        plt.xlabel('Ground Truth ($)')
        plt.ylabel('Model Estimate ($)')
        plt.xlim(0, max_val)
        plt.ylim(0, max_val)
        plt.title(title)
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.show()
    
    def generate_report(self) -> None:
        """Generate complete test report."""
        if not self.errors:
            return
        
        # Calculate metrics
        average_error = sum(self.errors) / len(self.errors)
        rmsle = math.sqrt(sum(self.sles) / len(self.sles))
        hits = sum(1 for color in self.colors if color == "green")
        hit_rate = hits / len(self.colors) * 100
        
        # Create title with metrics
        chart_title = (f"{self.title}\n"
                      f"Avg Error: ${average_error:,.2f} | "
                      f"RMSLE: {rmsle:,.2f} | "
                      f"Hit Rate: {hit_rate:.1f}%")
        
        # Show plot
        self.create_scatter_plot(chart_title)
        
        # Print summary
        print(f"\n{GREEN}=== TEST SUMMARY: {self.title} ==={RESET}")
        print(f"Tests run: {len(self.colors)}")
        print(f"Average Error: ${average_error:,.2f}")
        print(f"RMSLE: {rmsle:,.2f}")
        print(f"Hit Rate: {hit_rate:.1f}% ({hits}/{len(self.colors)})")
    
    def run_full_test(self) -> None:
        """Run complete test suite."""
        print(f"\n{YELLOW}Starting test: {self.title}{RESET}")
        print(f"Testing {self.size} items...\n")
        
        # Clear previous results
        self.guesses.clear()
        self.truths.clear()
        self.errors.clear()
        self.sles.clear()
        self.colors.clear()
        
        # Run tests
        for i in range(self.size):
            self.run_single_test(i)
        
        # Generate report
        self.generate_report()


class FineTuning(ABC):
    """Simple abstract base class for fine-tuning strategies."""
    
    @abstractmethod
    def optimize(self, predictor: Callable, data: List) -> Callable:
        """Optimize the predictor and return the improved version."""
        pass


class FineTuningRandomSeed(FineTuning):
    """Fine-tuning by testing different random seeds."""
    
    def __init__(self, iterations: int = 20):
        self.iterations = iterations
    
    def optimize(self, predictor: Callable, data: List) -> Callable:
        """Find the best random seed for the predictor."""
        print(f"\nOptimizing random seed with {self.iterations} iterations...")
        
        best_seed = 42
        best_score = float('inf')
        
        for i in range(self.iterations):
            test_seed = random.randint(1, 1000)
            
            # Test this seed
            def seeded_predictor(item):
                return predictor(item, random_seed=test_seed)
            
            # Quick evaluation (first 50 items)
            total_error = 0
            test_size = min(50, len(data))
            
            for j in range(test_size):
                guess = seeded_predictor(data[j])
                truth = data[j].price
                error = abs(guess - truth)
                total_error += error
            
            avg_error = total_error / test_size
            
            if avg_error < best_score:
                best_score = avg_error
                best_seed = test_seed
            
            if (i + 1) % 5 == 0:
                print(f"  Tested {i + 1}/{self.iterations} seeds. Best so far: {best_seed} (${best_score:.2f})")
        
        print(f"Best seed found: {best_seed} with average error: ${best_score:.2f}")
        
        # Return optimized predictor
        def optimized_predictor(item):
            return predictor(item, random_seed=best_seed)
        
        optimized_predictor.__name__ = f"{predictor.__name__}_seed_{best_seed}"
        return optimized_predictor
