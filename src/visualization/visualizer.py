"""
Data visualization module for analyzing dataset pricing and content metrics.

This module provides visualization classes for analyzing and visualizing
dataset information including price distributions and content length analysis.
"""

from typing import List, Tuple
from collections import Counter
from datasets import Dataset
import matplotlib.pyplot as plt

from src.data.models import Item


class DataVisualizer:
    """
    A class for analyzing and visualizing dataset metrics.
    
    This class provides methods to analyze price distributions and content
    length statistics from datasets, with configurable visualization options.
    """
    
    # Class constants for default visualization settings
    DEFAULT_FIGURE_SIZE = (15, 6)
    DEFAULT_LENGTH_BINS = range(0, 6000, 100)
    DEFAULT_PRICE_BINS = range(0, 1000, 10)
    DEFAULT_COLORS = {
        'length': 'lightblue',
        'price': 'orange'
    }
    
    def __init__(self):
        """Initialize the DataVisualizer class."""

        self._prices: List[float] = []
        self._lengths: List[int] = []
    
    def _extract_dataset_metrics(self, dataset: Dataset) -> Tuple[List[float], List[int]]:
        """
        Extract price and content length metrics from a dataset.
        
        Args:
            dataset: The dataset to analyze
            
        Returns:
            Tuple containing lists of valid prices and content lengths
        """
        prices = []
        lengths = []
        
        for datapoint in dataset:
            try:
                price = float(datapoint.get("price", 0))
                if price > 0:
                    prices.append(price)
                    
                    # Safely concatenate content fields
                    content_parts = [
                        str(datapoint.get("title", "")),
                        str(datapoint.get("description", "")),
                        str(datapoint.get("features", "")),
                        str(datapoint.get("details", ""))
                    ]
                    contents = "".join(content_parts)
                    lengths.append(len(contents))
                    
            except (ValueError, TypeError):
                continue
        
        return prices, lengths
    
    def _create_histogram(self, data: List[float], title: str, xlabel: str, 
            color: str, bins: range, figure_size: Tuple[int, int] = None) -> None:
        """
        Create and display a histogram for the given data.
        
        Args:
            data: The data to plot
            title: The title for the plot
            xlabel: The x-axis label
            color: The color for the histogram bars
            bins: The bins to use for the histogram
            figure_size: Optional custom figure size
        """
        if not data:
            return
        
        fig_size = figure_size or self.DEFAULT_FIGURE_SIZE
        
        plt.figure(figsize=fig_size)
        plt.title(title)
        plt.xlabel(xlabel)
        plt.ylabel('Count')
        plt.hist(data, rwidth=0.7, color=color, bins=bins)
        plt.show()
    
    def distribution_of_tokens(self, items: List[Item]) -> None:
        """Visualize the distribution of token lengths in the dataset."""
        tokens = [item.token_count for item in items]
        plt.figure(figsize=self.DEFAULT_FIGURE_SIZE)
        plt.title(f"Token counts: Avg {sum(tokens)/len(tokens):,.1f} and highest {max(tokens):,}")
        plt.xlabel('Length (tokens)')
        plt.ylabel('Count')
        plt.hist(tokens, rwidth=0.7, color="skyblue", bins=range(0, 300, 10))
        plt.show()
        
    def distribution_of_prices(self, items: List[Item]) -> None:
        """Visualize the distribution of prices in the dataset."""
        prices = [item.price for item in items]
        plt.figure(figsize=self.DEFAULT_FIGURE_SIZE)
        plt.title(f"Prices: Avg ${sum(prices)/len(prices):,.2f} and highest ${max(prices):,}")
        plt.xlabel('Price ($)')
        plt.ylabel('Count')
        plt.hist(prices, rwidth=0.7, color="blueviolet", bins=range(0, 1000, 10))
        plt.show()
        
    def distribution_by_category(self, items: List[Item]) -> None:
        """Visualize the distribution of items by category."""
        category_counts = Counter()
        for item in items:
            category_counts[item.category] += 1

        categories = list(category_counts.keys())
        counts = [category_counts[category] for category in categories]

        # Bar chart by category
        plt.figure(figsize=self.DEFAULT_FIGURE_SIZE)
        plt.bar(categories, counts, color="goldenrod")
        plt.title('Items by Category')
        plt.xlabel('Categories')
        plt.ylabel('Count')
        plt.xticks(rotation=45, ha='right')

        for i, v in enumerate(counts):
            plt.text(i, v, f"{v:,}", ha='center', va='bottom')

        plt.tight_layout()
        plt.show()
    
    def create_overview_plots(self, items: List[Item]) -> None:
        """Create overview visualizations for all items."""
        print("=== OVERVIEW: All Loaded Items ===")
        self.distribution_of_tokens(items)
        self.distribution_of_prices(items)
        self.distribution_by_category(items)
    
    def create_sample_plots(self, items: List[Item]) -> None:
        """Create visualizations for sampled items."""
        print("=== SAMPLE: After Sampling ===")
        self.distribution_of_prices(items)
        self.distribution_by_category(items)
    
    def create_test_plots(self, items: List[Item]) -> None:
        """Create visualizations for test subset."""
        print("=== TEST SUBSET: First 250 Test Items ===")
        self.distribution_of_prices(items)