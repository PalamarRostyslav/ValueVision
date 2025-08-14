"""
Data visualization module for analyzing dataset pricing and content metrics.

This module provides a DataViewer class for analyzing and visualizing
dataset information including price distributions and content length analysis.
"""

from typing import List, Dict, Any, Tuple, Optional
from datasets import Dataset
import matplotlib.pyplot as plt

# Configure logging

class DataViewer:
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
    
    def __init__(self, hf_token: Optional[str] = None, auto_login: bool = True):
        """
        Initialize the DataViewer class
        """
        
        # Instance variables to store analysis results
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
    
    def analyze_dataset(self, dataset: Dataset) -> Dict[str, Any]:
        """
        Analyze a dataset and return statistical information.
        
        Args:
            dataset: The dataset to analyze
            
        Returns:
            Dictionary containing analysis results
        """
        prices, lengths = self._extract_dataset_metrics(dataset)
        
        # Store results in instance variables
        self._prices = prices
        self._lengths = lengths
        
        # Calculate statistics
        analysis_results = {
            'price_stats': {
                'count': len(prices),
                'average': sum(prices) / len(prices) if prices else 0,
                'max': max(prices) if prices else 0,
                'min': min(prices) if prices else 0
            },
            'length_stats': {
                'count': len(lengths),
                'average': sum(lengths) / len(lengths) if lengths else 0,
                'max': max(lengths) if lengths else 0,
                'min': min(lengths) if lengths else 0
            }
        }
        
        return analysis_results
    
    def visualize_dataset(self, dataset: Dataset, show_lengths: bool = True, 
            show_prices: bool = True, custom_bins: Dict[str, range] = None) -> None:
        """
        Create visualizations for dataset metrics.
        
        Args:
            dataset: The dataset to visualize
            show_lengths: Whether to show content length histogram
            show_prices: Whether to show price histogram
            custom_bins: Optional custom bins for histograms
        """
        analysis_results = self.analyze_dataset(dataset)
        
        bins = custom_bins or {}
        length_bins = bins.get('length', self.DEFAULT_LENGTH_BINS)
        price_bins = bins.get('price', self.DEFAULT_PRICE_BINS)
        
        if show_lengths and self._lengths:
            length_stats = analysis_results['length_stats']
            length_title = (f"Content Lengths: Avg {length_stats['average']:,.0f} "
                f"and highest {length_stats['max']:,}")
            
            self._create_histogram(
                self._lengths, 
                length_title, 
                'Length (chars)',
                self.DEFAULT_COLORS['length'],
                length_bins
            )
        
        if show_prices and self._prices:
            price_stats = analysis_results['price_stats']
            price_title = (f"Prices: Avg ${price_stats['average']:,.2f} "
                f"and highest ${price_stats['max']:,}")
            
            self._create_histogram(
                self._prices,
                price_title,
                'Price ($)',
                self.DEFAULT_COLORS['price'],
                price_bins
            )
    
    def view_detailed_dataset(self, dataset: Dataset) -> Dict[str, Any]:
        """
        Legacy method name for backward compatibility.
        
        Args:
            dataset: The dataset to analyze and visualize
            
        Returns:
            Dictionary containing analysis results
        """
        self.visualize_dataset(dataset)
        return self.analyze_dataset(dataset)
    
    @property
    def last_prices(self) -> List[float]:
        """Get the prices from the last analyzed dataset."""
        return self._prices.copy()
    
    @property  
    def last_lengths(self) -> List[int]:
        """Get the content lengths from the last analyzed dataset."""
        return self._lengths.copy()