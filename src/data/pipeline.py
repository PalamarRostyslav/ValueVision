"""
Main data processing pipeline that orchestrates the entire workflow.
"""

import logging
import os 
import sys
import pickle
from typing import List, Tuple
from src.data import models

from src.utils.environment import EnvironmentManager
from src.data.sampling import DataSampler
from src.data.dataset_creator import DatasetCreator
from src.visualization.visualizer import DataVisualizer
from src.data.loaders import ItemLoader
from config.settings import DATASET_NAMES, TRAIN_SIZE, TEST_SIZE, OUTPUT_DIR, TRAIN_FILE, TEST_FILE

logger = logging.getLogger(__name__)


class DataPipeline:
    """Main pipeline for processing and preparing data."""
    
    def __init__(self, use_existing_data: bool = False):
        """
        Initialize the data pipeline.
        
        Args:
            use_existing_data: If True, try to load existing pickle files instead of processing raw data
        """
        self.use_existing_data = use_existing_data
        self.env_manager = EnvironmentManager()
        self.sampler = DataSampler()
        self.dataset_creator = DatasetCreator()
        self.visualizer = DataVisualizer()
        
        # Only authenticate with HuggingFace if we need to load new data
        if not use_existing_data:
            self.env_manager.authenticate_huggingface()
    
    def check_existing_files(self) -> bool:
        """
        Check if existing pickle files are available.
        
        Returns:
            True if both train and test pickle files exist
        """
        train_path = os.path.join(OUTPUT_DIR, TRAIN_FILE)
        test_path = os.path.join(OUTPUT_DIR, TEST_FILE)
        
        train_exists = os.path.exists(train_path)
        test_exists = os.path.exists(test_path)
        
        if train_exists and test_exists:
            logger.info(f"Found existing files: {train_path} and {test_path}")
            return True
        elif train_exists or test_exists:
            logger.warning("Only one of train/test files exists. Will need to regenerate both.")
            return False
        else:
            logger.info("No existing pickle files found.")
            return False
    
    def load_existing_datasets(self) -> Tuple[List, List]:
        """
        Load existing train and test datasets from pickle files.
        
        Returns:
            Tuple of (train_items, test_items)
            
        Raises:
            FileNotFoundError: If the pickle files don't exist
            Exception: If there's an error loading the files
        """
        train_path = os.path.join(OUTPUT_DIR, TRAIN_FILE)
        test_path = os.path.join(OUTPUT_DIR, TEST_FILE)
        
        try:
            logger.info("Loading existing training data...")
            
            if 'items' not in sys.modules:
                sys.modules['items'] = models
            
            with open(train_path, 'rb') as file:
                train = pickle.load(file)
            logger.info(f"Loaded {len(train):,} training items")
            
            logger.info("Loading existing test data...")
            with open(test_path, 'rb') as file:
                test = pickle.load(file)
            logger.info(f"Loaded {len(test):,} test items")
            
            return train, test
            
        except FileNotFoundError as e:
            logger.error(f"Pickle file not found: {e}")
            raise
        except Exception as e:
            logger.error(f"Error loading pickle files: {e}")
            raise
    
    def load_all_datasets(self, dataset_names: List[str] = None) -> List:
        """
        Load all specified datasets.
        
        Args:
            dataset_names: List of dataset names to load
            
        Returns:
            List of all loaded items
        """
        if dataset_names is None:
            dataset_names = DATASET_NAMES
        
        all_items = []
        
        for dataset_name in dataset_names:
            logger.info(f"Loading dataset: {dataset_name}")
            loader = ItemLoader(dataset_name)
            items = loader.load()
            all_items.extend(items)
            logger.info(f"Loaded {len(items):,} items from {dataset_name}")
        
        logger.info(f"Total items loaded: {len(all_items):,}")
        return all_items
    
    def process_and_sample_data(self, items: List) -> List:
        """
        Process and sample the loaded data.
        
        Args:
            items: List of all loaded items
            
        Returns:
            List of sampled items
        """
        logger.info("Creating initial visualizations...")
        self.visualizer.create_overview_plots(items)
        
        logger.info("Sampling items...")
        sample = self.sampler.sample_items(items)
        logger.info(f"Sample contains {len(sample):,} items")
        
        logger.info("Creating sample visualizations...")
        self.visualizer.create_sample_plots(sample)
        
        return sample
    
    def create_train_test_split(self, sample: List) -> Tuple[List, List]:
        """
        Create train/test split from sampled data.
        
        Args:
            sample: List of sampled items
            
        Returns:
            Tuple of (train_items, test_items)
        """
        train, test = self.sampler.shuffle_and_split(sample, TRAIN_SIZE, TEST_SIZE)
        
        logger.info(f"Train set size: {len(train):,}")
        logger.info(f"Test set size: {len(test):,}")
        
        # Show examples
        if train:
            print("Training example:")
            print(train[0].prompt)
            print()
        
        if test:
            print("Test example:")
            print(test[0].test_prompt())
            print()
        
        # Create test subset visualization
        test_subset = test[:250]
        self.visualizer.create_test_plots(test_subset)
        
        return train, test
    
    def save_datasets(self, train: List, test: List) -> None:
        """
        Save datasets in multiple formats.
        
        Args:
            train: Training items
            test: Test items
        """
        logger.info("Saving datasets...")
        
        # Save as pickle files
        self.dataset_creator.save_items_to_pickle(train, test)
        
        # Create and save HuggingFace dataset
        hf_dataset = self.dataset_creator.create_huggingface_dataset(train, test)
        self.dataset_creator.save_dataset_to_disk(hf_dataset)
        
        logger.info("All datasets saved successfully")
    
    def run_full_pipeline(self, dataset_names: List[str] = None, force_reload: bool = False) -> None:
        """
        Run the complete data processing pipeline.
        
        Args:
            dataset_names: Optional list of dataset names to process
            force_reload: If True, force reloading even if use_existing_data is True
        """
        logger.info("Starting data processing pipeline...")
        
        # Check if we should use existing data
        if self.use_existing_data and not force_reload:
            if self.check_existing_files():
                try:
                    train, test = self.load_existing_datasets()
                    
                    # Show examples
                    if train:
                        print("Training example:")
                        print(train[0].prompt)
                        print()
                    
                    if test:
                        print("Test example:")
                        print(test[0].test_prompt())
                        print()
                    
                    # Create visualizations for existing data
                    logger.info("Creating visualizations for existing data...")
                    all_items = train + test
                    self.visualizer.create_overview_plots(all_items)
                    
                    # Create subset for sample visualization
                    sample_size = min(len(all_items), 50000)  # Limit sample size for performance
                    sample_items = all_items[:sample_size]
                    self.visualizer.create_sample_plots(sample_items)
                    
                    # Test subset visualization
                    test_subset = test[:250] if len(test) > 250 else test
                    self.visualizer.create_test_plots(test_subset)
                    
                    # Optionally regenerate HuggingFace dataset
                    logger.info("Regenerating HuggingFace dataset from existing data...")
                    hf_dataset = self.dataset_creator.create_huggingface_dataset(train, test)
                    self.dataset_creator.save_dataset_to_disk(hf_dataset)
                    
                    logger.info("Pipeline completed using existing data!")
                    return
                    
                except Exception as e:
                    logger.error(f"Failed to load existing data: {e}")
                    logger.info("Falling back to full data processing...")
        
        # If we reach here, we need to process the data from scratch
        logger.info("Processing data from scratch...")
        
        # Ensure authentication if we're loading new data
        if self.use_existing_data:
            self.env_manager.authenticate_huggingface()
        
        # Load all data
        all_items = self.load_all_datasets(dataset_names)
        
        # Process and sample
        sample = self.process_and_sample_data(all_items)
        
        # Create train/test split
        train, test = self.create_train_test_split(sample)
        
        # Save datasets
        self.save_datasets(train, test)
        
        logger.info("Pipeline completed successfully!")
    
    def run_analysis_only(self) -> None:
        """
        Run only analysis and visualization on existing data.
        This is useful for quickly exploring the data without reprocessing.
        """
        logger.info("Running analysis on existing data...")
        
        if not self.check_existing_files():
            raise FileNotFoundError("No existing pickle files found. Run the full pipeline first.")
        
        try:
            train, test = self.load_existing_datasets()
            
            # Show examples
            print("=== DATA EXAMPLES ===")
            if train:
                print("Training example:")
                print(train[0].prompt)
                print()
            
            if test:
                print("Test example:")
                print(test[0].test_prompt())
                print()
            
            # Create all visualizations
            all_items = train + test
            
            print("=== VISUALIZATIONS ===")
            logger.info("Creating overview visualizations...")
            self.visualizer.create_overview_plots(all_items)
            
            # Create subset for performance
            sample_size = min(len(all_items), 50000)
            sample_items = all_items[:sample_size]
            self.visualizer.create_sample_plots(sample_items)
            
            # Test subset
            test_subset = test[:250] if len(test) > 250 else test
            self.visualizer.create_test_plots(test_subset)
            
            # Print summary statistics
            print("=== SUMMARY STATISTICS ===")
            print(f"Total items: {len(all_items):,}")
            print(f"Training items: {len(train):,}")
            print(f"Test items: {len(test):,}")
            
            # Category distribution
            from collections import Counter
            categories = Counter(item.category for item in all_items)
            print(f"Categories: {dict(categories)}")
            
            # Price statistics
            prices = [item.price for item in all_items]
            print(f"Price range: ${min(prices):.2f} - ${max(prices):.2f}")
            print(f"Average price: ${sum(prices)/len(prices):.2f}")
            
            logger.info("Analysis completed!")
            
        except Exception as e:
            logger.error(f"Failed to run analysis: {e}")
            raise
