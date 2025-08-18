"""
Data loading utilities for the ValueVision project.

This module contains the ItemLoader class and related data loading functions.
"""

from datetime import datetime
from tqdm import tqdm
from datasets import load_dataset
from concurrent.futures import ProcessPoolExecutor
import multiprocessing

from config.settings import CHUNK_SIZE, MIN_PRICE, MAX_PRICE, HF_DATASET_NAME
from src.data.models import Item

if __name__ == '__main__':
    multiprocessing.freeze_support()


def process_datapoint(datapoint):
    """
    Global function to process a single datapoint.
    This needs to be at module level for multiprocessing to work properly.
    """
    try:
        price_str = datapoint['price']
        if price_str:
            price = float(price_str)
            if MIN_PRICE <= price <= MAX_PRICE:
                if Item is not None:
                    item = Item(datapoint, price)
                    return item if item.include else None
                else:
                    return datapoint
    except (ValueError, KeyError):
        return None


def process_chunk(chunk):
    """
    Global function to process a chunk of datapoints.
    This needs to be at module level for multiprocessing to work properly.
    """
    batch = []
    for datapoint in chunk:
        result = process_datapoint(datapoint)
        if result:
            batch.append(result)
    return batch


class ItemLoader:
    """
    Loads and processes Amazon product data from HuggingFace datasets.
    
    This class handles downloading datasets, processing them in parallel,
    and converting raw data into Item objects.
    """
    
    def __init__(self, name):
        """
        Initialize the ItemLoader with a dataset name.
        
        Args:
            name: The name of the Amazon product category to load
        """
        self.name = name
        self.dataset = None

    def from_datapoint(self, datapoint):
        """
        Try to create an Item from this datapoint
        Return the Item if successful, or None if it shouldn't be included
        """
        return process_datapoint(datapoint)

    def from_chunk(self, chunk):
        """
        Create a list of Items from this chunk of elements from the Dataset
        """
        return process_chunk(chunk)

    def chunk_generator(self):
        """
        Iterate over the Dataset, yielding chunks of datapoints at a time
        """
        size = len(self.dataset)
        for i in range(0, size, CHUNK_SIZE):
            chunk_data = []
            for j in range(i, min(i + CHUNK_SIZE, size)):
                chunk_data.append(self.dataset[j])
            yield chunk_data

    def load_in_parallel(self, workers):
        """
        Use concurrent.futures to farm out the work to process chunks of datapoints -
        This speeds up processing significantly, but will tie up your computer while it's doing so!
        """
        results = []
        chunk_count = (len(self.dataset) // CHUNK_SIZE) + 1
        
        try:
            with ProcessPoolExecutor(max_workers=workers) as pool:
                for batch in tqdm(pool.map(process_chunk, self.chunk_generator()), total=chunk_count):
                    results.extend(batch)
        except Exception as e:
            print(f"Error in parallel processing: {e}")
            print("Falling back to sequential processing...")
            # Fallback to sequential processing
            for chunk in tqdm(self.chunk_generator(), total=chunk_count):
                batch = process_chunk(chunk)
                results.extend(batch)
        
        for result in results:
            if hasattr(result, 'category'):
                result.category = self.name
        return results
            
    def load(self, workers=8):
        """
        Load in this dataset; the workers parameter specifies how many processes
        should work on loading and scrubbing the data
        
        Args:
            workers: Number of parallel workers to use for processing
            
        Returns:
            List of processed Item objects
        """
        start = datetime.now()
        print(f"Loading dataset {self.name}", flush=True)
        self.dataset = load_dataset(HF_DATASET_NAME, f"raw_meta_{self.name}", split="full", trust_remote_code=True)
        results = self.load_in_parallel(workers)
        finish = datetime.now()
        print(f"Completed {self.name} with {len(results):,} datapoints in {(finish-start).total_seconds()/60:.1f} mins", flush=True)
        return results


if __name__ == '__main__':
    multiprocessing.freeze_support()
