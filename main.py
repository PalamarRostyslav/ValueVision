import os
import random
from dotenv import load_dotenv
from huggingface_hub import login
from datasets import load_dataset, Dataset, DatasetDict
import matplotlib.pyplot as plt
from collections import Counter, defaultdict
import numpy as np
import pickle
from data_viewer import DataViewer
from loaders import ItemLoader
from items import Item


def main():
    """Main function to load and process items."""
    load_dotenv(override=True)
    os.environ['OPENAI_API_KEY'] = os.getenv('OPENAI_API_KEY', 'default')
    os.environ['ANTHROPIC_API_KEY'] = os.getenv('ANTHROPIC_API_KEY', 'default')
    os.environ['HF_TOKEN'] = os.getenv('HF_TOKEN', 'default')

    hf_token = os.environ['HF_TOKEN']
    login(hf_token, add_to_git_credential=True)

    dataset_names = [
        "Appliances",
        "Automotive",
        "Electronics",
        "Office_Products",
        "Tools_and_Home_Improvement",
        "Cell_Phones_and_Accessories",
        "Toys_and_Games",
        "Appliances",
        "Musical_Instruments",
    ]
    
    items = []
    for dataset_name in dataset_names:
        loader = ItemLoader(dataset_name)
        items.extend(loader.load())

    print(f"A grand total of {len(items):,} items")
    
    dataViewer = DataViewer(hf_token=hf_token)
    for item in items:
        dataViewer.view_detailed_dataset(item)

if __name__ == '__main__':
    main()