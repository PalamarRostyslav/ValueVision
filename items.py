# Temporary compatibility file for pickle loading
# This allows old pickle files to be loaded after module reorganization

from src.data.models import Item

# Make Item available for pickle compatibility
__all__ = ['Item']
