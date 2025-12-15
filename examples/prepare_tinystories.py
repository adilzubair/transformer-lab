"""
Example: Download and prepare the TinyStories dataset.

This script downloads the TinyStories dataset from HuggingFace and saves
it to a text file for tokenizer training.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

# To use this example:
# Run: python examples/prepare_tinystories.py

from data import dataset

if __name__ == "__main__":
    print("Downloading and preparing TinyStories dataset...")
    print("This will create tinystories_train60.txt with 60,000 stories.")
