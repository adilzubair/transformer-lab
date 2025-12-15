"""
Example: Train a simple bigram language model on Shakespeare text.

This is the simplest language model that predicts the next token based only 
on the current token using an embedding table lookup.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

# Import and run the bigram model
# To use this example:
# 1. Make sure you have input.txt in the root directory
# 2. Run: python examples/train_bigram.py

from models.bigram import bigram

if __name__ == "__main__":
    print("Training simple bigram model...")
    print("Make sure 'input.txt' exists in the root directory.")
