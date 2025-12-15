"""
Example: Train a neural bigram network with embeddings and feedforward layer.

This model improves on the simple bigram by:
- Using learned character embeddings (n_embd dimension)
- Adding a feedforward neural network layer
- Using better hyperparameters
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

# To use this example:
# 1. Make sure you have input.txt in the root directory
# 2. Run: python examples/train_neural_bigram.py

from models.bigram import neural_bigram_network

if __name__ == "__main__":
    print("Training neural bigram network...")
    print("Make sure 'input.txt' exists in the root directory.")
