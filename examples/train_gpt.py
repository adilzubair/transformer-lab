"""
Example: Train a GPT-style transformer model.

This model implements the full transformer architecture with:
- Multi-head self-attention
- Positional embeddings
- Feedforward layers with residual connections
- Layer normalization
- Multiple transformer blocks

This is a character-level GPT model similar to the original GPT architecture.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

# To use this example:
# 1. Make sure you have input.txt in the root directory
# 2. Run: python examples/train_gpt.py

from models.gpt import gpt

if __name__ == "__main__":
    print("Training GPT-style transformer model...")
    print("Make sure 'input.txt' exists in the root directory.")
