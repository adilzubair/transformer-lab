# 🚀 Transformer Lab

A comprehensive implementation of language models from scratch, progressing from simple bigram models to full GPT-style transformers. 

## 📚 Table of Contents

- [Overview](#overview)
- [Features](#features)
- [Project Structure](#project-structure)
- [Installation](#installation)
- [Quick Start](#quick-start)
- [Models](#models)
- [Dataset Preparation](#dataset-preparation)
- [Training](#training)
- [Model Architecture Details](#model-architecture-details)
- [Hyperparameters](#hyperparameters)
- [Contributing](#contributing)
- [License](#license)

## 🎯 Overview

This repository implements three progressively complex language models:

1. **Simple Bigram Model** - Predicts next token based solely on current token
2. **Neural Bigram Network** - Adds embeddings and feedforward layers
3. **GPT Transformer** - Full transformer with multi-head attention and positional embeddings

All models are implemented from scratch using PyTorch, making them perfect for learning the fundamentals of language modeling and transformer architectures.

## ✨ Features

- 🔥 **Pure PyTorch Implementation** - No high-level abstractions, understand every line
- 📊 **Multiple Model Architectures** - From basic to advanced
- 🚄 **GPU Support** - Automatic CUDA detection and usage
- 📈 **Training Monitoring** - Real-time loss tracking for train/validation splits
- 🎲 **Text Generation** - Generate text samples after training
- 🗂️ **Clean Structure** - Professional project organization

## 📁 Project Structure

```
transformer-lab/
├── models/                      # Model implementations
│   ├── bigram/                  # Bigram models
│   │   ├── bigram.py            # Simple bigram language model
│   │   └── neural_bigram_network.py  # Neural bigram with embeddings
│   └── gpt/                     # GPT transformer
│       ├── gpt.py               # Full GPT implementation
│       └── tokenizer.json       # BPE tokenizer configuration
├── data/                        # Dataset utilities
│   └── dataset.py               # TinyStories dataset downloader
├── examples/                    # Example training scripts
│   ├── train_bigram.py          # Train simple bigram
│   ├── train_neural_bigram.py   # Train neural bigram
│   ├── train_gpt.py             # Train GPT model
│   └── prepare_tinystories.py   # Download TinyStories dataset
├── utils/                       # Utility functions (extensible)
├── requirements.txt             # Python dependencies
├── .gitignore                   # Git ignore rules
└── README.md                    # This file
```

## 🛠️ Installation

### Prerequisites

- Python 3.8 or higher
- CUDA-capable GPU (optional, but recommended for faster training)

### Setup

1. **Clone the repository**
   ```bash
   git clone <repository-url>
   cd transformer-lab
   ```

2. **Create a virtual environment** (recommended)
   ```bash
   python -m venv .venv
   source .venv/bin/activate  # On Windows: .venv\Scripts\activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

## 🚀 Quick Start

### 1. Prepare Your Dataset

Download the Shakespeare dataset (or use your own text):
```bash
wget https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt
```

Alternatively, use the TinyStories dataset:
```bash
python examples/prepare_tinystories.py
```

### 2. Train a Model

**Simple Bigram Model** (fastest, ~5 minutes):
```bash
python models/bigram/bigram.py
```

**Neural Bigram Network** (better quality):
```bash
python models/bigram/neural_bigram_network.py
```

**GPT Transformer** (best quality, requires more compute):
```bash
python models/gpt/gpt.py
```

### 3. See Results

Each model will:
- Display training progress with loss metrics
- Generate sample text at the end
- Show parameter count

## 🧠 Models

### 1. Simple Bigram Model

**File:** `models/bigram/bigram.py`

The simplest language model that uses an embedding table lookup to predict the next token based solely on the current token.

**Key Features:**
- Single embedding table (vocab_size × vocab_size)
- No learned context beyond immediate previous token
- Fast training (~3000 iterations)
- Good baseline for comparison

**Architecture:**
```
Input Token → Embedding Lookup → Logits → Softmax → Next Token
```

### 2. Neural Bigram Network

**File:** `models/bigram/neural_bigram_network.py`

An improved bigram model that adds neural network layers to learn richer representations.

**Key Features:**
- Character embeddings (vocab_size × n_embd)
- Feedforward neural network with ReLU activation
- Better capacity for learning patterns
- Improved text generation quality

**Architecture:**
```
Input Token → Embedding (32d) → Feedforward + ReLU → Linear Head → Logits
```

### 3. GPT Transformer

**File:** `models/gpt/gpt.py`

A full GPT-style transformer with multi-head self-attention, implementing the architecture from "Attention is All You Need".

**Key Features:**
- Multi-head self-attention (6 heads)
- Positional embeddings for sequence order
- 6 transformer blocks
- Layer normalization
- Residual connections
- Dropout for regularization
- ~10M parameters

**Architecture:**
```
Input → Token Embedding + Positional Embedding
     → [Transformer Block × 6]
     → Layer Norm
     → Linear Head
     → Logits
```

Each Transformer Block contains:
- Layer Norm → Multi-Head Attention → Residual
- Layer Norm → Feedforward Network → Residual

## 📊 Dataset Preparation

### Shakespeare Dataset
The default dataset is the Tiny Shakespeare corpus, containing ~1MB of Shakespeare's works.

```bash
wget https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt
```

### TinyStories Dataset
A larger, more diverse dataset of simple stories generated by GPT-3.5/4.

```bash
python examples/prepare_tinystories.py
```

This downloads 60,000 stories from HuggingFace's `roneneldan/TinyStories` dataset.

## 🎯 Training

### Training Process

All models follow a similar training loop:

1. **Data Loading**: Load text and create character-level tokenization
2. **Train/Val Split**: 90% training, 10% validation
3. **Batch Generation**: Random batches of sequences
4. **Training Loop**: 
   - Forward pass
   - Loss calculation (cross-entropy)
   - Backward pass
   - Parameter update (AdamW optimizer)
5. **Evaluation**: Periodic loss evaluation on both splits
6. **Generation**: Sample text generation at the end

### Monitoring Training

During training, you'll see output like:
```
10.788929 M parameters
step 0: train loss 4.2277, val loss 4.2324
step 500: train loss 1.9856, val loss 2.1523
step 1000: train loss 1.5234, val loss 1.8901
...
```

Lower loss indicates better model performance.

## ⚙️ Hyperparameters

### Simple Bigram Model
```python
batch_size = 32       # Parallel sequences
block_size = 8        # Context length
max_iters = 3000      # Training iterations
learning_rate = 1e-2  # Learning rate
```

### Neural Bigram Network
```python
batch_size = 32
block_size = 8
max_iters = 5000
learning_rate = 1e-3  # Lower for stability
n_embd = 32          # Embedding dimension
```

### GPT Transformer
```python
batch_size = 64
block_size = 256      # Longer context
max_iters = 5000
learning_rate = 3e-4
n_embd = 384         # Larger embeddings
n_head = 6           # Attention heads
n_layer = 6          # Transformer layers
dropout = 0.2        # Regularization
```

## 🔧 Model Architecture Details

### Multi-Head Attention

The GPT model uses multi-head self-attention with 6 heads:

```python
class Head(nn.Module):
    """One head of self-attention"""
    - Query, Key, Value projections
    - Scaled dot-product attention
    - Causal masking (autoregressive)
    - Dropout regularization
```

### Feedforward Network

Each transformer block includes a feedforward network:

```python
class FeedForward(nn.Module):
    - Linear layer: n_embd → 4 * n_embd
    - ReLU activation
    - Linear layer: 4 * n_embd → n_embd
    - Dropout
```

### Layer Normalization

Applied before attention and feedforward (Pre-LN architecture):

```python
x = x + self.sa(self.ln1(x))  # Attention with residual
x = x + self.ffwd(self.ln2(x))  # Feedforward with residual
```

## 🎓 Learning Resources

This implementation is inspired by:

- [Andrej Karpathy's Neural Networks: Zero to Hero series](https://karpathy.ai/zero-to-hero.html)
- ["Attention is All You Need" paper](https://arxiv.org/abs/1706.03762)
- [The Illustrated Transformer](http://jalammar.github.io/illustrated-transformer/)






