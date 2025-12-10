from datasets import load_dataset

# 1. Load the TinyStories dataset (streaming mode is fast and efficient)
dataset = load_dataset("roneneldan/TinyStories", split="train", streaming=True)

# 2. Extract text for training the tokenizer (take first 10,000 stories)
# We need a raw text file to train the BPE tokenizer you planned
iter_dataset = iter(dataset)
with open("tinystories_train40.txt", "w", encoding="utf-8") as f:
    for i in range(40000):
        data = next(iter_dataset)
        f.write(data['text'] + "\n") #type: ignore 

print("Data saved to tinystories_train.txt for tokenizer training.")