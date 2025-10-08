import torch
import json

# DATASET MACROS
TRAIN_SPLIT_SIZE = 0.99
CORPUS_PATH = "data/animesubs.txt"

# MODELING MACROS
SEQ_LEN = 512
EMBEDDING_SIZE = 256
ATTN_HEAD_COUNT = 4
LAYER_NUM = 6

VAL_STEPS = 10
VAL_INTERVAL = 50

CHECKPOINT_INTERVAL = 500

# TRAINING MACROS
BATCH_SIZE = 256
TOTAL_STEPS = 1000


def get_corpus(dataset_path):
    with open(dataset_path, 'r', encoding='latin') as f:
        text = f.read()
    return text

def get_vocab_dict(vocab_file):
    # from the vocab.json file
    loaded_json = json.load(open(vocab_file))
    itoc = {int(i):ch for i,ch in loaded_json.items()}
    ctoi = {ch:i for i,ch in itoc.items()}
    return itoc, ctoi

def get_train_val_split(data, train_size):
    n = int(len(data) * train_size)
    return data[:n], data[n:]

def get_batch(data_split, seq_len, batch_size, device):
    ix = torch.randint(len(data_split) - seq_len, (batch_size,))
    x = torch.stack([data_split[i:i+seq_len] for i in ix])
    y = torch.stack([data_split[i+1:i+seq_len+1] for i in ix])
    return x.to(device), y.to(device)

def load_checkpoint(checkpoint_path, model, optimizer, device):
    """Load a checkpoint and restore model, optimizer, scheduler states"""
    checkpoint = torch.load(checkpoint_path)
    
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    
    print(f"Diffusion checkpoint loaded from step {checkpoint['step']}")
    return checkpoint['step'], checkpoint['train_losses'], checkpoint['val_losses']