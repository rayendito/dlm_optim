import torch
import json

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

def get_batch_sequential(data_split, seq_len, batch_size, device, start_index, stride=None):
    if stride is None:
        stride = seq_len  # non-overlapping by default
    data_len = len(data_split)
    x, y = [], []
    for b in range(batch_size):
        i = (start_index + b * stride) % (data_len - seq_len)
        x.append(data_split[i:i+seq_len])
        y.append(data_split[i+1:i+seq_len+1])
    return torch.stack(x).to(device), torch.stack(y).to(device)

def load_checkpoint(checkpoint_path, model, optimizer):
    """Load a checkpoint and restore model, optimizer, scheduler states"""
    checkpoint = torch.load(checkpoint_path)
    
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    
    print(f"Diffusion checkpoint loaded from step {checkpoint['step']}")
    return checkpoint['step'], checkpoint['train_losses'], checkpoint['val_losses'], checkpoint['config']['data_pull_index']