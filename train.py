import json
from transformers import TransformerConfig, TransformerLM

def get_vocab_dict():
    # from the vocab.json file
    itoc = json.load(open("vocab.json"))
    ctoi = {ch:i for i,ch in itoc.items()}
    return itoc, ctoi




if __name__ == "__main__":
    itoc, ctoi = get_vocab_dict()
    encode = lambda s: [ctoi[ch] for ch in s]
    decode = lambda l: ''.join([itoc[i] for i in l])

    