import json, argparse

MASK_TOKEN = "\U0001F0A0"

parser = argparse.ArgumentParser()
parser.add_argument("--dataset_file", required=True, help="dataset file path")
parser.add_argument("--mask_vocab_size", type=int, required=True, help="number of tokens reserved for masking")
args = parser.parse_args()

with open(args.dataset_file, 'r', encoding='latin') as f:
    fulltext = f.read()

chars = sorted(list(set(fulltext)))
normal_vocab = {i:ch for i,ch in enumerate(chars)}
mask_vocab = {i:f"{MASK_TOKEN}{i}" for i in range(args.mask_vocab_size)}

with open("vocab/ddec_normal_vocab.json", "w+") as vocabfile1:
    vocabfile1.write(json.dumps(normal_vocab, indent=4))

with open("vocab/ddec_mask_vocab.json", "w+") as vocabfile2:
    vocabfile2.write(json.dumps(mask_vocab, indent=4))