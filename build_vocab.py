import json, argparse

MASK_TOKEN = "\U0001F0A0"

parser = argparse.ArgumentParser()
parser.add_argument("--dataset_file", required=True, help="dataset file path")
args = parser.parse_args()

with open(args.dataset_file, 'r', encoding='latin') as f:
    fulltext = f.read()

chars = sorted(list(set(fulltext)))

if(MASK_TOKEN in chars):
    raise ValueError("MASK_TOKEN is in chars!")
chars = [MASK_TOKEN] + chars
vocab_lookuptable = {i:ch for i,ch in enumerate(chars)}

with open("vocab.json", "w+") as vocabfile:
    vocabfile.write(json.dumps(vocab_lookuptable, indent=4))