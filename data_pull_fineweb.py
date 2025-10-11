from datasets import load_dataset

# Login using e.g. `huggingface-cli login` to access this dataset
ds = load_dataset("HuggingFaceFW/fineweb-edu", "sample-10BT", split="train[:50000]")
texts = ds["text"]

with open("data/fineweb_50K.txt", "w", encoding="utf-8") as f:
    f.write("\n".join(texts))