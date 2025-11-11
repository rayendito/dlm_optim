import random

# read file and remove empty lines
with open("data/fineweb_100K.txt", "r", encoding="utf-8") as f:
    lines = [line.strip() for line in f if line.strip()]

# shuffle lines
random.shuffle(lines)

# write back to a new file
with open("data/fineweb_100K_shuffled.txt", "w", encoding="utf-8") as f:
    f.write("\n".join(lines))
