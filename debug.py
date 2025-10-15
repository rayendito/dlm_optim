import torch, wandb, re, argparse
from architecture.transformers_diffusion import DiffusionTransformerConfig, DiffusionTransformerLM
from train_hyperparams import TRAIN_SPLIT_SIZE, CORPUS_PATH, SEQ_LEN, EMBEDDING_SIZE, ATTN_HEAD_COUNT, LAYER_NUM, BATCH_SIZE, TOTAL_STEPS, VAL_STEPS, VAL_INTERVAL, CHECKPOINT_INTERVAL
from utils.modeling_utils import get_corpus, get_vocab_dict, get_train_val_split, get_batch, get_batch_sequential, load_checkpoint
from train_diffusion import mask_tokens_batch

TRAIN_SPLIT_SIZE=0.99
CORPUS_PATH="data/animesubs.txt"
SEQ_LEN=16
EMBEDDING_SIZE=64
ATTN_HEAD_COUNT = 4
LAYER_NUM = 6
BATCH_SIZE = 4

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
itoc, ctoi = get_vocab_dict("vocab/vocab_withmask.json")
VOCAB_SIZE = len(itoc)
encode = lambda s: [ctoi[ch] for ch in s]
decode = lambda l: ''.join([itoc[i] for i in l])

def print_ids_to_tokens(xb):
    print([decode(x) for x in xb.tolist()])

model_config = DiffusionTransformerConfig(
    vocab_size=VOCAB_SIZE,
    seq_len=SEQ_LEN,
    embed_size=EMBEDDING_SIZE,
    head_num=ATTN_HEAD_COUNT,
    layer_num=LAYER_NUM
)
model = DiffusionTransformerLM(model_config).to(device)

corpus = get_corpus(CORPUS_PATH)
corpus_tokenized = torch.tensor(encode(corpus), dtype=torch.int64)

train_data, val_data = get_train_val_split(corpus_tokenized, TRAIN_SPLIT_SIZE)


xb, yb = get_batch_sequential(train_data, seq_len=SEQ_LEN, batch_size=BATCH_SIZE, device=device, start_index=0)
print_ids_to_tokens(xb)

noisy_batch, masked_indices, p_mask = mask_tokens_batch(xb)
# logits, loss = model(noisy_batch, targets=xb, masked_indices=masked_indices, p_mask=p_mask)

breakpoint()
