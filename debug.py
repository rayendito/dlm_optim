import torch, wandb, re, argparse
from architecture.transformers_diffusion import DiffusionTransformerConfig, DiffusionTransformerLM
from train_hyperparams import TRAIN_SPLIT_SIZE, CORPUS_PATH, SEQ_LEN, EMBEDDING_SIZE, ATTN_HEAD_COUNT, LAYER_NUM, BATCH_SIZE, TOTAL_STEPS, VAL_STEPS, VAL_INTERVAL, CHECKPOINT_INTERVAL
from utils.modeling_utils import get_corpus, get_vocab_dict, get_train_val_split, get_batch, get_batch_sequential, load_checkpoint


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def mask_tokens_batch(input_ids, eps: float=1e-3):
    b, l = input_ids.shape # batch size, length
    t = torch.rand(b, device=input_ids.device)
    p_mask = (1 - eps) * t + eps # making sure it's not 0. add with some eps
    p_mask = p_mask[:, None].repeat(1, l) # multiplying it by the dimension

    masked_indices = torch.rand((b, l), device=input_ids.device) < p_mask # masked_indices: bool^{b x l}
    mask_token_idx = ctoi["\U0001F0A0"]
    noisy_batch = torch.where(masked_indices, mask_token_idx, input_ids) # noisy_batch: token_idx^{b x l}
    return noisy_batch, masked_indices, p_mask

itoc, ctoi = get_vocab_dict("vocab/vocab_withmask.json")

encode = lambda s: [ctoi[ch] for ch in s]
decode = lambda l: ''.join([itoc[i] for i in l])

VOCAB_SIZE = len(itoc)
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

noisy_batch, masked_indices, p_mask = mask_tokens_batch(xb)
logits, loss = model(noisy_batch, targets=xb, masked_indices=masked_indices, p_mask=p_mask)
