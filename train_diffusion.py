import json, torch
from tqdm import tqdm
from transformers_diffusion import DiffusionTransformerConfig, DiffusionTransformerLM
from torch.nn import functional as F

def get_corpus(dataset_path):
    with open(dataset_path, 'r', encoding='latin') as f:
        text = f.read()
    return text

def get_vocab_dict():
    # from the vocab.json file
    loaded_json = json.load(open("vocab_withmask.json"))
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

if __name__ == "__main__":
    # DEVICE "MACRO"
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # DATASET MACROS
    TRAIN_SPLIT_SIZE = 0.99
    # CORPUS_PATH = "data/animesubs.txt"
    CORPUS_PATH = "data/cfg_artif_data.txt"
    
    # MODELING MACROS
    SEQ_LEN = 512
    EMBEDDING_SIZE = 256
    ATTN_HEAD_COUNT = 4
    LAYER_NUM = 6

    # TRAINING MACROS
    BATCH_SIZE = 256
    TOTAL_STEPS = 1000

    # TOKENIZER
    itoc, ctoi = get_vocab_dict()
    VOCAB_SIZE = len(itoc)
    encode = lambda s: [ctoi[ch] for ch in s]
    decode = lambda l: ''.join([itoc[i] for i in l])

    corpus = get_corpus(CORPUS_PATH)
    corpus = ''.join(filter(lambda character:ord(character) < 0x3000, corpus))

    corpus_tokenized = torch.tensor(encode(corpus), dtype=torch.int64)
    train_data, val_data = get_train_val_split(corpus_tokenized, TRAIN_SPLIT_SIZE)

    def mask_tokens_batch(input_ids, eps: float=1e-3):
        b, l = input_ids.shape # batch size, length
        t = torch.rand(b, device=input_ids.device)
        p_mask = (1 - eps) * t + eps # making sure it's not 0. add with some eps
        p_mask = p_mask[:, None].repeat(1, l) # multiplying it by the dimension

        masked_indices = torch.rand((b, l), device=input_ids.device) < p_mask # masked_indices: bool^{b x l}
        mask_token_idx = ctoi["\U0001F0A0"]
        noisy_batch = torch.where(masked_indices, mask_token_idx, input_ids) # noisy_batch: token_idx^{b x l}
        return noisy_batch, masked_indices, p_mask
    
    def train(
            model,
            optimizer,
            total_steps,
            seq_len = 256,
            batch_size = 256,
            val_steps=10,
            val_interval=50
    ):
        losses = []
        val_losses = []
        for steps in (bar := tqdm(range(total_steps))):  # increase number of steps for good results...
            # sample a batch of data
            xb, yb = get_batch(train_data, seq_len=seq_len, batch_size=batch_size, device=device)

            # regularization. truncate the input dim to 1% of the training steps
            # if torch.rand(1) < 0.01:
            #     random_length = torch.randint(1, xb.shape[1] + 1, (1,))
            #     xb = xb[:, :random_length]
            
            noisy_batch, masked_indices, p_mask = mask_tokens_batch(xb)

            # evaluate the loss using standard next-token prediction
            logits = model(noisy_batch)[0]
            token_loss = F.cross_entropy(
                logits[masked_indices], xb[masked_indices], reduction='none'
            ) / p_mask[masked_indices]
            loss = token_loss.sum() / (xb.shape[0] * xb.shape[1])

            # backprop
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()

            bar.set_description(f"loss: {loss.item():.2f}, val loss: {val_losses[-1] if val_losses else 0:.2f}")
            losses.append(loss.item())
            if steps % val_interval == 0:
                # Calculate validation loss
                with torch.no_grad():
                    val_loss = 0
                    for _ in range(val_steps):
                        xb, yb = get_batch(val_data, seq_len=seq_len, batch_size=batch_size, device=device)
                        noisy_batch, masked_indices, p_mask = mask_tokens_batch(xb)
                        logits = model(noisy_batch)[0]
                        token_loss = F.cross_entropy(logits[masked_indices], xb[masked_indices], reduction='none') / p_mask[masked_indices]
                        val_loss_batch = token_loss.sum() / (xb.shape[0] * xb.shape[1])
                        val_loss += val_loss_batch.item()
                    val_loss /= val_steps
                    val_losses.append(val_loss)
        print('final loss:', loss.item(), 'final val loss:', val_loss)
        return losses, val_losses

    def test_generation(model, sentence):
        idx = encode(sentence)
        return decode(
            model.generate(prompt=torch.tensor([idx], dtype=torch.long).to(device), steps=256, gen_length=128, block_length=32, temperature=1, cfg_scale=0, remasking='low_confidence')[0].tolist()
        )

    # MAKING THE MODEL
    model_config = DiffusionTransformerConfig(
        vocab_size=VOCAB_SIZE,
        seq_len=SEQ_LEN,
        embed_size=EMBEDDING_SIZE,
        head_num=ATTN_HEAD_COUNT,
        layer_num=LAYER_NUM
    )
    model = DiffusionTransformerLM(model_config)
    # model = torch.compile(model)
    model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-3)
    
    SENTENCE = "Once upon a time"
    print("before ==================")
    print(test_generation(model, SENTENCE))
    
    losses, val_losses = train(model, optimizer, total_steps=TOTAL_STEPS, seq_len=SEQ_LEN, batch_size=BATCH_SIZE)
    
    print("after ==================")
    print(test_generation(model, SENTENCE))
