import json, torch
from transformers import TransformerConfig, TransformerLM

def get_corpus(dataset_path):
    with open(dataset_path, 'r', encoding='latin') as f:
        text = f.read()
    return text

def get_vocab_dict():
    # from the vocab.json file
    loaded_json = json.load(open("vocab.json"))
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
    TRAIN_SPLIT_SIZE = 0.99
    CORPUS_PATH = "data/animesubs.txt"
    SEQ_LEN = 16

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    itoc, ctoi = get_vocab_dict()
    vocab_size = len(itoc)
    encode = lambda s: [ctoi[ch] for ch in s]
    decode = lambda l: ''.join([itoc[i] for i in l])

    corpus = get_corpus(CORPUS_PATH)
    corpus = ''.join(filter(lambda character:ord(character) < 0x3000, corpus))

    corpus_tokenized = torch.tensor(encode(corpus), dtype=torch.int64)
    train_data, val_data = get_train_val_split(corpus_tokenized, TRAIN_SPLIT_SIZE)

    model_config = TransformerConfig(
        vocab_size=vocab_size,
        seq_len=SEQ_LEN,
        embed_size=256,
        head_num=4,
        layer_num=6
    )
    model = TransformerLM(model_config)
    model.to(device)
    xb, yb = get_batch(train_data, 5, 1, device)
    logits, loss = model(xb, yb)

    breakpoint()


    # def train(
    #     model,
    #     optimizer,
    #     seq_len = 256,
    #     batch_size = 256,
    #     val_steps = 10,
    #     val_interval = 50
    # ):
    #     total_steps = seq_len * batch_size
    #     losses = []
    #     val_losses = []
    #     for steps in (bar := tqdm(range(total_steps))):  # increase number of steps for good results...
    #         # sample a batch of data
    #         xb, yb = get_batch(train_data, seq_len=seq_len, batch_size=batch_size)

    #         # evaluate the loss
    #         logits, loss = model(xb, yb)

    #         # backprop
    #         optimizer.zero_grad(set_to_none=True)
    #         loss.backward()
    #         optimizer.step()

    #         bar.set_description(f"loss: {loss.item():.2f}, val loss: {val_losses[-1] if val_losses else 0:.2f}")
    #         losses.append(loss.item())
    #         if steps % val_interval == 0:
    #             # Calculate validation loss
    #             with torch.no_grad():
    #                 val_loss = 0
    #                 for _ in range(val_steps):
    #                     xb, yb = get_batch(val_data, seq_len=seq_len, batch_size=batch_size)
    #                     _, loss = model(xb, yb)
    #                     val_loss += loss.item()
    #                 val_loss /= val_steps
    #                 val_losses.append(val_loss)
    #                 print('val loss:', val_loss.item())
    #     print('final loss:', loss.item(), 'final val loss:', val_loss)


