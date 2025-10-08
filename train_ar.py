import torch
from tqdm import tqdm
from transformers_ar import TransformerConfig, TransformerLM
from modeling_utils import TRAIN_SPLIT_SIZE, CORPUS_PATH, SEQ_LEN, EMBEDDING_SIZE, ATTN_HEAD_COUNT, LAYER_NUM, BATCH_SIZE, TOTAL_STEPS, get_corpus, get_vocab_dict, get_train_val_split, get_batch

if __name__ == "__main__":
    # DEVICE "MACRO"
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # TOKENIZER
    itoc, ctoi = get_vocab_dict("vocab.json")
    VOCAB_SIZE = len(itoc)
    encode = lambda s: [ctoi[ch] for ch in s]
    decode = lambda l: ''.join([itoc[i] for i in l])

    corpus = get_corpus(CORPUS_PATH)
    corpus = ''.join(filter(lambda character:ord(character) < 0x3000, corpus))

    corpus_tokenized = torch.tensor(encode(corpus), dtype=torch.int64)
    train_data, val_data = get_train_val_split(corpus_tokenized, TRAIN_SPLIT_SIZE)

    def train(
        model,
        optimizer,
        total_steps,
        seq_len = 256,
        batch_size = 256,
        val_steps = 10,
        val_interval = 50
    ):
        losses = []
        val_losses = []
        for steps in (bar := tqdm(range(total_steps))):  # increase number of steps for good results...
            # sample a batch of data
            xb, yb = get_batch(train_data, seq_len=seq_len, batch_size=batch_size, device=device)

            # evaluate the loss
            logits, loss = model(xb, yb)

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
                        _, loss = model(xb, yb)
                        val_loss += loss.item()
                    val_loss /= val_steps
                    val_losses.append(val_loss)
                    print('val loss:', val_loss)
        print('final loss:', loss.item(), 'final val loss:', val_loss)
        return losses, val_losses
    
    def test_generation(model, sentence, max_new_tokens=50):
        idx = encode(sentence)
        return decode(
            model.generate(idx=torch.tensor([idx], dtype=torch.long).to(device), max_new_tokens=max_new_tokens, temperature=0.5, use_cache=True)[0].tolist()
        )

    # MAKING THE MODEL
    model_config = TransformerConfig(
        vocab_size=VOCAB_SIZE,
        seq_len=SEQ_LEN,
        embed_size=EMBEDDING_SIZE,
        head_num=ATTN_HEAD_COUNT,
        layer_num=LAYER_NUM
    )
    model = TransformerLM(model_config)
    # model = torch.compile(model)
    model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-3)

    SENTENCE = "Once upon a time"
    print("before ==================")
    print(test_generation(model, SENTENCE))
    losses, val_losses = train(model, optimizer, total_steps=TOTAL_STEPS, seq_len=SEQ_LEN, batch_size=BATCH_SIZE)
    print("after ==================")
    print(test_generation(model, SENTENCE))