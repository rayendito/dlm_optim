import torch, wandb, re, argparse
from tqdm import tqdm
from architecture.transformers_ar import TransformerConfig, TransformerLM
from train_hyperparams import TRAIN_SPLIT_SIZE, CORPUS_PATH, SEQ_LEN, EMBEDDING_SIZE, ATTN_HEAD_COUNT, LAYER_NUM, BATCH_SIZE, TOTAL_STEPS, VAL_STEPS, VAL_INTERVAL, CHECKPOINT_INTERVAL
from utils.modeling_utils import get_corpus, get_vocab_dict, get_train_val_split, get_batch, get_batch_sequential, load_checkpoint
from utils.wandb_utils import get_wandb_config, save_checkpoint

if __name__ == "__main__":
    MODEL_TYPE = "autoregressive"
    TRAINING_VARIATION = "default"
    EXP_NAME = f"{MODEL_TYPE}_{TRAINING_VARIATION}"
    CHECKPOINT_PATH = None
    CHECKPOINT_STEP_COUNT = 0
    if CHECKPOINT_PATH is not None:
        CHECKPOINT_STEP_COUNT = int(re.search(r'\d+', CHECKPOINT_PATH).group())
        EXP_NAME = f"{CHECKPOINT_STEP_COUNT}_" + EXP_NAME

    # DISABLING WANDB LOGS FOR DEBUGGING
    parser = argparse.ArgumentParser()
    parser.add_argument("--disable_log", action="store_true")
    DISABLE_LOG = parser.parse_args().disable_log

    # DEVICE "MACRO"
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # TOKENIZER
    itoc, ctoi = get_vocab_dict("vocab/vocab.json")
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
        data_pull_index = None
    ):
        losses = []
        val_losses = []
        for step in (bar := tqdm(range(total_steps))):  # increase number of steps for good results...
            if(data_pull_index is not None):
                xb, yb = get_batch_sequential(train_data, seq_len=seq_len, batch_size=batch_size, device=device, start_index=data_pull_index)
            else:
                xb, yb = get_batch(train_data, seq_len=seq_len, batch_size=batch_size, device=device)

            print(xb)
            print(";")
            print(yb)
            print("==========")

            # evaluate the loss
            logits, loss = model(xb, yb)

            # backprop
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()

            bar.set_description(f"loss: {loss.item():.2f}, val loss: {val_losses[-1] if val_losses else 0:.2f}")
            losses.append(loss.item())
            data_pull_index += batch_size * seq_len
            
            if not DISABLE_LOG:
                wandb.log({
                    'train_loss': loss.item(),
                    'step': step
                })

            if step % VAL_INTERVAL == 0:
                # Calculate validation loss
                with torch.no_grad():
                    val_loss = 0
                    for _ in range(VAL_STEPS):
                        xb, yb = get_batch(val_data, seq_len=seq_len, batch_size=batch_size, device=device)
                        _, loss = model(xb, yb)
                        val_loss += loss.item()
                    val_loss /= VAL_STEPS
                    val_losses.append(val_loss)
                    if not DISABLE_LOG:
                        wandb.log({
                            'val_loss': val_loss,
                            'step': step
                        })
                    print('val loss:', val_loss)
            
            if step % CHECKPOINT_INTERVAL == 0:
                # TODO(?) decouple saving checkpoint locally and to wandb
                if not DISABLE_LOG:
                    save_checkpoint(
                        model=model,
                        optimizer=optimizer,
                        step=step+CHECKPOINT_STEP_COUNT,
                        losses=losses,
                        val_losses=val_losses,
                        seq_len=seq_len,
                        batch_size=batch_size,
                        total_steps=total_steps,
                        ckpt_name=EXP_NAME,
                        data_pull_index=data_pull_index,
                    )
        print('final loss:', loss.item(), 'final val loss:', val_loss)
        # TODO(?) decouple saving checkpoint locally and to wandb
        if not DISABLE_LOG:
            save_checkpoint(
                model=model,
                optimizer=optimizer,
                step=step+CHECKPOINT_STEP_COUNT,
                losses=losses,
                val_losses=val_losses,
                seq_len=seq_len,
                batch_size=batch_size,
                total_steps=total_steps,
                data_pull_index=data_pull_index,
                ckpt_name=EXP_NAME
            )
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
    model.to(device)
    # model = torch.compile(model)
    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-3)
    data_pull_index = 0
    if CHECKPOINT_PATH is not None:
        _, _, _, data_pull_index = load_checkpoint(
            checkpoint_path=CHECKPOINT_PATH,
            model=model,
            optimizer=optimizer
        )
    
    if not DISABLE_LOG:
        wandb_config = get_wandb_config(
            model_type = MODEL_TYPE,
            variation = TRAINING_VARIATION,
            dataset = CORPUS_PATH,
            model = model,
            optimizer=optimizer,
            seq_len=SEQ_LEN,
            batch_size=BATCH_SIZE,
            total_steps=TOTAL_STEPS
        )

    SENTENCE = "Once upon a time"
    print("before ==================")
    print(test_generation(model, SENTENCE))
    
    if not DISABLE_LOG:
        wandb.init(
            entity="rayendito",
            project="dlm_optim",
            name=EXP_NAME,
            config=wandb_config,
        )
    
    losses, val_losses = train(model, optimizer, total_steps=TOTAL_STEPS, seq_len=SEQ_LEN, batch_size=BATCH_SIZE, data_pull_index=data_pull_index)
    
    if not DISABLE_LOG:
        wandb.finish()

    print("after ==================")
    print(test_generation(model, SENTENCE))
