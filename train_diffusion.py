import torch, wandb, re
from tqdm import tqdm
from architecture.transformers_diffusion import DiffusionTransformerConfig, DiffusionTransformerLM
from utils.modeling_utils import TRAIN_SPLIT_SIZE, CORPUS_PATH, SEQ_LEN, EMBEDDING_SIZE, ATTN_HEAD_COUNT, LAYER_NUM, BATCH_SIZE, TOTAL_STEPS, VAL_STEPS, VAL_INTERVAL, CHECKPOINT_INTERVAL, get_corpus, get_vocab_dict, get_train_val_split, get_batch, get_batch_sequential, load_checkpoint
from utils.wandb_utils import get_wandb_config, save_checkpoint

if __name__ == "__main__":
    MODEL_TYPE = "diffusion"
    TRAINING_VARIATION = "default"
    EXP_NAME = f"{MODEL_TYPE}_{TRAINING_VARIATION}"
    CHECKPOINT_PATH = "model_checkpoints/diffusion_default_checkpoint_step_999.pt"
    # CHECKPOINT_PATH = None
    CHECKPOINT_STEP_COUNT = 0
    if CHECKPOINT_PATH is not None:
        CHECKPOINT_STEP_COUNT = int(re.search(r'\d+', CHECKPOINT_PATH).group())
        EXP_NAME = f"{CHECKPOINT_STEP_COUNT}_" + EXP_NAME
    
    # DEVICE "MACRO"
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # TOKENIZER
    itoc, ctoi = get_vocab_dict("vocab/vocab_withmask.json")
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
            data_pull_index = None
    ):
        losses = []
        val_losses = []
        for step in (bar := tqdm(range(total_steps))):  # increase number of steps for good results...
            # sample a batch of data
            if(data_pull_index):
                xb, yb = get_batch_sequential(train_data, seq_len=seq_len, batch_size=batch_size, device=device, start_index=data_pull_index)
            else:
                xb, yb = get_batch(train_data, seq_len=seq_len, batch_size=batch_size, device=device)

            # regularization. truncate the input dim to 1% of the training steps
            # if torch.rand(1) < 0.01:
            #     random_length = torch.randint(1, xb.shape[1] + 1, (1,))
            #     xb = xb[:, :random_length]
            
            noisy_batch, masked_indices, p_mask = mask_tokens_batch(xb)

            # evaluate the loss using standard next-token prediction
            logits, loss = model(noisy_batch, targets=xb, masked_indices=masked_indices, p_mask=p_mask)
            # token_loss = F.cross_entropy(
            #     logits[masked_indices], xb[masked_indices], reduction='none'
            # ) / p_mask[masked_indices]
            # loss = token_loss.sum() / (xb.shape[0] * xb.shape[1])

            # backprop
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()

            wandb.log({
                'train_loss': loss.item(),
                'step': step
            })

            bar.set_description(f"loss: {loss.item():.2f}, val loss: {val_losses[-1] if val_losses else 0:.2f}")
            losses.append(loss.item())
            data_pull_index += batch_size
            if step % VAL_INTERVAL == 0:
                # Calculate validation loss
                with torch.no_grad():
                    val_loss = 0
                    for _ in range(VAL_STEPS):
                        xb, yb = get_batch(val_data, seq_len=seq_len, batch_size=batch_size, device=device)
                        noisy_batch, masked_indices, p_mask = mask_tokens_batch(xb)
                        logits, val_loss_batch = model(noisy_batch, targets=xb, masked_indices=masked_indices, p_mask=p_mask)
                        val_loss += val_loss_batch.item()
                    val_loss /= VAL_STEPS
                    val_losses.append(val_loss)
                    wandb.log({
                        'val_loss': val_loss,
                        'step': step
                    })
                    print('val loss:', val_loss)
            if step % CHECKPOINT_INTERVAL == 0:
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
        print('final loss:', loss.item(), 'final val loss:', val_loss)
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
    data_pull_index = 0
    if CHECKPOINT_PATH is not None:
        _, _, _, data_pull_index = load_checkpoint(
            checkpoint_path=CHECKPOINT_PATH,
            model=model,
            optimizer=optimizer
        )

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
    
    wandb.init(
        entity="rayendito",
        project="dlm_optim",
        name=EXP_NAME,
        config=wandb_config,
    )
    losses, val_losses,  = train(model, optimizer, total_steps=TOTAL_STEPS, seq_len=SEQ_LEN, batch_size=BATCH_SIZE, data_pull_index=data_pull_index)
    wandb.finish()

    print("after ==================")
    print(test_generation(model, SENTENCE))
