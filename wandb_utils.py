def get_wandb_config(model_type, variation, dataset, model, optimizer, seq_len, batch_size, total_steps):
    config = {
        "model_type": model_type,
        "variation": variation,
        "dataset": dataset,
        "seq_len": seq_len,
        "batch_size": batch_size,
        "total_steps": total_steps,
        "optimizer": optimizer.__class__.__name__,
        "initial_lr": optimizer.param_groups[0]['lr'],
    }

    config.update({
        "vocab_size": model.token_embedding_table.num_embeddings,
        "embed_size": model.blocks[0].sa_heads.head_size * model.head_num,
        "head_num": model.head_num,
        "layer_num": model.layer_num,
        "total_params": sum(p.numel() for p in model.parameters() if p.requires_grad)
    })

    return config