import os, torch, wandb

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

def save_checkpoint(model, optimizer, step, losses, val_losses, seq_len, batch_size, total_steps, ckpt_name, save_dir="model_checkpoints"):
    """Save a complete checkpoint including model, optimizer, scheduler states and training metrics"""
    os.makedirs(save_dir, exist_ok=True)
    
    checkpoint = {
        'step': step,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'train_losses': losses,
        'val_losses': val_losses,
        'config': {
            'seq_len': seq_len,
            'batch_size': batch_size,
            'total_steps': total_steps,
            'vocab_size': model.token_embedding_table.num_embeddings,
            'embed_size': model.blocks[0].sa_heads.head_size * model.head_num,
            'head_num': model.head_num,
            'layer_num': model.layer_num
        }
    }
    
    checkpoint_path = os.path.join(save_dir, f'{ckpt_name}_checkpoint_step_{step}.pt')
    torch.save(checkpoint, checkpoint_path)
    wandb.save(checkpoint_path)
    print(f"Model checkpoint saved at step {step}: {checkpoint_path}")
    
    return checkpoint_path