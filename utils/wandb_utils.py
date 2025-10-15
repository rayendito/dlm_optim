import os, torch, wandb

def get_wandb_config(model_type, variation, dataset, model, optimizer, scheduler, seq_len, batch_size, total_steps):
    config = {
        "model_type": model_type,
        "variation": variation,
        "dataset": dataset,
        "seq_len": seq_len,
        "batch_size": batch_size,
        "total_steps": total_steps,
        "optimizer": optimizer.__class__.__name__,
        "scheduler": scheduler.__class__.__name__,
        "initial_lr": optimizer.param_groups[0]['lr'],
    }

    config.update({
        "vocab_size": model.token_embedding_table.num_embeddings,
        "embed_size": model.blocks[0].sa_heads.head_size * model.head_num,
        "head_num": model.head_num,
        "layer_num": model.layer_num,
        "total_params": sum(p.numel() for p in model.parameters() if p.requires_grad)
    })

    # Add scheduler-specific config
    if hasattr(scheduler, 'T_max'):
        config['scheduler_T_max'] = scheduler.T_max
    if hasattr(scheduler, 'eta_min'):
        config['scheduler_eta_min'] = scheduler.eta_min

    return config

def save_checkpoint(model, optimizer, scheduler, step, losses, val_losses, seq_len, batch_size, total_steps, ckpt_name, data_pull_index, save_dir="model_checkpoints"):
    """Save a complete checkpoint including model, optimizer, scheduler states and training metrics"""
    os.makedirs(save_dir, exist_ok=True)
    
    checkpoint = {
        'step': step,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict(),
        'train_losses': losses,
        'val_losses': val_losses,
        'config': {
            'seq_len': seq_len,
            'batch_size': batch_size,
            'total_steps': total_steps,
            'vocab_size': model.token_embedding_table.num_embeddings,
            'embed_size': model.blocks[0].sa_heads.head_size * model.head_num,
            'head_num': model.head_num,
            'layer_num': model.layer_num,
            'data_pull_index': data_pull_index,
        }
    }
    
    checkpoint_path = os.path.join(save_dir, f'{ckpt_name}_checkpoint_step_{step}.pt')
    torch.save(checkpoint, checkpoint_path)
    wandb.save(checkpoint_path)
    print(f"Model checkpoint saved at step {step}: {checkpoint_path}")
    
    return checkpoint_path