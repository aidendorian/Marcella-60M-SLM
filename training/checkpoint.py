import torch
import os


def save_checkpoint(model, optimizer, save_dir, filename,
                    loss=None, iteration=None,
                    shard_id=0, seq_idx=0,
                    wandb_run_id=None):
    os.makedirs(save_dir, exist_ok=True)

    checkpoint = {
        'model_state_dict':model.state_dict(),
        'optimizer_state_dict':optimizer.state_dict(),
        'loss':loss,
        'iteration':iteration,
        'rng_state':torch.get_rng_state(),
        'shard_id':shard_id,
        'seq_idx':seq_idx,
        'wandb_run_id':wandb_run_id,
    }

    filepath = os.path.join(save_dir, filename)
    torch.save(checkpoint, filepath)


def load_checkpoint(model, optimizer, checkpoint_path):
    if not os.path.exists(checkpoint_path):
        raise ValueError(f'Checkpoint not found at {checkpoint_path}')

    checkpoint = torch.load(checkpoint_path, weights_only=False)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

    if 'rng_state' in checkpoint:
        torch.set_rng_state(checkpoint['rng_state'])

    return (
        checkpoint.get('loss', None),
        model,
        optimizer,
        checkpoint.get('iteration', 0),
        checkpoint.get('shard_id', 0),
        checkpoint.get('seq_idx', 0),
        checkpoint.get('wandb_run_id', None),
    )