import torch
import os

def save_checkpoint(model, optimizer, save_dir, filename, loss=None, iteration=None, chunks_yielded=0):
    os.makedirs(save_dir, exist_ok=True)
    checkpoint = {
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss,
        'iteration': iteration,
        'rng_state': torch.get_rng_state(),
        'chunks_yielded': chunks_yielded,
    }
    
    filepath = os.path.join(save_dir, filename)
    torch.save(checkpoint, filepath)

def load_checkpoint(model, optimizer, checkpoint_path):
    if not os.path.exists(checkpoint_path):
        raise ValueError(f'Checkpoint not Found at {checkpoint_path}')
    
    checkpoint = torch.load(checkpoint_path)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

    if 'rng_state' in checkpoint:
        torch.set_rng_state(checkpoint['rng_state'])
    
    loss = checkpoint.get('loss', None)
    iteration = checkpoint.get('iteration', None)
    examples_consumed = checkpoint.get('examples_consumed', 0)
    chunks_yielded = checkpoint.get('chunks_yielded', 0)
    return loss, model, optimizer, iteration, chunks_yielded