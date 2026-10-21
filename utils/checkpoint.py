import torch
import os
import re

def load_checkpoint(unet, scheduler, vae=None, class_embedder=None, optimizer=None, checkpoint_path='checkpoints/checkpoint.pth'):
    
    print("loading checkpoint")
    # Loading on CPU also supports checkpoints originally saved on a GPU.
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    
    print("loading unet")
    unet.load_state_dict(checkpoint['unet_state_dict'])
    print("loading scheduler")
    scheduler_state = dict(checkpoint['scheduler_state_dict'])
    # Older checkpoints serialized the training timetable. Keep the timetable
    # chosen by the current DDPM/DDIM inference configuration instead.
    scheduler_state.pop('timesteps', None)
    current_scheduler_state = scheduler.state_dict()
    if 'timesteps' in current_scheduler_state:
        scheduler_state['timesteps'] = current_scheduler_state['timesteps']
    scheduler.load_state_dict(scheduler_state)
    
    if vae is not None and 'vae_state_dict' in checkpoint:
        print("loading vae")
        vae.load_state_dict(checkpoint['vae_state_dict'])
    
    if class_embedder is not None and 'class_embedder_state_dict' in checkpoint:
        print("loading class_embedder")
        class_embedder.load_state_dict(checkpoint['class_embedder_state_dict'])

    if optimizer is not None and 'optimizer_state_dict' in checkpoint:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

    return checkpoint.get('epoch')

def save_checkpoint(unet, scheduler, vae=None, class_embedder=None, optimizer=None, epoch=None, save_dir='checkpoints'):
    os.makedirs(save_dir, exist_ok=True)

    # Define checkpoint file name
    filename = 'checkpoint.pth' if epoch is None else f'checkpoint_epoch_{epoch}.pth'
    checkpoint_path = os.path.join(save_dir, filename)

    checkpoint = {
        'unet_state_dict': unet.state_dict(),
        'scheduler_state_dict': scheduler.state_dict(),
    }
    
    if vae is not None:
        checkpoint['vae_state_dict'] = vae.state_dict()
    
    if class_embedder is not None:
        checkpoint['class_embedder_state_dict'] = class_embedder.state_dict()
    
    if optimizer is not None:
        checkpoint['optimizer_state_dict'] = optimizer.state_dict()
    
    if epoch is not None:
        checkpoint['epoch'] = epoch
    
    # Save checkpoint
    torch.save(checkpoint, checkpoint_path)
    print(f"Checkpoint saved at {checkpoint_path}")
    
    # Manage checkpoint history
    manage_checkpoints(save_dir, keep_last_n=10)


def manage_checkpoints(save_dir, keep_last_n=10):
    if keep_last_n < 0:
        raise ValueError("keep_last_n must be nonnegative.")
    # Ignore unrelated files and legacy checkpoints without a numeric epoch.
    checkpoints = []
    for filename in os.listdir(save_dir):
        match = re.fullmatch(r'checkpoint_epoch_(\d+)\.pth', filename)
        if match and os.path.isfile(os.path.join(save_dir, filename)):
            checkpoints.append((int(match.group(1)), filename))
    checkpoints.sort()
    for _, filename in checkpoints[:max(0, len(checkpoints) - keep_last_n)]:
        checkpoint_path = os.path.join(save_dir, filename)
        os.remove(checkpoint_path)
        print(f"Removed old checkpoint: {checkpoint_path}")
