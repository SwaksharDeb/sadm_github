from DDPM import DDPM, ContextUnet
from ViVit import ViViT
from ACDC_loader import ACDCDataset
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, DistributedSampler
import torch.backends.cudnn as cudnn
import os
from tqdm import tqdm
import numpy as np
import warnings
import time
import glob
import matplotlib.pyplot as plt
warnings.filterwarnings('ignore')

print_freq = 1
save_freq = 100
max_saved_models = 6
sample_freq = 500

DATA_DIR = "data"
RESULT_DIR = "results"
sampling_visualization = 'sampling_visualization'
os.makedirs(RESULT_DIR, exist_ok=True)
os.makedirs(sampling_visualization, exist_ok=True)
        
assert os.path.isdir(DATA_DIR), f"{DATA_DIR} is not a directory."
assert os.path.isdir(RESULT_DIR), f"{RESULT_DIR} is not a directory."

def cycle(dl):
    """Infinite data loader cycle"""
    while True:
        for data in dl:
            yield data

def init_distributed():
    """Initialize distributed training"""
    dist.init_process_group(backend='nccl')
    local_rank = int(os.environ.get('LOCAL_RANK', '0'))
    torch.cuda.set_device(local_rank)
    return local_rank

def get_latest_checkpoint(checkpoint_dir='results', pattern='checkpoint_step_*.pt'):
    files = glob.glob(os.path.join(checkpoint_dir, pattern))
    if not files:
        print("No checkpoint files found.")
        return None, 0
    latest = max(files, key=os.path.getmtime)
    checkpoint = torch.load(latest, map_location='cuda')
    step = checkpoint.get('step', 0)
    print(f"Resuming from step: {step}")
    return checkpoint, step

def load_checkpoint(model, path):
    if os.path.exists(path):
        print(f"Loading checkpoint: {path}")
        model.load_state_dict(torch.load(path, map_location='cuda'))
    else:
        print(f"Checkpoint not found: {path}")

def save_combined_middle_slices(x_gen_tensor, step, save_dir):
    """
    Save a single .png image that combines the 64th axial, coronal, and sagittal slices.

    Args:
        x_gen_tensor (torch.Tensor): shape [B, 1, D, H, W]
        step (int): training step
        save_dir (str): where to save the image
    """
    os.makedirs(save_dir, exist_ok=True)

    # Select first sample and remove channel dimension
    vol = x_gen_tensor[0, 0].cpu().numpy()  # shape [128, 128, 128]
    vol_min = vol.min()
    vol_max = vol.max()

    # Normalize for visualization
    #vol = (vol - vol.min()) / (vol.max() - vol.min() + 1e-8)

    # Get 64th slices
    axial = vol[64, :, :]      # XY plane
    coronal = vol[:, 64, :]    # XZ plane
    sagittal = vol[:, :, 64]   # YZ plane

    # Combine into one image (horizontal layout)
    combined = np.concatenate([axial, coronal, sagittal], axis=1)

    # Save the image
    save_path = os.path.join(save_dir, f"x_gen_step_{step}_combined.png")
    plt.imsave(save_path, combined, cmap='gray', vmin=vol_min, vmax=vol_max)

def create_single_gpu_model_for_sampling(vivit_model, nn_model, n_T, device):
    """Create a single GPU DDPM model for sampling"""
    # Create new DDPM instance for sampling (not wrapped in DDP)
    sampling_ddpm = DDPM(
        vivit_model=vivit_model, 
        nn_model=nn_model,
        betas=(1e-4, 0.02), 
        n_T=n_T, 
        device=device, 
        drop_prob=0.1
    )
    return sampling_ddpm

def train():
    # Initialize distributed training
    local_rank = init_distributed()
    is_main_process = local_rank == 0
    
    # Set up CUDA
    cudnn.enabled = True
    cudnn.benchmark = True

    # Set random seeds for reproducibility
    torch.manual_seed(42)
    torch.cuda.manual_seed(42)
    
    device = torch.device("cuda")
    train_num_steps = 100000  # Reduced for testing
    
    # INCREASED BATCH SIZE - This is crucial for multi-GPU efficiency
    batch_size = 2  # Increased from 2 to 8 per GPU
    
    image_size = (128, 128, 128)
    num_frames = 3

    # DDPM hyperparameters
    n_T = 500
    n_feat = 32
    lrate = 1e-4

    # ViViT hyperparameters
    patch_size = (32, 32, 32)

    # Create datasets with MORE WORKERS
    train_dataset = ACDCDataset(data_dir="data", split="train")
    train_sampler = DistributedSampler(train_dataset, shuffle=True)
    train_loader = DataLoader(
        train_dataset, 
        batch_size=batch_size, 
        sampler=train_sampler,
        num_workers=4,  # Increased from 1 to 4
        pin_memory=True,
        persistent_workers=True,  # Keep workers alive
        prefetch_factor=2  # Prefetch more batches
    )

    # Create infinite data loader
    dl = cycle(train_loader)

    # Validation dataset (only on main process)
    if is_main_process:
        valid_dataset = ACDCDataset(data_dir="data", split="test")
        valid_loader = DataLoader(
            valid_dataset, 
            batch_size=batch_size, 
            shuffle=False, 
            num_workers=2,
            pin_memory=True
        )
        x_val, x_prev_val, file_val = next(iter(valid_loader))
        x_prev_val = x_prev_val.to(device)

    # Initialize models
    vivit_model = ViViT(image_size, patch_size, num_frames).to(device)
    nn_model = ContextUnet(in_channels=1, n_feat=n_feat, in_shape=(1, *image_size)).to(device)
    
    # Create DDPM model
    ddpm = DDPM(vivit_model=vivit_model, nn_model=nn_model,
                betas=(1e-4, 0), n_T=n_T, device=device, drop_prob=0.1)   #guidance 0.02
    ddpm.to(device)

    # Load latest checkpoint if available    
    checkpoint, step = get_latest_checkpoint(RESULT_DIR)
    if checkpoint:
        vivit_model.load_state_dict(checkpoint['vivit_model'])
        nn_model.load_state_dict(checkpoint['nn_model'])
    else:
        step = 0  # No checkpoint found, start fresh

    
    # Wrap with DDP - with optimizations
    if torch.cuda.device_count() > 1:
        if is_main_process:
            print(f"Using {torch.cuda.device_count()} GPUs")
        ddpm = DDP(ddpm, 
                  device_ids=[local_rank], 
                  output_device=local_rank,
                  find_unused_parameters=False,  # Optimization
                  broadcast_buffers=False)  # Optimization if no buffer sync needed

    # Initialize optimizer
    optim = torch.optim.Adam(ddpm.parameters(), lr=lrate)

    if is_main_process:
        print(f"Starting training for {train_num_steps} steps...")
        print(f"Dataset size: {len(train_dataset)}")
        print(f"Effective batch size: {batch_size * torch.cuda.device_count()}")

    # Training loop
    #step = 0
    local_step = step
    loss_ema = None
    
    # Timing variables
    data_time = 0
    forward_time = 0
    backward_time = 0
    step_start_time = time.time()
    global_step = local_rank + local_step * torch.distributed.get_world_size()

    while step < train_num_steps:
        # Data loading timing
        data_start = time.time()
        try:
            x, x_prev, file = next(dl)
        except:
            dl = cycle(train_loader)
            x, x_prev, file = next(dl)
        data_time += time.time() - data_start
        
        ddpm.train()
        
        # Learning rate decay
        #current_lr = lrate * (1 - step / train_num_steps)
        #optim.param_groups[0]['lr'] = current_lr
        
        # Forward pass timing
        forward_start = time.time()
        optim.zero_grad()
        x = x.to(device, non_blocking=True)  # Non-blocking transfer
        x_prev = x_prev.to(device, non_blocking=True)
        loss = ddpm(x, x_prev, step)
        forward_time += time.time() - forward_start
        
        # Backward pass timing
        backward_start = time.time()
        loss.backward()
        backward_time += time.time() - backward_start
        
        if loss_ema is None:
            loss_ema = loss.item()
        else:
            loss_ema = 0.95 * loss_ema + 0.05 * loss.item()
        
        optim.step()
        
        step += 1
        local_step += 1
        global_step = local_rank + local_step * torch.distributed.get_world_size()
        
        # REDUCED PRINTING FREQUENCY - only every 100 steps
        if global_step % print_freq == 0:
            step_time = time.time() - step_start_time
            it_per_sec = print_freq / step_time if step_time > 0 else 0
            
            print(f'Step {global_step}: Loss: {loss_ema:.6f} | '
                  f'LR: {lrate:.6f} | '
                  f'{it_per_sec:.2f} it/sec | '
                  f'Data: {data_time/step_time*100:.1f}% | '
                  f'Forward: {forward_time/step_time*100:.1f}% | '
                  f'Backward: {backward_time/step_time*100:.1f}%')
            
            # Reset timing
            step_start_time = time.time()
            data_time = forward_time = backward_time = 0

        #### Save models every 1000 steps
        if is_main_process and global_step % save_freq == 0:
            ddpm.eval()
            model_to_save = ddpm.module if torch.cuda.device_count() > 1 else ddpm
              
            torch.save({'step': step,
                'vivit_model': model_to_save.vivit_model.state_dict(),
                'nn_model': model_to_save.nn_model.state_dict()
            }, os.path.join(RESULT_DIR, f"checkpoint_step_{step}.pt"))
            if is_main_process:
                print(f"Model saved at step {step}")
            
            # Manage model history: delete oldest if more than threshold
            result_folder = RESULT_DIR
            model_files = sorted(
                [os.path.join(result_folder, f) for f in os.listdir(result_folder) if f.endswith('.pt')],
                key=os.path.getmtime  # Sort by modification time
            )

            if len(model_files) > max_saved_models:
                num_to_delete = len(model_files) - max_saved_models
                for old_model in model_files[:num_to_delete]:
                    try:
                        os.remove(old_model)
                        print(f"Deleted old model: {old_model}")
                    except Exception as e:
                        print(f"Failed to delete model {old_model}: {e}")

        ## Generate samples every 5000 steps
        if is_main_process and global_step % sample_freq == 0:
            ddpm.eval()
            print(f"Generating samples at step {step}...")
            with torch.no_grad():
                model_to_sample = ddpm.module if torch.cuda.device_count() > 1 else ddpm
                x_gen, x_gen_store = model_to_sample.sample(x_prev_val, device, guide_w=0.2)
                
                save_combined_middle_slices(x_gen, step, sampling_visualization)
                print(f"Saved 64th axial, coronal, sagittal slices as PNGs at step {step}")
                
    # Final model saving
    if is_main_process:
        model_to_save = ddpm.module if torch.cuda.device_count() > 1 else ddpm
        torch.save(model_to_save.vivit_model.state_dict(), f"{RESULT_DIR}/vivit_model_final.pt")
        torch.save(model_to_save.nn_model.state_dict(), f"{RESULT_DIR}/nn_model_final.pt")
        print("Training completed!")

if __name__ == "__main__":
    train()