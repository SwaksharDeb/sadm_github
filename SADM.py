from DDPM import DDPM, ContextUnet
from ViVit import ViViT
from ACDC_loader import ACDCDataset
import torch, os
from tqdm import tqdm
from torch.utils.data import DataLoader
import numpy as np
import glob 
import matplotlib.pyplot as plt

DATA_DIR = "data"
RESULT_DIR = "results"
sampling_visualization = 'sampling_visualization'

os.makedirs(sampling_visualization, exist_ok=True)
os.makedirs(RESULT_DIR, exist_ok=True)
max_saved_models = 6

assert os.path.isdir(DATA_DIR), f"{DATA_DIR} is not a directory."
assert os.path.isdir(RESULT_DIR), f"{RESULT_DIR} is not a directory."

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

def train():
    device = torch.device("cuda")
    n_epoch = 300
    batch_size = 2 #* torch.cuda.device_count()
    image_size = (128, 128, 128)
    num_frames = 3

    # DDPM hyperparameters
    n_T = 500  # 500
    # need A100 for 64/128
    n_feat = 32  # 128 ok, 256 better (but slower) 
    lrate = 1e-3

    # ViViT hyperparameters
    patch_size = (32, 32, 32)  #can be increased
    # patch_size = (16, 16, 16)

    dataset = ACDCDataset(data_dir="data", split="train")
    train_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=1)

    valid_loader = DataLoader(ACDCDataset(data_dir="data", split="test"), batch_size=batch_size, shuffle=False, num_workers=1)
    x_val, x_prev_val, file_val = next(iter(valid_loader))
    x_prev_val = x_prev_val.to(device)


    vivit_model = ViViT(image_size, patch_size, num_frames).to(device)
    nn_model = ContextUnet(in_channels=1, n_feat=n_feat, in_shape=(1, *image_size)).to(device)
    
    # vivit_model = torch.nn.DataParallel(vivit_model)
    # nn_model = torch.nn.DataParallel(nn_model)

    ddpm = DDPM(vivit_model=vivit_model, nn_model=nn_model,
                betas=(1e-4, 0.02), n_T=n_T, device=device, drop_prob=0.1)
    
    ddpm.to(device)

    ## load the latest checkpoint
    checkpoint, step = get_latest_checkpoint(RESULT_DIR)
    if checkpoint:
        print("Loading the checkpoint")
        vivit_model.load_state_dict(checkpoint['vivit_model'])
        nn_model.load_state_dict(checkpoint['nn_model'])
    else:
        print("Can not load the checkpoint")
        step = 0  # No checkpoint found, start fresh


    optim = torch.optim.Adam(ddpm.parameters(), lr=lrate)

    for ep in range(n_epoch):
        print(f'epoch {ep}')
        ddpm.train()

        # linear lrate decay
        #optim.param_groups[0]['lr'] = lrate*(1-ep/n_epoch)

        pbar = tqdm(train_loader)
        loss_ema = None
        for step, (x, x_prev, file) in enumerate(pbar):
            optim.zero_grad()
            x = x.to(device)
            x_prev = x_prev.to(device)
            loss = ddpm(x, x_prev, step)
            loss.backward()
            if loss_ema is None:
                loss_ema = loss.item()
            else:
                loss_ema = 0.95 * loss_ema + 0.05 * loss.item()
            pbar.set_description(f"loss: {loss_ema:.4f}")
            optim.step()

            # with torch.no_grad():
            #     x_gen, x_gen_store = ddpm.sample(x_prev_val, device, guide_w=0.2)
            #     save_combined_middle_slices(x_gen, step, sampling_visualization)
            #     print(f"Saved 64th axial, coronal, sagittal slices as PNGs at step {step}")

        ddpm.eval()
        
            
        if (ep)%2==0:
            with torch.no_grad():
                x_gen, x_gen_store = ddpm.sample(x_prev_val, device, guide_w=0.2)
                #np.save(f"{RESULT_DIR}/x_gen_{ep}_{file_val}.npy", x_gen)
                #np.save(f"{RESULT_DIR}/x_gen_store_{ep}_{file_val}.npy", x_gen_store)
                save_combined_middle_slices(x_gen, step, sampling_visualization)
                print(f"Saved 64th axial, coronal, sagittal slices as PNGs at step {step}")
            
            model_to_save = ddpm
            torch.save({'step': step,
                'vivit_model': model_to_save.vivit_model.state_dict(),
                'nn_model': model_to_save.nn_model.state_dict()
            }, os.path.join(RESULT_DIR, f"checkpoint_step_{step}.pt"))
            print(f'model is saved as checkpoint_step_{step}.pt')
            
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

            # torch.save(vivit_model.state_dict(), f"{RESULT_DIR}/vivit_model{ep}.pt")
            # torch.save(nn_model.state_dict(), f"{RESULT_DIR}/nn_model{ep}.pt")

    torch.save(vivit_model.state_dict(), f"{RESULT_DIR}/vivit_model{ep}.pt")
    torch.save(nn_model.state_dict(), f"{RESULT_DIR}/nn_model{ep}.pt")


if __name__=="__main__":
    train()