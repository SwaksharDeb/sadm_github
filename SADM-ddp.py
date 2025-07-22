from DDPM import DDPM, ContextUnet
from ViVit import ViViT
from ACDC_loader import ACDCDataset
import torch, os
from tqdm import tqdm
from torch.utils.data import DataLoader
import numpy as np

import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler
from torch.amp import autocast, GradScaler



DATA_DIR = "data"
RESULT_DIR = "results"
os.makedirs(RESULT_DIR, exist_ok=True)

assert os.path.isdir(DATA_DIR), f"{DATA_DIR} is not a directory."
assert os.path.isdir(RESULT_DIR), f"{RESULT_DIR} is not a directory."

def setup_ddp():
    dist.init_process_group(backend="nccl")
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    return local_rank

def cleanup_ddp():
    dist.destroy_process_group()
    
def train():
    # device = torch.device("cuda")
    
    local_rank = setup_ddp()
    device = torch.device(f"cuda:{local_rank}")
    
    print('check', local_rank)
    print()
    
    scaler = GradScaler('cuda')
    
    n_epoch = 300
    batch_size = 2 #* torch.cuda.device_count()
    image_size = (128, 128, 128)
    num_frames = 3

    # DDPM hyperparameters
    n_T = 500  # 500
    n_feat = 128  # 128 ok, 256 better (but slower)
    lrate = 1e-4

    # ViViT hyperparameters
    patch_size = (32, 32, 32)
    print('check', local_rank)
    print()
    dataset = ACDCDataset(data_dir="data", split="train")
    train_sampler = DistributedSampler(dataset)
    train_loader = DataLoader(dataset, batch_size=batch_size, sampler=train_sampler, num_workers=1, pin_memory=True)
    print('check', local_rank)
    print()
    # train_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=1)
    
    if local_rank == 0:
        valid_loader = DataLoader(ACDCDataset(data_dir="data", split="test"), batch_size=batch_size, shuffle=False, num_workers=1)
        x_val, x_prev_val, file_val = next(iter(valid_loader))
        x_prev_val = x_prev_val.to(device)
    print('check', local_rank)
    print()

    # valid_loader = DataLoader(ACDCDataset(data_dir="data", split="test"), batch_size=batch_size, shuffle=False, num_workers=1)
    # x_val, x_prev_val, file_val = next(iter(valid_loader))
    # x_prev_val = x_prev_val.to(device)
    
    nn_model = ContextUnet(in_channels=1, n_feat=n_feat, in_shape=(1, *image_size))#.to(device)
    print('check', local_rank)
    print()

    vivit_model = ViViT(image_size, patch_size, num_frames)#.to(device)
    print('check', local_rank)
    print()
    
    
    # vivit_model = torch.nn.DataParallel(vivit_model)
    # nn_model = torch.nn.DataParallel(nn_model)
    
    # vivit_model = DDP(vivit_model, device_ids=[local_rank])
    # nn_model = DDP(nn_model, device_ids=[local_rank])

    ddpm = DDPM(vivit_model=vivit_model, nn_model=nn_model,
                betas=(1e-4, 0.02), n_T=n_T, device=device, drop_prob=0.1)
    
    ddpm.to(device)
    
    ddpm = DDP(ddpm, device_ids=[local_rank])
    print('check', local_rank)
    print()

    optim = torch.optim.Adam(ddpm.parameters(), lr=lrate)

    for ep in range(n_epoch):
        train_sampler.set_epoch(ep)

        print(f'epoch {ep}')
        ddpm.train()
                
        if local_rank == 0:
            print(f"[GPU {local_rank}] Epoch {ep}")
        print('check', local_rank)
        print()

        # linear lrate decay
        optim.param_groups[0]['lr'] = lrate*(1-ep/n_epoch)
        
        # pbar = tqdm(train_loader, disable=(local_rank != 0))
        pbar = iter(train_loader)
        loss_ema = None
        print('check', local_rank)
        print()
        for _ in range(len(pbar)):
            print('check', local_rank)
            print()
            x, x_prev, file_tr = next(pbar)
            x = x.to(device, non_blocking=True)
            x_prev = x_prev.to(device, non_blocking=True)
            optim.zero_grad()
            print('check', local_rank)
            print()
            # loss = ddpm(x, x_prev)
            with autocast():
                loss = ddpm(x, x_prev)

            scaler.scale(loss).backward()
            scaler.step(optim)
            scaler.update()
            print('check', local_rank)
            print()
            # loss.backward()
            # optim.step()
            if loss_ema is None:
                loss_ema = loss.item()
            else:
                loss_ema = 0.95 * loss_ema + 0.05 * loss.item()
            if local_rank == 0:
                pbar.set_description(f"loss: {loss_ema:.4f}")
            print('check', local_rank)
            print()

        # Only rank 0 does validation/saving
        if local_rank == 0:
            ddpm.eval()
            with torch.no_grad():
                x_gen, x_gen_store = ddpm.sample(x_prev_val, device, guide_w=0.2)
                np.save(f"{RESULT_DIR}/x_gen_{ep}_{file_val}.npy", x_gen)
                np.save(f"{RESULT_DIR}/x_gen_store_{ep}_{file_val}.npy", x_gen_store)
                
                torch.save(vivit_model.state_dict(), f"{RESULT_DIR}/vivit_model{ep}.pt")
                torch.save(nn_model.state_dict(), f"{RESULT_DIR}/nn_model{ep}.pt")

    if local_rank == 0:
        torch.save(vivit_model.state_dict(), f"{RESULT_DIR}/vivit_model{ep}.pt")
        torch.save(nn_model.state_dict(), f"{RESULT_DIR}/nn_model{ep}.pt")

    cleanup_ddp()

if __name__=="__main__":
    train()