
import os
import numpy as np
import torch
from torch.utils.data import Dataset
import glob

class ACDCDataset(Dataset):
    def __init__(self, data_dir, split="train"):
        self.split = split
        self.data_dir = os.path.join(data_dir, split)
        self.file_list = sorted(glob.glob(os.path.join(self.data_dir, "*.npy")))
        self.files = np.load(os.path.join(data_dir, f"{split}_files.npy"))
        self.frames = None

        # Infer number of frames from first sample
        sample = np.load(self.file_list[0])
        self.frames = sample.shape[0]  # timepoints

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, idx):
        data = np.load(self.file_list[idx]).astype(np.float32)  # Shape: (T, H, W, D)

        target_idx = np.random.randint(1, self.frames)

        # Create binary mask to randomly drop frames (except first and target)
        missing_mask = [1]
        if target_idx > 1:
            if target_idx > 2:
                missing_mask += list(np.random.randint(0, 2, size=(target_idx - 2,)))
            missing_mask += [1]
        missing_mask += [0] * (self.frames - len(missing_mask))
        missing_mask = np.array(missing_mask, dtype=np.float32)

        # Create x_prev using masked frames (no explicit channel, so add one)
        # Shape: (T-1, 1, H, W, D)
        x_prev = np.clip(data[:-1] * missing_mask[:-1, None, None, None], 0., 1.)
        x_prev = x_prev[:, None]  # add channel dim

        # Target frame
        x = data[target_idx][None]  # shape: (1, H, W, D)

        return torch.from_numpy(x), torch.from_numpy(x_prev), self.files[idx]


# class ACDCDataset(Dataset):
#     def __init__(self, data_dir, split="train"):
#         self.data = np.transpose(np.load(os.path.join('./data/', f"{split}_dat.npy")), (0, 1, 4, 3, 2))[:, :, None]
#         self.files = np.load('./data/'+split+'_files.npy')
#         self.frames = self.data.shape[1]

#     def __len__(self):
#         return len(self.data)

#     def __getitem__(self, idx):
#         target_idx = np.random.randint(1, self.frames)
#         missing_mask = [1]
#         if target_idx>1:
#             if target_idx>2:
#                 missing_mask = np.append(missing_mask, np.random.randint(0, 2, size=(target_idx-2,)))
#             missing_mask = np.append(missing_mask, [1])
#         missing_mask = np.append(missing_mask, np.zeros(self.frames - len(missing_mask))).astype(np.float32)

#         x_prev = np.clip(self.data[idx, :-1] * missing_mask[:-1, None, None ,None, None], 0., 1.)
#         x = self.data[idx, target_idx]
#         return torch.from_numpy(x), torch.from_numpy(x_prev),  self.files[idx]
