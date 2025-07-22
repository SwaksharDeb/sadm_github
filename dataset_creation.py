# import os
# import torch
# import numpy as np

# import os
# import torch
# import numpy as np

# def save_each_subject_separately(split='train'):
#     """
#     Loads each .pt file from the dataset, converts to .npy, and saves it immediately.
#     Avoids keeping everything in memory.
#     """
#     dataset_path = './dataset/'
#     save_dir = os.path.join('data', split)
#     os.makedirs(save_dir, exist_ok=True)

#     files = []
#     idx = 0

#     for sp in ['ad_split', 'cn_split', 'mci_split']:
#         sp_dir = os.path.join(dataset_path, sp, split)
#         if not os.path.exists(sp_dir):
#             print(f"Warning: {sp_dir} does not exist. Skipping.")
#             continue

#         for f in os.listdir(sp_dir):
#             full_path = os.path.join(sp_dir, f)
#             try:
#                 data = torch.load(full_path)
#                 out_file = os.path.join(save_dir, f"{idx:04d}.npy")
#                 np.save(out_file, data.numpy())
#                 files.append(f)
#                 print(f"Saved: {out_file} from {f}")
#                 idx += 1
#             except Exception as e:
#                 print(f"Failed to load {f}: {e}")

#     # Save the list of filenames (original dataset .pt filenames)
#     np.save(os.path.join('data', f"{split}_files.npy"), files)
#     print(f"Saved file list: data/{split}_files.npy")

# if __name__ == "__main__":
#     save_each_subject_separately('train')
#     save_each_subject_separately('test')




import os, torch, numpy
subs=[]; files=[]
os.makedirs("data_2", exist_ok=True)
path = './dataset/'
for sp in ['ad_split', 'cn_split', 'mci_split']:
    sp_dir = os.path.join(path, sp, 'train')
    for f in os.listdir(sp_dir):
        subs.append(torch.load(os.path.join(sp_dir, f)))
        files.append(f)
subs = torch.stack(subs, dim=0).numpy()
numpy.save('data_2/train_dat.npy', subs)
numpy.save('data_2/train_files.npy', files)


import os, torch, numpy
subs=[]; files=[]
path = './dataset/'
for sp in ['ad_split', 'cn_split', 'mci_split']:
    sp_dir = os.path.join(path, sp, 'test')
    for f in os.listdir(sp_dir):
        subs.append(torch.load(os.path.join(sp_dir, f)))
        files.append(f)
subs = torch.stack(subs, dim=0).numpy()
numpy.save('data_2/test_dat.npy', subs)
numpy.save('data_2/test_files.npy', files)