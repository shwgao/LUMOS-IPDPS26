import numpy as np
import torch
import os
import glob
from torch.utils.data import Dataset
from torch.utils.data import DataLoader

from utils import delete_constant_columns, reshape_vector, Load_Dataset_Surrogate


# Custom dataset class
class NPZDataset(Dataset):
    def __init__(self, npz_root):
        self.files = glob.glob(npz_root + "/*.npz")

    def __getitem__(self, index):
        sample = np.load(self.files[index])
        x = torch.from_numpy(sample["data"])
        y = sample["label"][0]
        return x, y

    def __len__(self):
        return len(self.files)


def get_loader(batch_size=32, val_only=False, data_only=False, rank=0):
    basePath = '/nfs/hpc/share/gaosho/dataset/loxia/stemdl_ds1/'
    # modelPath = params_in.output_dir / f'stemdlModel.h5'
    trainingPath =  os.path.join(basePath, 'training')
    validationPath = os.path.join(basePath, 'validation')

    # Datasets: training (138717 files), validation (20000 files), 
    # testing (20000 files), prediction (8438 files), 197kbytes each
    train_dataset = NPZDataset(trainingPath)
    val_dataset = NPZDataset(validationPath) if rank == 0 else None

    if data_only:
        return train_dataset, val_dataset

    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4,
                              pin_memory=True) if not val_only else None

    return train_loader, val_loader
