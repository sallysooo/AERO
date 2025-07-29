
'''
import numpy as np

y_seq = np.array([0,0,0,1,0,0,0,0,1,0]) # 10 packets
window_size = 4
stride = 2

all_indices = np.arange(window_size, len(y_seq) + 1, stride)
labels = [y_seq[i - window_size : i].max() for i in all_indices]
print(labels) # [1,1,0,1]

# window 1: [0,0,0,1] → 1
# window 2: [0,1,0,0] → 1
# window 3: [0,0,0,0] → 0
# window 4: [0,0,0,1] → 1
# [np.int64(1), np.int64(1), np.int64(0), np.int64(1)]

'''


import sys
from pathlib import Path

# path setting
BASE_DIR = Path(__file__).resolve().parent.parent.parent # LAB/Modularization/
SAVE_DIR = BASE_DIR / 'saved_models'
SAVE_DIR.mkdir(parents=True, exist_ok=True)

sys.path.append(str(BASE_DIR / 'src')) # for module import

import numpy as np
import torch
from utils.data_utils import seed_everything, get_processed_dataloader

# Configuration
seed = 42
seed_everything(seed)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
_, val_loader, _ = get_processed_dataloader()


val_labels = []
for _, labels in val_loader:
    val_labels.append(labels.numpy()) # still ERROR...

val_labels = np.concatenate(val_labels)
print("Number of anomalous samples in validation set:", (val_labels == 1).sum())
