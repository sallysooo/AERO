import sys
import os
from pathlib import Path

# path setting
BASE_DIR = Path(__file__).resolve().parent.parent.parent # LAB/Modularization/
SAVE_DIR = BASE_DIR / 'saved_models'
SAVE_DIR.mkdir(parents=True, exist_ok=True)

sys.path.append(str(BASE_DIR / 'src')) # for module import

import torch
from models.modeling import Encoder
from utils.data_utils import seed_everything

seed = 42
seed_everything(seed)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


# 1. Generate model instance
# if you want to extract the trained encoder only, implement as below :
encoder = Encoder(window_size=2048)
checkpoint = torch.load(SAVE_DIR / f'step1_best_model_encoder_{seed}.pt')
encoder.load_state_dict(checkpoint['encoder_state_dict'])
encoder.to(device)
print(encoder.eval())

# 2. Usage example
'''
with torch.no_grad():
    h = encoder((t, p, s)) # latent vector

'''
