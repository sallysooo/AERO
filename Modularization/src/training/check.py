# Test file to check the recorded epoch for each model files.

import sys
import os
from pathlib import Path

# path setting
BASE_DIR = Path(__file__).resolve().parent.parent.parent # LAB/Modularization/
SAVE_DIR = BASE_DIR / 'saved_models'
SAVE_DIR.mkdir(parents=True, exist_ok=True)

sys.path.append(str(BASE_DIR / 'src')) # for module import

import torch
from models.modeling import Encoder, PointMapper
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
print(f"Best encoder was saved at epoch1 : {checkpoint['epoch']}")


# 2. Pretrained PointMapper
point_mapper = PointMapper()
checkpoint2 = torch.load(SAVE_DIR / f'step2_best_model_point_mapper_{seed}.pt')
point_mapper.load_state_dict(checkpoint2['point_mapper_state_dict'])
point_mapper.to(device)
print(f"Best pointmapper was saved at epoch2 : {checkpoint2['epoch']}")


# 3. Finetuned PointMapper
point_mapper = PointMapper()
checkpoint3 = torch.load(SAVE_DIR / f'step4_finetuned_point_mapper_{seed}.pt')
point_mapper.load_state_dict(checkpoint2['point_mapper_state_dict'])
point_mapper.to(device)
print(f"Best pointmapper was saved at epoch3 : {checkpoint3['epoch']}")






