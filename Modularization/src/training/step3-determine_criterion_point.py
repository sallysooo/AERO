import sys
from pathlib import Path

# path setting
BASE_DIR = Path(__file__).resolve().parent.parent.parent # LAB/Modularization/
SAVE_DIR = BASE_DIR / 'saved_models'
SAVE_DIR.mkdir(parents=True, exist_ok=True)

sys.path.append(str(BASE_DIR / 'src')) # for module import

import torch
import torch.nn.functional as F
from tqdm import tqdm
from models.modeling import Encoder, PointMapper
from utils.data_utils import seed_everything, get_processed_dataloader

seed = 42
seed_everything(seed)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
train_loader, _, _ = get_processed_dataloader()


# 1. Encoder
encoder = Encoder(window_size=2048)
checkpoint1 = torch.load(SAVE_DIR / f'step1_best_model_encoder.pt')
encoder.load_state_dict(checkpoint1['encoder_state_dict'])
encoder.to(device)
encoder.eval() 

# 2. PointMapper
point_mapper = PointMapper()
checkpoint2 = torch.load(SAVE_DIR / f'step2_best_model_point_mapper.pt')
point_mapper.load_state_dict(checkpoint2['point_mapper_state_dict'])
point_mapper.to(device)
point_mapper.eval() 

# Model freeze
for param in encoder.parameters():
    param.requires_grad = False

for param in point_mapper.parameters():
    param.requires_grad = False

def calculate_criterion_point(encoder, point_mapper, dataloader, device):
    M_all = []

    with torch.no_grad():
        pb = tqdm(dataloader, desc='Determine a')
        for batch, _ in pb:
            t, p, s = batch
            t, p, s = t.to(device), p.to(device), s.to(device)
            
            h = encoder((t, p, s))   # (b, 704)
            m = point_mapper(h)      # (b, d_m)
            M_all.append(m)
    
    M_all = torch.cat(M_all, dim=0)  # (N, d_m)
    a = M_all.mean(dim=0)            # (d_m,)

    return a
            
a = calculate_criterion_point(encoder, point_mapper, train_loader, device)
torch.save(a.cpu(), SAVE_DIR / f"step3_criterion_point_a.pt")
