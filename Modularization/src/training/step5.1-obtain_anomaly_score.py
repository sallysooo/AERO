import sys
from pathlib import Path

# path setting
BASE_DIR = Path(__file__).resolve().parent.parent.parent # LAB/Modularization/
SAVE_DIR = BASE_DIR / 'saved_models'
SAVE_DIR.mkdir(parents=True, exist_ok=True)

sys.path.append(str(BASE_DIR / 'src')) # for module import

import torch
from tqdm import tqdm
from models.modeling import Encoder, PointMapper
from utils.data_utils import seed_everything, get_processed_dataloader
import numpy as np

# Configuration
seed = 42
seed_everything(seed)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
_, val_loader, _ = get_processed_dataloader()


# 1. Load pretrained Encoder
encoder = Encoder(window_size=2048)
checkpoint1 = torch.load(SAVE_DIR / f'step1_best_model_encoder.pt')
encoder.load_state_dict(checkpoint1['encoder_state_dict'])
encoder.to(device)
encoder.eval() 

# 2. Load finetuned PointMapper
point_mapper = PointMapper()
checkpoint2 = torch.load(SAVE_DIR / f'step4_finetuned_point_mapper.pt')
point_mapper.load_state_dict(checkpoint2['point_mapper_state_dict'])
point_mapper.to(device)
point_mapper.eval() 

# 3. Load critertion point a 
a = torch.load(SAVE_DIR / f"step3_criterion_point_a.pt").to(device)


def calculate_anomaly_scores(encoder, point_mapper, criterion_point, dataloader, device):
    # calculate anomaly score for each data in Sv
    all_scores = []

    with torch.no_grad():
        pb = tqdm(dataloader, desc='Inference on Sv')
        for batch, _ in pb:
            t, p, s = batch
            t, p, s = t.to(device), p.to(device), s.to(device)
            
            h = encoder((t, p, s))   # (b, 704)
            m = point_mapper(h)      # (b, d_m)
            # anomaly score
            scores = ((m - criterion_point)**2).sum(dim=1) # (batch,)
            all_scores.append(scores)

    scores = torch.cat(all_scores, dim=0).cpu().numpy()    

    return scores
            
anomaly_scores = calculate_anomaly_scores(encoder, point_mapper, a, val_loader, device) # : list l in the paper
np.save(SAVE_DIR / f'step5_anomaly_scores.npy', anomaly_scores)
print(anomaly_scores[:10])

'''
ver1(20, 10, 150):
[7.1846196e-10 7.1274658e-10 7.0360628e-10 6.9229611e-10 6.8181660e-10
 6.7246730e-10 6.7020695e-10 6.7287997e-10 6.7428385e-10 6.7031003e-10]
 
Ver2(E31, E26, 150):
[2.3472368e-09 2.2766338e-09 2.2530919e-09 2.2196591e-09 2.2441973e-09
 2.2587425e-09 2.3408857e-09 2.3507889e-09 2.3692051e-09 2.3956122e-09]
'''
# Since Sv set is consisted of benign data only, we can now calculate the p-percentile of the outlier score and obtain the threshold τ.