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
checkpoint1 = torch.load(SAVE_DIR / f'step1_best_model_encoder_{seed}.pt')
encoder.load_state_dict(checkpoint1['encoder_state_dict'])
encoder.to(device)
encoder.eval() 

# 2. Load finetuned PointMapper
point_mapper = PointMapper()
checkpoint2 = torch.load(SAVE_DIR / f'step4_finetuned_point_mapper_ver2_{seed}.pt')
point_mapper.load_state_dict(checkpoint2['point_mapper_state_dict'])
point_mapper.to(device)
point_mapper.eval() 

# 3. Load critertion point a 
a = torch.load(SAVE_DIR / f"step3_criterion_point_a_{seed}.pt").to(device)


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
np.save(SAVE_DIR / f'step5_anomaly_scores_ver2_{seed}.npy', anomaly_scores)
print(anomaly_scores[:10])
'''
ver1(epoch3=150):
[5.5829998e-09 5.4068185e-09 5.4098184e-09 5.3713185e-09 5.4181033e-09
 5.4534821e-09 5.4186016e-09 5.2496194e-09 5.2000981e-09 5.2491895e-09]
 
Ver2(Earlystopping ver.):
[2.6919240e-09 2.6271207e-09 2.5697080e-09 2.5697964e-09 2.6360110e-09
 2.6584144e-09 2.7549187e-09 2.8181013e-09 2.9413252e-09 2.8680758e-09]
'''
# Since Sv set is consisted of benign data only, we can now calculate the p-percentile of the outlier score and obtain the threshold τ.