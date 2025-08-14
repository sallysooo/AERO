import sys
from pathlib import Path

# path setting
BASE_DIR = Path(__file__).resolve().parent.parent.parent # LAB/Modularization/
SAVE_DIR = BASE_DIR / 'saved_models'
SAVE_DIR.mkdir(parents=True, exist_ok=True)
sys.path.append(str(BASE_DIR / 'src')) # for module import

import numpy as np
from sklearn.metrics import precision_score, recall_score, f1_score
import matplotlib.pyplot as plt
import torch
from tqdm import tqdm
from models.modeling import Encoder, PointMapper
from utils.data_utils import seed_everything, get_processed_dataloader

seed_everything(42)

# Configuration
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
_, _, test_loader = get_processed_dataloader()


# Load scores & models
anomaly_scores_val = np.load(SAVE_DIR / f'step5_anomaly_scores.npy')

test_labels = [] # 0 : benign / 1 : intrusion
test_scores = []

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


with torch.no_grad():
    pb = tqdm(test_loader, desc='Test Evaluation')
    for batch, labels in pb:
        t, p, s = batch
        t, p, s = t.to(device), p.to(device), s.to(device)
        # t, p, s = (x.to(device) for x in batch)
            
        h = encoder((t, p, s))   # (b, 704)
        m = point_mapper(h)      # (b, d_m)
        # anomaly score
        scores = ((m - a)**2).sum(dim=1) # (batch,)
        test_scores.append(scores.cpu().numpy())
        test_labels.append(labels.numpy())

test_scores = np.concatenate(test_scores)
test_labels = np.concatenate(test_labels)

np.save(SAVE_DIR / f"step5.2_anomaly_scores_test.npy", test_scores)
np.save(SAVE_DIR / f"step5.2_labels_test.npy", test_labels)


# ---- Percentile sweep (2-stage) ----
# 1) coarse: 0.9900 ~ 0.9990
p_coarse = np.arange(0.9900, 0.9990 + 1e-12, 0.001)
# 2) fine:   0.9990 ~ 0.9996 (논문 구간: 0.9990~0.9994)
p_fine   = np.arange(0.9990, 0.9996 + 1e-12, 0.0001)
p_values = np.unique(np.concatenate([p_coarse, p_fine]))

# # ---- Fine sweep in the extreme tail (0.9990 ~ 1.0000) ----
# # 0.9990~0.9999: 1e-4 간격, 0.99990~0.99999: 1e-5 간격
# p_range_1 = np.arange(0.9990, 0.9999 + 1e-12, 1e-4)
# p_range_2 = np.arange(0.99990, 0.99999 + 1e-12, 1e-5)

# p_values = np.unique(np.clip(np.concatenate([p_range_1, p_range_2]), 0.0, 1.0 - 1e-12))


precisions, recalls, f1s = [], [], []

for p in p_values:
    tau = np.percentile(anomaly_scores_val, p*100.0)
    pred = (test_scores >= tau).astype(int)

    precision = precision_score(test_labels, pred, zero_division=0)
    recall = recall_score(test_labels, pred, zero_division=0)
    f1 = f1_score(test_labels, pred, zero_division=0)

    precisions.append(precision)
    recalls.append(recall)
    f1s.append(f1)


# save best threshold
best_idx = np.argmax(f1s)
best_p = p_values[best_idx]
best_tau = np.percentile(anomaly_scores_val, best_p*100.0)
print(f"Best F1-score: {f1s[best_idx]:.4f} at p={best_p:.4f}, tau={best_tau:.8e}")

plt.figure(figsize=(10, 6))
plt.plot(p_values, precisions, color='blue', label='Precision', marker='s')
plt.plot(p_values, recalls, color='green', label='Recall', marker='^')
plt.plot(p_values, f1s, color='orange', label='F1-score', marker='o')
plt.xlabel('p')
plt.ylabel('Score')
plt.title('Evaluation metrics vs. threshold percentile p')
plt.legend()
plt.grid(True)

from pathlib import Path
IMG_DIR = BASE_DIR / 'img' 
IMG_DIR.mkdir(parents=True, exist_ok=True)
out_path = IMG_DIR / 'ver6.1.png'
plt.tight_layout()
plt.savefig(out_path, dpi=150)
print(f"Plot saved to: {out_path}")
plt.show()


'''
ver0(ver2 in previous step) => Best F1-score: 0.9865 at p = 0.973, tau = 0.0000000094 (SOTA)
ver1 => Best F1-score: 0.8269 at p = 0.94, tau = 0.0000000018
ver2 => Best F1-score: 0.6935 at p = 0.9, tau = 0.0000000026
ver3 => Best F1-score: 0.5946 at p = 0.8, tau = 0.0000000054
ver4 => Best F1-score: 0.9018 at p = 0.9750000000000001, tau = 0.0000000202
ver5 => Best F1-score: 0.6308 at p = 0.9, tau = 0.0000000132

ver6 => Best F1-score: 0.9799 at p=0.9996, tau=8.16895898e-09
'''
