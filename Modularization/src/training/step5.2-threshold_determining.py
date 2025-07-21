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
import torch.optim as optim 
from tqdm import tqdm
from models.modeling import Encoder, PointMapper
from utils.data_utils import seed_everything, get_processed_dataloader
import wandb

# Configuration
seed = 42
seed_everything(seed)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
_, _, test_loader = get_processed_dataloader()


# Load scores & models
anomaly_scores_val = np.load(SAVE_DIR / f'step5_anomaly_scores_{seed}.npy')

######CHECK!!!!!!
test_labels = [] # 0 : benign / 1 : intrusion
test_scores = []

# 1. Load pretrained Encoder
encoder = Encoder(window_size=2048)
checkpoint1 = torch.load(SAVE_DIR / f'step1_best_model_encoder_{seed}.pt')
encoder.load_state_dict(checkpoint1['encoder_state_dict'])
encoder.to(device)
encoder.eval() 

# 2. Load finetuned PointMapper
point_mapper = PointMapper()
checkpoint2 = torch.load(SAVE_DIR / f'step4_finetuned_point_mapper_{seed}.pt')
point_mapper.load_state_dict(checkpoint2['point_mapper_state_dict'])
point_mapper.to(device)
point_mapper.eval() 

# 3. Load critertion point a 
a = torch.load(SAVE_DIR / f"step3_criterion_point_a_{seed}.pt").to(device)


with torch.no_grad():
    pb = tqdm(test_loader, desc='Test Evaluation')
    for batch, labels in pb:
        t, p, s = batch
        t, p, s = t.to(device), p.to(device), s.to(device)
            
        h = encoder((t, p, s))   # (b, 704)
        m = point_mapper(h)      # (b, d_m)
        # anomaly score
        scores = ((m - a)**2).sum(dim=1) # (batch,)
        test_scores.append(scores.cpu().numpy())
        test_labels.append(labels.numpy())

test_scores = np.concatenate(test_scores)
test_labels = np.concatenate(test_labels)

# range setting
p_values = np.arange(0.9950, 0.9997, 0.0001)
precisions, recalls, f1s = [], [], []

for p in p_values:
    τ = np.percentile(anomaly_scores_val, p*100)
    pred = (test_scores >= τ).astype(int)

    precision = precision_score(test_labels, pred)
    recall = recall_score(test_labels, pred)
    f1 = f1_score(test_labels, pred)

    precisions.append(precision)
    recalls.append(recall)
    f1s.append(f1)


# save best threshold
best_idx = np.argmax(f1s)
best_p = p_values[best_idx]
best_tau = np.percentile(anomaly_scores_val, best_p*100)
print(f"Best F1-score: {f1s[best_idx]:.4f} at p = {best_p}, τ = {best_tau:.6f}")

plt.figure(figsize=(10, 6))
plt.plot(p_values, precisions, label='Precision', marker='s')
plt.plot(p_values, recalls, label='Recall', marker='^')
plt.plot(p_values, f1s, label='F1-score', marker='o')
plt.xlabel('p')
plt.ylabel('Score')
plt.title('Evaluation metrics vs. threshold percentile p')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()


