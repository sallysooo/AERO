import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import torch
import torch.nn.functional as F
import numpy as np
from sklearn.metrics import roc_auc_score, roc_curve, precision_recall_fscore_support
import matplotlib as plt
from models.modeling import Autoencoder
from utils.data_utils import get_processed_dataloader

best_model_path = './saved_models/step1_autoencoder_best_model.pt'
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

_, _, test_loader = get_processed_dataloader()

model = Autoencoder().to(device)
checkpoint = torch.load(best_model_path, map_location=device)
model.load_state_dict(checkpoint['model_state_dict'])


def compute_reconstruction_error(model, dataloader, device):
    model.eval()
    errors = []
    labels = []
    with torch.no_grad():
        for x, y in dataloader:
            t, p, s = (item.to(device) for item in x)
            t_hat, p_hat, s_hat = model((t, p, s))

            # MSE reconstruction error per sample

            batch_error = (
                F.mse_loss(t_hat, t, reduction='none').mean(dim=1) + # [batch_size, feature_dim] -> [batch_size] : get 1 error per 1 sample
                F.mse_loss(p_hat, p, reduction='none').mean(dim=1) +
                F.mse_loss(s_hat, s, reduction='none').mean(dim=1)
            )
            errors.extend(batch_error.detach().cpu().tolist())
            labels.extend(y.detach().cpu().tolist())
    return np.array(errors), np.array(labels)


# ROC/AUC, Precision, Recall, F1-score evaluation function
def evaluate_performance(errors, labels):
    # ROC/AUC
    roc_auc = roc_auc_score(labels, errors)
    fpr, tpr, thresholds = roc_curve(labels, errors)

    # Precision / Recall / F1 across thresholds
    precisions, recalls, f1s = [], [], []
    for t in thresholds:
        preds = (errors >= t).astype(int)
        precision, recall, f1, _ = precision_recall_fscore_support(labels, preds, average='binary')
        precisions.append(precision)
        recalls.append(recall)
        f1s.append(f1)

    # find best F1-score
    best_idx = np.argmax(f1s)
    best_threshold = thresholds[best_idx]

    return{
        'roc_auc' : roc_auc,
        'f1' : f1s[best_idx],
        'precision' : precisions[best_idx],
        'recall' : recalls[best_idx],
        'threshold' : best_threshold,
        'roc_curve': (fpr, tpr)
    }

print(f"Best model from epoch {checkpoint['epoch']} with val_loss {checkpoint['val_loss']:.6f}")

# start evaluation
errors, labels = compute_reconstruction_error(model, test_loader, device)
metrics = evaluate_performance(errors, labels)

# 결과 출력
print("================EVALUATION================")
print(f"[Best Threshold] {metrics['threshold']:.6f}")
print(f"[AUC] {metrics['roc_auc']:.4f}")
print(f"[Precision] {metrics['precision']:.4f}")
print(f"[Recall] {metrics['recall']:.4f}")
print(f"[F1 Score] {metrics['f1']:.4f}")

# ROC curve 시각화
fpr, tpr = metrics['roc_curve']
plt.plot(fpr, tpr, label=f"AUC = {metrics['roc_auc']:.4f}")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve")
plt.legend()
plt.grid()
plt.show()
