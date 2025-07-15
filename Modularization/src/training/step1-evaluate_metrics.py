import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import torch
import torch.nn.functional as F
import numpy as np
from sklearn.metrics import roc_auc_score, roc_curve, precision_recall_fscore_support
import matplotlib.pyplot as plt
from models.modeling import Autoencoder
from utils.data_utils import get_processed_dataloader
from tqdm import tqdm

best_model_path = './saved_models/step1_autoencoder_best_model.pt'
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

_, _, test_loader = get_processed_dataloader()

model = Autoencoder().to(device)
checkpoint = torch.load(best_model_path, map_location=device)
model.load_state_dict(checkpoint['model_state_dict'])



def compute_reconstruction_error(model, dataloader, device):

    model.eval()
    all_errors = []

    with torch.no_grad():
        for x, _ in tqdm(dataloader, desc="Evaluating"):
            try:
                inputs = [item.to(device) for item in x]
                outputs = model(tuple(inputs))

                batch_errors = []
                for inp, out in zip(inputs, outputs):
                    err = F.mse_loss(out, inp, reduction='none')\
                        .reshape(inp.size(0), -1).mean(dim=1)
                    batch_errors.append(err)

                total_error = torch.stack(batch_errors, dim=1).sum(dim=1)
                all_errors.append(total_error.cpu())

            except Exception as e:
                print(f"[!] Error during batch evaluation: {e}")
                continue

    all_errors = torch.cat(all_errors).numpy()
    return all_errors


    
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
# errors, labels = compute_reconstruction_error(model, test_loader, device)
errors = compute_reconstruction_error(model, test_loader, device)
plt.hist(errors, bins=100)
plt.title("Reconstruction Error Distribution")
plt.xlabel("Error")
plt.ylabel("Count")
plt.grid()
plt.show()



'''
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


'''


