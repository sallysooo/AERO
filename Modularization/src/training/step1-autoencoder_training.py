import sys
from pathlib import Path

# path setting
BASE_DIR = Path(__file__).resolve().parent.parent.parent # LAB/Modularization/
SAVE_DIR = BASE_DIR / 'saved_models'
SAVE_DIR.mkdir(parents=True, exist_ok=True)

sys.path.append(str(BASE_DIR / 'src')) # for module import

import torch
import torch.nn.functional as F
import torch.optim as optim 
from tqdm import tqdm
from models.modeling import Autoencoder
from utils.data_utils import seed_everything, get_processed_dataloader
import wandb

# Configuration
seed = 42
seed_everything(seed)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
train_loader, val_loader, _ = get_processed_dataloader()

# model setting w/ wandb init
model = Autoencoder().to(device)
optimizer = optim.Adam(model.parameters(), lr=1e-4)

wandb.init(
    project="AERO",
    name=f"step1_autoencoder_seed{seed}",
    config={
        "epochs": 100,
        "batch_size": train_loader.batch_size,
        "lr": 1e-4,
        "model": "Autoencoder",
        "optimizer": "Adam"
    }
)


def train_one_epoch(model, dataloader, optimizer, device):
    model.train()
    total_loss = 0
    pb = tqdm(dataloader, desc='Training')
    for x, _ in pb:
        t, p, s = (item.to(device) for item in x)  # [64, 9], [64, 2048, 9], [64, 9]
        optimizer.zero_grad()
        t_hat, p_hat, s_hat = model((t, p, s))     # [64, 9], [64, 2048, 9], [64, 9]

        loss = F.mse_loss(t_hat, t) + F.mse_loss(p_hat, p) + F.mse_loss(s_hat, s)
        loss.backward() 
        optimizer.step() 

        total_loss += loss.item()
        pb.set_postfix(total_loss=total_loss)
    return total_loss / len(dataloader)

def evaluate_on_val(model, dataloader, device):
    model.eval()
    total_loss = 0
    pb = tqdm(dataloader, desc='Validation')
    with torch.no_grad():
        for x, _ in pb:
            t, p, s = (item.to(device) for item in x) # [64, 9], [64, 2048, 9], [64, 9]
            t_hat, p_hat, s_hat = model((t, p, s))

            loss = F.mse_loss(t_hat, t) + F.mse_loss(p_hat, p) + F.mse_loss(s_hat, s)
            
            total_loss += loss.item()
            pb.set_postfix(total_loss=total_loss)
    return total_loss / len(dataloader)



# Early Stopping settings
patience = 10
best_val_loss = float('inf')
counter = 0

epoch1 = 100
for epoch in range(epoch1):
    train_loss = train_one_epoch(model, train_loader, optimizer, device)
    val_loss = evaluate_on_val(model, val_loader, device)

    print(f"Epoch {epoch+1} | Train Loss: {train_loss:.6f} | Val Loss: {val_loss:.6f}")

    if val_loss < best_val_loss:
        best_val_loss = val_loss
        counter = 0
        encoder_model_path = SAVE_DIR / f'step1_best_model_encoder_{seed}.pt'
        encoder = model.get_encoder()
        torch.save({
            'epoch': epoch + 1,
            'encoder_state_dict': encoder.state_dict(), # save the trained encoder only (discard decoder)
            'val_loss': val_loss,
        }, encoder_model_path)

        wandb.save(str(encoder_model_path))
        print("Best model updated!")
    else:
        counter += 1
        if counter >= patience:
            print(f"Early stopping at epoch {epoch+1}")
            break

