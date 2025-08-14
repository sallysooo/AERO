# no seed, no early stopping

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
from utils.data_utils import get_processed_dataloader, seed_everything
import wandb

seed_everything(42)

# Configuration
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
train_loader, val_loader, _ = get_processed_dataloader()

# model setting w/ wandb init
model = Autoencoder().to(device)
optimizer = optim.Adam(model.parameters(), lr=1e-4)

epoch1 = 20

wandb.init(
    project="AERO",
    name=f"step1_autoencoder_training",
    config={
        "epochs": epoch1,
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
    for i, (x, _) in enumerate(pb, 1):
        t, p, s = (item.to(device) for item in x)  # [64, 9], [64, 2048, 9], [64, 9]
        optimizer.zero_grad()
        t_hat, p_hat, s_hat = model((t, p, s))     # [64, 9], [64, 2048, 9], [64, 9]

        loss = F.mse_loss(t_hat, t) + F.mse_loss(p_hat, p) + F.mse_loss(s_hat, s)
        loss.backward() 
        optimizer.step() 

        total_loss += loss.item()
        pb.set_postfix(total_loss=total_loss)
    return total_loss / i

@torch.no_grad()
def evaluate_on_val(model, dataloader, device):
    model.eval()
    total_loss = 0
    pb = tqdm(dataloader, desc='Validation')
    with torch.no_grad():
        for i, (x, _) in enumerate(pb, 1):
            t, p, s = (item.to(device) for item in x) # [64, 9], [64, 2048, 9], [64, 9]
            t_hat, p_hat, s_hat = model((t, p, s))

            loss = F.mse_loss(t_hat, t) + F.mse_loss(p_hat, p) + F.mse_loss(s_hat, s)
            
            total_loss += loss.item()
            pb.set_postfix(total_loss=total_loss)
    return total_loss / i


best_val_loss = float('inf')
best_epoch = -1
encoder_model_path = SAVE_DIR / f'step1_best_model_encoder.pt'


for epoch in range(1, epoch1+1):
    train_loss = train_one_epoch(model, train_loader, optimizer, device)
    val_loss = evaluate_on_val(model, val_loader, device)

    wandb.log({"epoch": epoch, "train_loss": train_loss, "val_loss": val_loss})
    print(f"[Epoch {epoch:02d}] train={train_loss:.6f}  val={val_loss:.6f}")

    if val_loss < best_val_loss:
        best_val_loss, best_epoch = val_loss, epoch
        encoder = model.get_encoder()
        torch.save({
            'epoch': epoch,
            'encoder_state_dict': encoder.state_dict(), # save the trained encoder only (discard decoder)
            'val_loss': val_loss,
        }, encoder_model_path)
        wandb.save(str(encoder_model_path))

print(f"Best val @ epoch {best_epoch}: {best_val_loss:.6f}")
