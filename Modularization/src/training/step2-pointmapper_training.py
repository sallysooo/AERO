# no seed, no early stopping

import sys
from pathlib import Path

# path setting
BASE_DIR = Path(__file__).resolve().parent.parent.parent # LAB/Modularization/
SAVE_DIR = BASE_DIR / 'saved_models'
SAVE_DIR.mkdir(parents=True, exist_ok=True)
sys.path.append(str(BASE_DIR / 'src')) # for module import

import torch
import torch.optim as optim 
from tqdm import tqdm
from models.modeling import Encoder, PointMapper
from utils.data_utils import get_processed_dataloader, seed_everything
import wandb

seed_everything(42)

# Configuration
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
train_loader, val_loader, _ = get_processed_dataloader()

# model setting w/ wandb init
point_mapper = PointMapper().to(device)
optimizer = optim.Adam(point_mapper.parameters(), lr=1e-5)

epoch2 = 10

wandb.init(
    project="AERO",
    name=f"step2_pointmapper_training",
    config={
        "epochs": epoch2,
        "batch_size": train_loader.batch_size,
        "lr": 1e-5,
        "model": "PointMapper",
        "optimizer": "Adam"
    }
)

encoder = Encoder(window_size=2048)
checkpoint = torch.load(SAVE_DIR / f'step1_best_model_encoder.pt')
encoder.load_state_dict(checkpoint['encoder_state_dict'])
encoder.to(device)
encoder.eval() 

# Freeze encoder
for param in encoder.parameters():
    param.requires_grad = False

def train_one_epoch(model, dataloader, optimizer, device):
    model.train()
    total_loss = 0

    pb = tqdm(dataloader, desc='PM Training')
    for i, (batch, _) in enumerate(pb, 1):
        t, p, s = batch
        t, p, s = t.to(device), p.to(device), s.to(device)
        # t, p, s = (item.to(device) for item in batch)

        with torch.no_grad():
            h = encoder((t, p, s)) # (b, 704) : extract h from the previously trained encoder
        
        m = model(h)        # (b ,16)
        m_bar = m.mean(dim=0, keepdim=True) # (1, 16) : Column-wise mean of M

        # formula (3) : L_Pre = sum(||m_i - m_bar||^2)
        loss = ((m - m_bar)**2).sum(dim=1).mean() # (b, 16)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        pb.set_postfix(total_loss=total_loss)

    return total_loss / i

@torch.no_grad()
def evaluate_on_val(model, dataloader, device):
    # ==== Validation ====
    model.eval()
    total_loss = 0
    with torch.no_grad():
        pb = tqdm(dataloader, desc='PM Validation')
        for i, (batch, _) in enumerate(pb, 1):
            t, p, s = batch
            t, p, s = t.to(device), p.to(device), s.to(device)
            # t, p, s = (item.to(device) for item in batch)

            h = encoder((t, p, s))
            m = model(h)
            m_bar = m.mean(dim=0, keepdim=True)
            loss = ((m - m_bar)**2).sum(dim=1).mean()
            total_loss += loss.item()
            pb.set_postfix(total_loss=total_loss)

    return total_loss / i


best_val_loss = float('inf')
best_epoch = -1
point_mapper_model_path = SAVE_DIR / f'step2_best_model_point_mapper.pt'


for epoch in range(1, epoch2+1):
    train_loss = train_one_epoch(point_mapper, train_loader, optimizer, device)
    val_loss = evaluate_on_val(point_mapper, val_loader, device)

    wandb.log({
        "epoch": epoch,
        "train_loss": train_loss,
        "val_loss": val_loss
    })
    print(f"[Epoch {epoch:02d}] pm_train={train_loss:.10f}  pm_val={val_loss:.10f}")

    if val_loss < best_val_loss:
        best_val_loss, best_epoch = val_loss, epoch
        torch.save({
            'epoch': epoch,
            'point_mapper_state_dict': point_mapper.state_dict(), 
            'val_loss': val_loss,
        }, point_mapper_model_path)
        wandb.save(str(point_mapper_model_path))

print(f"Best PM val @ epoch {best_epoch}: {best_val_loss:.10f}")
wandb.finish()
