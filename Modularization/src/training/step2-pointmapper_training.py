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
from utils.data_utils import seed_everything, get_processed_dataloader
import wandb

# Configuration
seed = 42
seed_everything(seed)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
train_loader, val_loader, _ = get_processed_dataloader()

# model setting w/ wandb init
point_mapper = PointMapper().to(device)
optimizer = optim.Adam(point_mapper.parameters(), lr=1e-5)

wandb.init(
    project="AERO",
    name=f"step2_pointmapper_seed{seed}",
    config={
        "epochs": 100,
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
    for batch, _ in pb:
        t, p, s = batch
        t, p, s = t.to(device), p.to(device), s.to(device)

        with torch.no_grad():
            h = encoder((t, p, s)) # (b, 704) : extract h from the previously trained encoder
        
        m = model(h)        # (b ,16)
        m_bar = m.mean(dim=0, keepdim=True) # (1, 16) : Column-wise mean of M

        # formula (3) : L_Pre = sum(||m_i - m_bar||^2)
        loss = ((m - m_bar)**2).sum() # (b, 16)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        pb.set_postfix(total_loss=total_loss)

    return total_loss / len(dataloader)


def evaluate_on_val(model, dataloader, device):
    # ==== Validation ====
    model.eval()
    total_loss = 0
    with torch.no_grad():
        pb = tqdm(dataloader, desc='PM Validation')
        for batch, _ in pb:
            t, p, s = batch
            t, p, s = t.to(device), p.to(device), s.to(device)

            h = encoder((t, p, s))
            m = model(h)
            m_bar = m.mean(dim=0, keepdim=True)
            loss = ((m - m_bar)**2).sum()
            total_loss += loss.item()
            pb.set_postfix(total_loss=total_loss)

    return total_loss / len(val_loader)


# Early Stopping settings
patience = 10
best_val_loss = float('inf')
counter = 0
min_delta = 1e-9

epoch2 = 10
for epoch in range(epoch2):
    train_loss = train_one_epoch(point_mapper, train_loader, optimizer, device)
    val_loss = evaluate_on_val(point_mapper, val_loader, device)

    print(f"Epoch {epoch+1} | Train Loss: {train_loss:.10f} | Val Loss: {val_loss:.10f}")

    wandb.log({
        "epoch": epoch + 1,
        "train_loss": train_loss,
        "val_loss": val_loss
    })

    # Early stopping logic
    if best_val_loss - val_loss > min_delta:
        best_val_loss = val_loss
        counter = 0
        point_mapper_model_path = SAVE_DIR / f'step2_best_model_point_mapper.pt'
        torch.save({
            'epoch': epoch + 1,
            'point_mapper_state_dict': point_mapper.state_dict(), 
            'val_loss': val_loss,
        }, point_mapper_model_path)

        wandb.save(str(point_mapper_model_path))
        print("Best model updated!")
    else:
        counter += 1
        if counter >= patience:
            print(f"Early stopping at epoch {epoch+1}")
            break

wandb.finish()