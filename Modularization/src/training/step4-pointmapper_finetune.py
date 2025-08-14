# Eliminated the early stopping logic, and fixed as epoch3=150 to make the model overfit intentionally (same method in the paper)

import sys
from pathlib import Path

# path setting
BASE_DIR = Path(__file__).resolve().parent.parent.parent # LAB/Modularization/
SAVE_DIR = BASE_DIR / 'saved_models'
SAVE_DIR.mkdir(parents=True, exist_ok=True)
sys.path.append(str(BASE_DIR / 'src')) # for module import

import torch
from tqdm import tqdm
import torch.optim as optim 
from models.modeling import Encoder, PointMapper
from utils.data_utils import seed_everything, get_processed_dataloader
import wandb

# Configuration
seed_everything(42)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
train_loader, val_loader, _ = get_processed_dataloader()

# 1. Load pretrained Encoder
encoder = Encoder(window_size=2048)
checkpoint1 = torch.load(SAVE_DIR / f'step1_best_model_encoder.pt')
encoder.load_state_dict(checkpoint1['encoder_state_dict'])
encoder.to(device)
encoder.eval() 

# 2. Load pretrained PointMapper
point_mapper = PointMapper()
checkpoint2 = torch.load(SAVE_DIR / f'step2_best_model_point_mapper.pt')
point_mapper.load_state_dict(checkpoint2['point_mapper_state_dict'])
point_mapper.to(device)

# 3. Load critertion point a 
a = torch.load(SAVE_DIR / f"step3_criterion_point_a.pt").to(device)

# 4. Freeze encoder
for param in encoder.parameters():
    param.requires_grad = False

# 5. Optimizer for fine-tuning point mapper
optimizer = optim.Adam(point_mapper.parameters(), lr=1e-5)
epoch3 = 150

# wandb init
wandb.init(
    project="AERO",
    name=f"step4_pointmapper_finetune",
    config={
        "epochs": epoch3,
        "batch_size": train_loader.batch_size,
        "lr": 1e-5,
        "model": "Pointmapper",
        "optimizer": "Adam"
    }
)


# ==== Training ====
def train_one_epoch(model, dataloader, encoder, a, optimizer, device):
    model.train()
    total_loss = 0

    pb = tqdm(dataloader, desc='Fine-Tuning PM')
    for i, (batch, _) in enumerate(pb, 1):
        t, p, s = batch
        t, p, s = t.to(device), p.to(device), s.to(device)
        # t, p, s = (x.to(device) for x in batch)

        with torch.no_grad():
            h = encoder((t, p, s)) # (b, 704) : extract h from the previously trained encoder
        
        m = model(h)               # (b, d_m)
        loss = ((m - a)**2).sum(dim=1).mean() # L_M using fixed a

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()           # update parameter per epoch

        total_loss += loss.item()
        pb.set_postfix(total_loss=total_loss)

    return total_loss / i


# ==== Validation ====
@torch.no_grad()
def evaluate_on_val(model, dataloader, encoder, a, device):
    model.eval()
    total_loss = 0

    with torch.no_grad():
        pb = tqdm(dataloader, desc='Validation')
        for i, (batch, _) in enumerate(pb, 1):
            t, p, s = batch
            t, p, s = t.to(device), p.to(device), s.to(device)
            # t, p, s = (x.to(device) for x in batch)

            h = encoder((t, p, s))
            m = model(h)
            loss = ((m - a)**2).sum(dim=1).mean()
            total_loss += loss.item()
            pb.set_postfix(total_loss=total_loss)

    return total_loss / i


best_val, best_epoch = float('inf'), -1
best_path = SAVE_DIR / 'step4_best_point_mapper.pt'
final_path = SAVE_DIR / 'step4_finetuned_point_mapper.pt'

for epoch in range(epoch3+1):
    train_loss = train_one_epoch(point_mapper, train_loader, encoder, a, optimizer, device)
    val_loss = evaluate_on_val(point_mapper, val_loader, encoder, a, device)

    print(f"[Epoch {epoch:03d}] pm_train={train_loss:.10f}  pm_val={val_loss:.10f}")

    wandb.log({
        "epoch": epoch + 1,
        "train_loss": train_loss,
        "val_loss": val_loss
    })

    if val_loss < best_val:
        best_val, best_epoch = val_loss, epoch
        torch.save({
            'epoch': epoch,
            'point_mapper_state_dict': point_mapper.state_dict(), 
            'val_loss': val_loss,
        }, best_path)
        wandb.save(str(best_path))

# save last epoch
torch.save({
    'epoch': epoch3,
    'point_mapper_state_dict': point_mapper.state_dict(),
    'val_loss': val_loss
}, final_path)
wandb.save(str(final_path))

print(f"Best PM (fine-tune) @ epoch {best_epoch}: {best_val:.10f}")
wandb.finish()