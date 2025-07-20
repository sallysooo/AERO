# Ver2 : Eliminated the early stopping logic, and fixed as epoch3=150 to make the model overfit intentionally

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
seed = 42
seed_everything(seed)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
train_loader, val_loader, _ = get_processed_dataloader()

# 1. Load pretrained Encoder
encoder = Encoder(window_size=2048)
checkpoint1 = torch.load(SAVE_DIR / f'step1_best_model_encoder_{seed}.pt')
encoder.load_state_dict(checkpoint1['encoder_state_dict'])
encoder.to(device)
encoder.eval() 

# 2. Load pretrained PointMapper
point_mapper = PointMapper()
checkpoint2 = torch.load(SAVE_DIR / f'step2_best_model_point_mapper_{seed}.pt')
point_mapper.load_state_dict(checkpoint2['point_mapper_state_dict'])
point_mapper.to(device)
point_mapper.eval() 

# 3. Load critertion point a 
a = torch.load(SAVE_DIR / f"step3_criterion_point_a_{seed}.pt").to(device)

# 4. Freeze encoder
for param in encoder.parameters():
    param.requires_grad = False

# 5. Optimizer for fine-tuning point mapper
optimizer = optim.Adam(point_mapper.parameters(), lr=1e-5)

# wandb init
wandb.init(
    project="AERO",
    name=f"step4_pointmapper_finetune_ver2_seed{seed}",
    config={
        "epochs": 150,
        "batch_size": train_loader.batch_size,
        "lr": 1e-5,
        "model": "Pointmapper",
        "optimizer": "Adam"
    }
)


# ==== Training ====
def train_one_epoch(model, dataloader, encoder, criterion_point, optimizer, device):
    model.train()
    total_loss = 0

    pb = tqdm(dataloader, desc='Fine-Tuning PM')
    for batch, _ in pb:
        t, p, s = batch
        t, p, s = t.to(device), p.to(device), s.to(device)

        with torch.no_grad():
            h = encoder((t, p, s)) # (b, 704) : extract h from the previously trained encoder
        
        m = model(h)               # (b, d_m)
        loss = ((m - criterion_point)**2).sum() # L_M using fixed a

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        pb.set_postfix(total_loss=total_loss)

    return total_loss / len(dataloader)


# ==== Validation ====
def evaluate_on_val(model, dataloader, encoder, criterion_point, device):
    model.eval()
    total_loss = 0

    with torch.no_grad():
        pb = tqdm(dataloader, desc='Validation')
        for batch, _ in pb:
            t, p, s = batch
            t, p, s = t.to(device), p.to(device), s.to(device)

            h = encoder((t, p, s))
            m = model(h)
            loss = ((m - criterion_point)**2).sum()
            total_loss += loss.item()
            pb.set_postfix(total_loss=total_loss)

    return total_loss / len(val_loader)



epoch3 = 150
for epoch in range(epoch3):
    train_loss = train_one_epoch(point_mapper, train_loader, encoder, a, optimizer, device)
    val_loss = evaluate_on_val(point_mapper, val_loader, encoder, a, device)

    print(f"Epoch {epoch+1} | Train Loss: {train_loss:.10f} | Val Loss: {val_loss:.10f}")

    wandb.log({
        "epoch": epoch + 1,
        "train_loss": train_loss,
        "val_loss": val_loss
    })

    save_path = SAVE_DIR / f'step4_finetuned_point_mapper_ver2_{seed}.pt'
    torch.save({
        'epoch': epoch + 1,
        'point_mapper_state_dict': point_mapper.state_dict(), 
        'val_loss': val_loss,
    }, save_path)

    wandb.save(str(save_path))

wandb.finish()