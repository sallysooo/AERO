import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

save_dir = './saved_models'
os.makedirs(save_dir, exist_ok=True)

from models.modeling import Autoencoder
import torch

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

model = Autoencoder().to(device)

checkpoint = torch.load(f'{save_dir}/step1_autoencoder_best_model.pt')
model.load_state_dict(checkpoint['model_state_dict'])

print(f"Best model from epoch {checkpoint['epoch']} with loss {checkpoint['val_loss']:.6f}")
