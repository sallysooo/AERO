import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

save_dir = './saved_models'
os.makedirs(save_dir, exist_ok=True)

from models.modeling import Encoder
import torch

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# 1. Generate model instance
# if you want to extract the trained encoder only, implement as below :
encoder = Encoder(window_size=2048)
checkpoint = torch.load(f'{save_dir}/step1_best_model_encoder.pt')
encoder.load_state_dict(checkpoint['encoder_state_dict'])
encoder.to(device)
print(encoder.eval())

# 2. Usage example
'''
with torch.no_grad():
    h = encoder((t, p, s)) # latent vector

'''
