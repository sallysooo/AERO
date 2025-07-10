import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

save_dir = './saved_models'
os.makedirs(save_dir, exist_ok=True)

from models.modeling import Autoencoder
import torch
import torch.nn.functional as F
import torch.optim as optim 
from utils.data_utils import get_processed_dataset
from tqdm import tqdm

dataloader = get_processed_dataset()
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


model = Autoencoder().to(device)
optimizer = optim.Adam(model.parameters(), lr=1e-4)
best_loss = float('inf')
best_model_path = None

epoch1 = 5 # 논문에서는 20이지만 일단 작은 수로 잘 돌아가는지 테스트
for epoch in range(epoch1):
    loss_AE = 0.0              # loss값 누적할 변수 초기화
    pb = tqdm(dataloader, desc=f'{epoch=}')
    for x, _ in pb:
        t, p, s = (item.to(device) for item in x) 

        optimizer.zero_grad()       # 이전 배치에서 계산된 기울기 초기화
        t_hat, p_hat, s_hat = model((t, p, s))

        loss_t = F.mse_loss(t_hat, t, reduction='mean')  
        loss_p = F.mse_loss(p_hat, p, reduction='mean')
        loss_s = F.mse_loss(s_hat, s, reduction='mean')

        loss = loss_t + loss_p + loss_s     

        loss.backward() 
        optimizer.step() 
        loss_AE += loss.item() # batch의 loss값을 누적해 epoch 전체 loss를 계산

        pb.set_postfix(loss_t=loss_t.item(), loss_p=loss_p.item(), loss_s=loss_s.item(), loss_AE=loss_AE)

    cost = loss_AE / len(dataloader) # average loss 값 계산
    print(f'Epoch {epoch+1}: avg loss = {cost:.6f}')

    # Save every epoch
    torch.save(model.state_dict(), f'{save_dir}/step1_autoencoder_epoch_{epoch+1}.pt')
    
    # Save best model
    if cost < best_loss:
        best_loss = cost
        best_model_path = f'{save_dir}/step1_autoencoder_best_model.pt'
        torch.save({
            'epoch': epoch + 1,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': cost,
        }, best_model_path)





