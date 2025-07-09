import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from models.modeling import Autoencoder
import torch
import torch.nn.functional as F
import torch.optim as optim 
from utils.data_utils import get_processed_dataset
from tqdm import tqdm

T, P, S, dataset, dataloader = get_processed_dataset()
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


model = Autoencoder().to(device)
# encoder = autoencoder.encoder
optimizer = optim.Adam(model.parameters(), lr=1e-4)


epoch1 = 2 # 논문에서는 20이지만 일단 작은 수로 잘 돌아가는지 테스트
for epoch in range(epoch1):
    loss_AE = 0.0              # loss값 누적할 변수 초기화
    pb = tqdm(dataloader, desc=f'{epoch=}')
    for x, _ in pb:
        t, p, s = (item.to(device) for item in x) 

        optimizer.zero_grad()       # 이전 배치에서 계산된 기울기 초기화
        t_hat, p_hat, s_hat = model((t, p, s))
        # print(f't : {t.shape} | p : {p.shape} | s : {s.shape}')
        # print(f't_hat : {t_hat.shape} | p_hat : {p_hat.shape} | s_hat : {s_hat.shape}')
        # loss = ((t - t_hat)**2 + (p - p_hat)**2 + (s - s_hat)**2).mean() # MSE
        
        # T loss
        loss_t = F.mse_loss(t_hat, t, reduction='mean')  
        # print(loss_t)
        # P loss
        loss_p = F.mse_loss(p_hat, p, reduction='mean')
        # print(loss_p)
        # S loss
        loss_s = F.mse_loss(s_hat, s, reduction='mean')
        # print(loss_s)

        # loss 합
        loss = loss_t + loss_p + loss_s     
        # print(loss)

        loss.backward() 
        optimizer.step() 
        loss_AE += loss.item() # batch의 loss값을 누적해 epoch 전체 loss를 계산

        pb.set_postfix(loss_t=loss_t.item(), loss_p=loss_p.item(), loss_s=loss_s.item(), loss_AE=loss_AE)
        # print(loss_AE)

    cost = loss_AE / len(dataloader) # average loss 값 계산
    print(f'{cost:.6f}')
    # if (epoch + 1) % 10 == 0 or epoch == 0:
        # print(f"[{epoch + 1}] loss: {cost:.3f}") # 현재 epoch와 평균 loss 값을 출력





