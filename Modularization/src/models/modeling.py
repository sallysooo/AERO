import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim 
import math


class SeparableConv1d(nn.Module):
    def __init__(self, input_c, output_c, kernel_size, **kwargs):
        super().__init__()
        self.depthwise = nn.Conv1d(input_c, input_c, kernel_size, groups=input_c, **kwargs)
        self.pointwise = nn.Conv1d(input_c, output_c, 1)
    
    def forward(self, x):
        x = self.depthwise(x) 
        x = self.pointwise(x)
        return x

class SeparableConvTranspose1d(nn.Module):
    def __init__(self, input_c, output_c, kernel_size, **kwargs):
        super().__init__()
        self.depthwise = nn.ConvTranspose1d(input_c, input_c, kernel_size, groups=input_c, **kwargs)
        self.pointwise = nn.ConvTranspose1d(input_c, output_c, 1)
    
    def forward(self, x):
        x = self.depthwise(x)
        x = self.pointwise(x)
        return x

def calculate_num_p_layers(input_length, min_length=10):
    num_layers = 0
    while input_length > min_length:
        input_length //= 2
        num_layers += 1
    return num_layers


def calculate_upsampling_steps(window_size):
    log2_ws = math.log2(window_size)
    if not log2_ws.is_integer(): # 2의 배수가 아니면 예외처리
        raise ValueError(f"window_size {window_size} must be a power of 2.")
    return int(log2_ws)  # ex: 2048 → 11, 1024 → 10


class Encoder(nn.Module):
    def __init__(self, window_size=2048):
        super().__init__()
        
        # 1. Encoder 
        self.t_fc1 = nn.Linear(9, 64)
        self.t_fc2 = nn.Linear(64, 64)

        self.s_fc1 = nn.Linear(9, 64)
        self.s_fc2 = nn.Linear(64, 64)

        num_p_layers = calculate_num_p_layers(window_size)
        p_layers = []
        in_ch = 9
        for i in range(num_p_layers):
            out_ch = 64 if i < num_p_layers - 1 else 576
            p_layers.append(SeparableConv1d(in_ch, out_ch, kernel_size=3, padding=1))
            in_ch = out_ch
        self.p_layers = nn.ModuleList(p_layers)

    def forward(self, x):
        t, p, s = x

        # 1. Encoder
        t = F.relu(self.t_fc1(t))
        t = F.relu(self.t_fc2(t))

        s = F.relu(self.s_fc1(s))
        s = F.relu(self.s_fc2(s))
        
        '''
        for self.p_layers = SeparableConv1D layer
        each layer : (b, in_ch, seq_len) -> (b, out_ch, seq_len) + add ReLU activation function
        ex) (b, 9, 2048) → relu → (b, 64, 2048) → maxpool → (b, 64, 1024) → relu & maxpool → (b, 64, 512) ... → (b, 128, 512) 
        - 각 layer 마다 seq_len이 절반으로 감소: 2048 → 1024 → 512 → 256 → 128 → 64 → 32 → 16 ← (8번 Pooling)
        - 각 layer 마다 channel 수를 점점 키워서 576 만들기: 9 → 32 → 64 → 128 → 256 → ... → 576
        '''
        # p : (b, 2048, 9)
        p = p.permute(0, 2, 1) # (b, 9, 2048) : (batch, channel, length)
        for layer in self.p_layers[:-1]:
            p = F.relu(layer(p)) # target is C / same length
            p = F.max_pool1d(p, 2) # shorten length 1/2 by maxpool
        p = self.p_layers[-1](p) # (b, 576, 16)
        p = F.adaptive_max_pool1d(p, 1).squeeze(2)  # latent vector (b, 576) : output length = 1

        # latent vector
        h = torch.cat([t, p, s], dim=1) # (b, 704)
        return h


class Decoder(nn.Module):
    def __init__(self, window_size=2048):
        super().__init__()
        # 2. Decoder 
        self.t_dec_fc1 = nn.Linear(704, 64)
        self.t_dec_fc2 = nn.Linear(64, 64)
        self.t_dec_fc3 = nn.Linear(64, 9)

        self.s_dec_fc1 = nn.Linear(704, 64)
        self.s_dec_fc2 = nn.Linear(64, 64)
        self.s_dec_fc3 = nn.Linear(64, 9)

        num_upsample_layers = calculate_upsampling_steps(window_size)
        
        self.p_dec_layers = nn.ModuleList()
        in_ch = 704
        for i in range(num_upsample_layers):
            out_ch = 64
            self.p_dec_layers.append(SeparableConvTranspose1d(in_ch, out_ch, kernel_size=3, padding=1))
            in_ch = out_ch
        # last layer : channel 64 -> 9
        self.p_output_layer = SeparableConvTranspose1d(64, 9, kernel_size=3, padding=1)

        
    def forward(self, h):
        # 2. Decoder
        t = F.relu(self.t_dec_fc1(h))
        t = F.relu(self.t_dec_fc2(t))
        t = self.t_dec_fc3(t)

        s = F.relu(self.s_dec_fc1(h))
        s = F.relu(self.s_dec_fc2(s))
        s = self.s_dec_fc3(s)

        '''
        - unsqueeze(2) : add new dimension as (b, 704) -> (b, 704, 1)
        - upsampling: 16 → 2048 requires 7 upsampling steps: 1 → 2 → 4 → ... → 128 → 256 → 512 → 1024 → 2048
        
        Latent vector (b, 704)
         ↓ unsqueeze → (b, 704, 1)
         ↓ ConvTranspose1d: 704 → 64
         ↓ interpolate (2x length) → (b, 64, 2)
         ↓ ConvTranspose1d: 64 → 64
         ↓ interpolate → (b, 64, 4)
         ↓ ...
         ↓ Final ConvTranspose1d: 64 → 9
         ↓ (b, 9, 2048)
         ↓ permute → (b, 2048, 9)

        '''
        p = h.unsqueeze(2) # (b, 704) -> (b, 704, 1)
        for layer in self.p_dec_layers:
            p = F.relu(layer(p))
            p = F.interpolate(p, scale_factor=2, mode='nearest') # upsample : double the length
            
        p = self.p_output_layer(p) # final channel 64 -> 9 : (b, 9, 2048)
        p = p.permute(0, 2, 1) # (b, 2048, 9)

        return t, p, s


class Autoencoder(nn.Module):
    def __init__(self, window_size=2048):
        super(Autoencoder, self).__init__()
        self.encoder = Encoder(window_size)
        self.decoder = Decoder(window_size)

    def forward(self, x):
        h = self.encoder(x)
        return self.decoder(h)






