# -*- coding: utf-8 -*-
# @FileName : DQN.py
# @Project  : MAHJONG AI
# @Author   : WANG Jianxing
# @Time     : 2021/5/4 22:45
from functools import partial
from collections import Counter
import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class SamePadConv2d(nn.Conv2d):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.padding = (self.kernel_size[0] // 2, self.kernel_size[1] // 2)

conv3x1 = partial(SamePadConv2d, kernel_size=(3,1))

class ResidualBlock(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.layer1 = self.make_layer(in_channels)
        self.layer2 = self.make_layer(in_channels)

    def make_layer(self, in_channels, dropout_prob=0.5):
        layer = nn.Sequential(
            conv3x1(in_channels, in_channels),
            nn.BatchNorm2d(256),
            nn.Dropout2d(dropout_prob),
            nn.LeakyReLU()
        )
        return layer

    def forward(self, x):
        out = self.layer1(x)
        out = self.layer2(out)
        out += x
        return out

class MJNet(nn.Module):
    def __init__(self, history_len=4, n_cls=34, n_residuals=50):
        super().__init__()
        self.net = self.create_model((history_len+1)*39, n_residuals, n_cls)

    def forward(self, x):
        return self.net(x)

    def create_model(self, in_channels, n_residuals, n_cls):
        # First layer
        module_list = nn.ModuleList([
            conv3x1(in_channels, 256),
            nn.BatchNorm2d(256),
            nn.Dropout2d(0.5),
            nn.LeakyReLU()
        ])
        # Adding residual blocks
        for layer_i in range(n_residuals):
            module_list.append(ResidualBlock(256))

        # Flatten & then fc layers
        module_list.append(nn.Flatten())
        out_feat = 1024
        module_list += nn.ModuleList([
            *self.linear_block(256*34, 1024, dropout_prob=0.2),
            *self.linear_block(1024, 256, dropout_prob=0.2),
            NoisyFactorizedLinear(256, n_cls)
        ])

        return nn.Sequential(*module_list)

    def linear_block(self, n_feat, out_feat, dropout_prob=0.5):
        block = nn.ModuleList([
            nn.Linear(n_feat, out_feat),
            nn.BatchNorm1d(out_feat),
            nn.Dropout(dropout_prob),
            nn.LeakyReLU()
        ])
        return block


class NoisyFactorizedLinear(nn.Linear):
    """
    NoisyNet layer with factorized gaussian noise
    nn.Linear already initializes weight and bias to mu_w and mu_b
    """
    def __init__(self, in_features, out_features, sigma_zero=0.5, bias=True):
        super(NoisyFactorizedLinear, self).__init__(in_features, out_features, bias=bias)
        sigma_init = sigma_zero / math.sqrt(in_features)
        self.sigma_w = nn.Parameter(torch.Tensor(out_features, in_features).fill_(sigma_init))
        self.register_buffer("epsilon_input", torch.zeros(1, in_features))
        self.register_buffer("epsilon_output", torch.zeros(out_features, 1))
        if bias:
            self.sigma_b = nn.Parameter(torch.Tensor(out_features).fill_(sigma_init))

    def forward(self, input):
        bias = self.bias
        func = lambda x: torch.sign(x) * torch.sqrt(torch.abs(x))

        with torch.no_grad():
            torch.randn(self.epsilon_input.size(), out=self.epsilon_input)
            torch.randn(self.epsilon_output.size(), out=self.epsilon_output)
            eps_in = func(self.epsilon_input)
            eps_out = func(self.epsilon_output)  # eps_b
            noise = torch.mul(eps_in, eps_out)  # eps_w
        if bias is not None:
            bias = bias + self.sigma_b * eps_out.t()  # bias = mu_b + sigma_b * eps_b
        return F.linear(input, self.weight + self.sigma_w * noise, bias)  # weight: mu_w


class BaseModel:
    def __init__(self, weight_path=None, device=None):
        if device is None:
            self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        else:
            self.device = device
        self._load_models(weight_path)

    def _load_models(self,weight_path):
        self.model = MJNet()
        if weight_path is not None:
            self.model.load_state_dict(torch.load(weight_path, map_location=torch.device('cpu')))
            print(f'Loaded checkpoint {weight_path.split("/")[-1]}')

        self.model.to(self.device)

    def predict(self, inp):
        '''
        Run forward propagation for all models

        Args:
        - inp (torch.float32): array of size [bs, (hist+1)*39, 34, 1]

        Returns:
        - preds (torch.float32): array of size [bs, n_cls]
        '''
        self.model.eval()
        with torch.no_grad():
            preds = self.model(inp)
        return preds.cpu()


class DQNModel(BaseModel):
    def __init__(self, device=None):
        super().__init__(device)

# print(MJNet())