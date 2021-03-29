import torch
import torch.nn as nn

from functools import partial

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
    def __init__(self, history_len, n_cls=4, n_residuals=50):
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
            nn.Linear(256, n_cls)
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