import torch
import torch.nn as nn


class DenseBlock1D(nn.Module):
    def __init__(self, input_channels, growth_rate, num_layers):
        super(DenseBlock1D, self).__init__()
        self.layers = nn.ModuleList()
        for i in range(num_layers):
            self.layers.append(self._make_layer(input_channels + i * growth_rate, growth_rate))

    def _make_layer(self, in_channels, out_channels):
        layer = nn.Sequential(
            nn.BatchNorm1d(in_channels),
            nn.ReLU(),
            nn.Conv1d(in_channels, out_channels, kernel_size=3, padding=1)
        )
        return layer

    def forward(self, x):
        features = [x]
        for layer in self.layers:
            new_feature = layer(torch.cat(features, dim=1))
            features.append(new_feature)
        return torch.cat(features, dim=1)


class DenseNet1D(nn.Module):
    def __init__(self, input_channels = 33, growth_rate = 12, block_layers = [4,4,4]):
        super(DenseNet1D, self).__init__()
        self.dense_blocks = nn.ModuleList()
        num_channels = input_channels
        for num_layers in block_layers:
            self.dense_blocks.append(DenseBlock1D(num_channels, growth_rate, num_layers))
            num_channels += num_layers * growth_rate
            self.dense_blocks.append(nn.Conv1d(num_channels, num_channels, kernel_size=1))
        self.global_pool = nn.AdaptiveAvgPool1d(1)


    def forward(self, x):
        x = x.unsqueeze(2)
        for block in self.dense_blocks:
            x = block(x)
        x = self.global_pool(x).squeeze(-1)
        return x



