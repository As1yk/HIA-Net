import torch.nn as nn
import torch


def basic_block(in_channels, out_channels):
    '''
    Returns a block of conv-bn-relu-pool
    '''
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, 3, padding=1),
        nn.BatchNorm2d(out_channels),
        nn.ReLU(inplace=False)
    )

class ChannelAttention(nn.Module):
    def __init__(self, in_planes, ratio=2):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        self.fc1   = nn.Conv2d(in_planes, in_planes // ratio, 1, bias=False)
        self.relu1 = nn.ReLU(inplace=False)
        self.fc2   = nn.Conv2d(in_planes // ratio, in_planes, 1, bias=False)

        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.fc2(self.relu1(self.fc1(self.avg_pool(x)))))
        max_out = self.fc2(self.relu1(self.fc1(self.max_pool(x)))))
        out = avg_out + max_out
        return self.sigmoid(out)

class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=3):
        super(SpatialAttention, self).__init__()
        self.conv1 = nn.Conv2d(2, 1, kernel_size, padding=1, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = torch.cat([avg_out, max_out], dim=1)
        x = self.conv1(x)
        return self.sigmoid(x)


class ResCBAM(nn.Module):
    '''
    Feature Extraction Model
    Model as described in the reference paper,
    source: https://github.com/jakesnell/prototypical-networks/blob/f0c48808e496989d01db59f86d4449d7aee9ab0c/protonets/models/few_shot.py#L62-L84
    The output of the backbone is a flattened feature vector, not softmax probabilities
    '''
    def __init__(self, input_channels=5, hid_channels=64, output_dim=256):
        super(ResCBAM, self).__init__()
        self.input_channels = input_channels
        self.rb1 = nn.Sequential(
            basic_block(input_channels, hid_channels),
            basic_block(hid_channels, hid_channels)
        )
        self.res1 = basic_block(input_channels, hid_channels)

        self.ca1 = ChannelAttention(hid_channels)
        self.sa1 = SpatialAttention()

        self.bn1 = nn.BatchNorm2d(hid_channels)
        self.maxpool = nn.MaxPool2d(2)
        self.linear = nn.Linear(1024, output_dim)


    def forward(self, x):
        assert x.shape[1] == self.input_channels

        # Retain the original input to avoid in-place modification
        x_res = self.res1(x)
        x_rb1 = self.rb1(x)
        x_sum = x_rb1 + x_res

        # Channel attention and spatial attention modules
        x_ca1 = self.ca1(x_sum) * x_sum
        x_sa1 = self.sa1(x_ca1) * x_ca1

        # Batch normalization and pooling
        x_bn1 = self.bn1(x_sa1)
        x_pooled = self.maxpool(x_bn1)

        # Flatten the feature map into a single feature vector
        x_flat = x_pooled.view(x_pooled.size(0), -1)
        output = self.linear(x_flat)

        return output
