import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from .rescnn import *
from .Cross_Att import *
from .ETnet import *
class MyModel(nn.Module):
    def __init__(self, eeg_input_dim, eye_input_dim, output_dim):
        super(MyModel, self).__init__()
        self.rescbam = ResCBAM()  # ResCBAM 模块
        self.ETnet = DenseNet1D()
        self.encoder = MLCrossAttentionGating(eeg_input_dim, eye_input_dim , d_model=output_dim)

    def forward(self, eeg_input, eye_input):
        # 将 EEG 和眼动数据输入 ResCBAM 和编码器
        eeg_features = self.rescbam(eeg_input)
        et_features = self.ETnet(eye_input)
        fusion = self.encoder(eeg_features, et_features)
        return fusion  # 返回组合后的特征

