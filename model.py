import torch.nn as nn
from Feature.EEGencoder import *
from HAIAM.DIA import *
from Feature.EMencoder import *
class MyModel(nn.Module):
    def __init__(self, eeg_input_dim, eye_input_dim, output_dim):
        super(MyModel, self).__init__()
        self.rescbam = ResCBAM()
        self.ETnet = DenseNet1D()
        self.encoder = MLCrossAttentionGating(eeg_input_dim, eye_input_dim , d_model=output_dim)

    def forward(self, eeg_input, eye_input):
        eeg_features = self.rescbam(eeg_input)
        et_features = self.ETnet(eye_input)
        fusion = self.encoder(eeg_features, et_features)
        return fusion

