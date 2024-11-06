import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class CrossAttentionLayer(nn.Module):
    def __init__(self, d_model):
        super(CrossAttentionLayer, self).__init__()
        self.linear_q = nn.Linear(d_model, d_model)
        self.linear_k = nn.Linear(d_model, d_model)
        self.linear_v = nn.Linear(d_model, d_model)

    def forward(self, x_q, x_kv):
        # 生成Q、K、V
        Q = self.linear_q(x_q)  # (batch_size, d_model)
        K = self.linear_k(x_kv)  # (batch_size, d_model)
        V = self.linear_v(x_kv)  # (batch_size, d_model)

        # 计算注意力权重
        attention_scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(Q.size(-1))
        attention_weights = F.softmax(attention_scores, dim=-1)

        # 计算加权的V
        cross_attended_features = torch.matmul(attention_weights, V)  # (batch_size, d_model)

        return cross_attended_features

class MLCrossAttentionGating(nn.Module):
    def __init__(self, eeg_dim=256, et_dim=177, d_model=256,num_layers = 3):
        super(MLCrossAttentionGating, self).__init__()
        self.d_model = d_model

        self.linear_eeg = nn.Linear(eeg_dim, d_model)
        self.linear_et = nn.Linear(et_dim, d_model)

        # 门控机制线性层，用于计算门控权重
        self.gate_eeg = nn.Linear(d_model, d_model)
        self.gate_et = nn.Linear(d_model, d_model)

        # 三层独立的交叉注意力模块
        self.cross_attention_layers = nn.ModuleList([
            CrossAttentionLayer(d_model) for _ in range(num_layers)
        ])

        # 融合后的MLP
        self.fusion_mlp = nn.Sequential(
            nn.Linear(2 * d_model, d_model),
            nn.ReLU(),
            nn.Linear(d_model, d_model)
        )

    def forward(self, eeg_input, et_input):
        # 特征映射
        eeg_transformed = eeg_input
        # eeg_transformed = self.linear_eeg(eeg_input)  # (batch_size, d_model)
        et_transformed = self.linear_et(et_input)  # (batch_size, d_model)

        residual_output = eeg_transformed  # 初始化残差输出

        # 保存每一层的输出
        intermediate_outputs = []

        for layer in self.cross_attention_layers:
            # 计算交叉注意力特征
            cross_attended_features = layer(residual_output.unsqueeze(1), et_transformed.unsqueeze(1))  # (batch_size, 1, d_model)

            # 门控机制：对EEG和ET特征进行加权
            # gate_eeg = torch.sigmoid(self.gate_eeg(residual_output))  # (batch_size, d_model)
            gate_et = torch.sigmoid(self.gate_et(cross_attended_features.squeeze(1)))  # (batch_size, d_model)

            # 应用门控权重
            # gated_eeg = residual_output * gate_eeg  # (batch_size, d_model)
            gated_et = cross_attended_features.squeeze(1) * gate_et  # (batch_size, d_model)
            combined = torch.cat((residual_output, gated_et), dim=1)  # 形状为 (batch_size, 512)
            fused_output = self.fusion_mlp(combined)  # (batch_size, seq_len_eeg, d_model)

            # 残差连接：将当前层的输出与上一层的残差相加
            residual_output = eeg_transformed + fused_output  # (batch_size, d_model)

            # 保存中间层的输出
            intermediate_outputs.append(residual_output)

        return intermediate_outputs
