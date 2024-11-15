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
        # Generate Q, K, V
        Q = self.linear_q(x_q)  # (batch_size, d_model)
        K = self.linear_k(x_kv)  # (batch_size, d_model)
        V = self.linear_v(x_kv)  # (batch_size, d_model)

        # Calculate attention scores
        attention_scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(Q.size(-1))
        attention_weights = F.softmax(attention_scores, dim=-1)

        # Calculate weighted V
        cross_attended_features = torch.matmul(attention_weights, V)  # (batch_size, d_model)

        return cross_attended_features

class DIA(nn.Module):
    def __init__(self, eeg_dim=256, et_dim=177, d_model=256, num_layers=3):
        super(DIA, self).__init__()
        self.d_model = d_model

        self.linear_et = nn.Linear(et_dim, d_model)

        # Gating mechanism linear layers for calculating gate weights
        self.gate_eeg = nn.Linear(d_model, d_model)
        self.gate_et = nn.Linear(d_model, d_model)

        self.cross_attention_layers = nn.ModuleList([
            CrossAttentionLayer(d_model) for _ in range(num_layers)
        ])

        # MLP for the fused output
        self.fusion_mlp = nn.Sequential(
            nn.Linear(2 * d_model, d_model),
            nn.ReLU(),
            nn.Linear(d_model, d_model)
        )

    def forward(self, eeg_input, et_input):
        # Feature transformation
        eeg_transformed = eeg_input
        et_transformed = self.linear_et(et_input)  # (batch_size, d_model)

        residual_output = eeg_transformed  # Initialize residual output

        # Store the output from each layer
        intermediate_outputs = []

        for layer in self.cross_attention_layers:
            # Compute cross-attended features
            cross_attended_features = layer(residual_output.unsqueeze(1), et_transformed.unsqueeze(1))  # (batch_size, 1, d_model)
            gate_et = torch.sigmoid(self.gate_et(cross_attended_features.squeeze(1)))  # (batch_size, d_model)

            # Apply gate weights
            gated_et = cross_attended_features.squeeze(1) * gate_et  # (batch_size, d_model)
            combined = torch.cat((residual_output, gated_et), dim=1)  # Shape (batch_size, 512)
            fused_output = self.fusion_mlp(combined)  # (batch_size, seq_len_eeg, d_model)

            # Residual connection: add the current layer output to the previous layer residual
            residual_output = eeg_transformed + fused_output  # (batch_size, d_model)

            # Store the intermediate layer outputs
            intermediate_outputs.append(residual_output)

        return intermediate_outputs
