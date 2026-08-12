import torch
import torch.nn as nn

class AutoEncoder(nn.Module):
    def __init__(self, input_dim, hidden_dim, lambda_l21=0.001):
        self.lambda_l21 = lambda_l21
        super().__init__()

        self.encoder_layer = nn.Linear(input_dim, hidden_dim)

        self.encoder = nn.Sequential(
            self.encoder_layer,
            nn.ReLU()
        )

        self.decoder = nn.Sequential(
            nn.Linear(hidden_dim, input_dim)
        )

    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded
    
    def l21_penalty(self):
        weights = self.encoder_layer.weight
        return torch.norm(weights, dim=0).sum()