import torch
import torch.nn as nn


class FrequencyPredictor(nn.Module):
    def __init__(self, hidden_dim=128, grid_size=8):
        super().__init__()
        self.grid_size = grid_size

        self.encoder = nn.Sequential(
            nn.Conv2d(3, 64, 7, stride=2, padding=3),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(3, stride=2, padding=1),

            self._res_block(64, 64),
            nn.Conv2d(64, 128, 3, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),

            self._res_block(128, 128),
            nn.AdaptiveAvgPool2d((grid_size, grid_size)),
        )

        self.head = nn.Sequential(
            nn.Conv2d(128, 256, 1),
            nn.ReLU(inplace=True),
            nn.Dropout2d(0.1),
            nn.Conv2d(256, 128, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 3, 1),
            nn.Sigmoid(),
        )

    def _res_block(self, in_c, out_c):
        return nn.Sequential(
            nn.Conv2d(in_c, out_c, 3, padding=1),
            nn.BatchNorm2d(out_c),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_c, out_c, 3, padding=1),
            nn.BatchNorm2d(out_c),
        )

    def forward(self, x):
        x = self.encoder(x)
        return self.head(x)


def create_frequency_predictor(device="cuda", dtype=torch.float16):
    return FrequencyPredictor().to(device=device, dtype=dtype)
