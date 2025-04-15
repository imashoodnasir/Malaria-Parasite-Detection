import torch
import torch.nn as nn
import torch.nn.functional as F
from cnn_backbone import CNNBackbone
from capsule_layers import PrimaryCaps, DigitCaps

class HybridCapNet(nn.Module):
    def __init__(self, num_classes=4):
        super(HybridCapNet, self).__init__()
        self.backbone = CNNBackbone()
        self.primary_caps = PrimaryCaps(in_channels=256, num_capsules=8, capsule_dim=16, kernel_size=3, stride=2)
        self.digit_caps = DigitCaps(num_caps_in=288, dim_caps_in=16, num_caps_out=num_classes, dim_caps_out=16, routing_iters=3)
        self.decoder = nn.Sequential(
            nn.Linear(16 * num_classes, 512),
            nn.ReLU(),
            nn.Linear(512, 1024),
            nn.ReLU(),
            nn.Linear(1024, 3 * 224 * 224),
            nn.Sigmoid()
        )

    def forward(self, x):
        features = self.backbone(x)
        primary = self.primary_caps(features)
        digit = self.digit_caps(primary)
        v_length = torch.norm(digit, dim=2)
        _, max_length_indices = v_length.max(dim=1)
        y = F.one_hot(max_length_indices, num_classes=digit.size(1)).float()
        masked = (digit * y.unsqueeze(2)).view(x.size(0), -1)
        recon = self.decoder(masked)
        recon = recon.view(-1, 3, 224, 224)
        return v_length, recon
