import torch
import torch.nn as nn
import torch.nn.functional as F

class Squash(nn.Module):
    def forward(self, x):
        norm = torch.norm(x, dim=-1, keepdim=True)
        scale = (norm**2) / (1 + norm**2)
        return scale * (x / (norm + 1e-8))

class PrimaryCaps(nn.Module):
    def __init__(self, in_channels, num_capsules, capsule_dim, kernel_size, stride):
        super(PrimaryCaps, self).__init__()
        self.capsules = nn.Conv2d(in_channels, num_capsules * capsule_dim, kernel_size=kernel_size, stride=stride, padding=0)
        self.num_capsules = num_capsules
        self.capsule_dim = capsule_dim
        self.squash = Squash()

    def forward(self, x):
        out = self.capsules(x)
        batch_size = out.size(0)
        out = out.view(batch_size, self.num_capsules, self.capsule_dim, -1)
        out = out.permute(0, 3, 1, 2).contiguous()
        out = out.view(batch_size, -1, self.capsule_dim)
        return self.squash(out)

class DigitCaps(nn.Module):
    def __init__(self, num_caps_in, dim_caps_in, num_caps_out, dim_caps_out, routing_iters=3):
        super(DigitCaps, self).__init__()
        self.num_caps_in = num_caps_in
        self.dim_caps_in = dim_caps_in
        self.num_caps_out = num_caps_out
        self.dim_caps_out = dim_caps_out
        self.routing_iters = routing_iters
        self.W = nn.Parameter(torch.randn(1, num_caps_in, num_caps_out, dim_caps_out, dim_caps_in))
        self.squash = Squash()

    def forward(self, x):
        batch_size = x.size(0)
        x = x.unsqueeze(2).unsqueeze(4)
        W = self.W.repeat(batch_size, 1, 1, 1, 1)
        u_hat = torch.matmul(W, x)
        b_ij = torch.zeros(batch_size, self.num_caps_in, self.num_caps_out, 1, device=x.device)
        for r in range(self.routing_iters):
            c_ij = F.softmax(b_ij, dim=2)
            s_j = (c_ij * u_hat).sum(dim=1, keepdim=True)
            v_j = self.squash(s_j)
            if r < self.routing_iters - 1:
                b_ij = b_ij + (u_hat * v_j).sum(-1, keepdim=True)
        return v_j.squeeze(1).squeeze(-1)
