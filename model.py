"""Small 1D-CNN policy: (108 pooled LiDAR ranges + speed) -> (steer, speed)."""
import numpy as np
import torch
import torch.nn as nn
from common import N_FEATURES, STEER_MAX, SPEED_MAX, policy_input


class LidarPolicy(nn.Module):
    def __init__(self, n_beams=N_FEATURES):
        super().__init__()
        self.n_beams = n_beams
        self.conv = nn.Sequential(
            nn.Conv1d(1, 16, 5, stride=2, padding=2), nn.ReLU(),
            nn.Conv1d(16, 32, 5, stride=2, padding=2), nn.ReLU(),
            nn.Conv1d(32, 64, 3, stride=2, padding=1), nn.ReLU(),
        )
        with torch.no_grad():
            flat = self.conv(torch.zeros(1, 1, n_beams)).numel()
        self.head = nn.Sequential(nn.Linear(flat + 1, 128), nn.ReLU(), nn.Linear(128, 2))

    def forward(self, x):                       # x: (B, n_beams + 1)
        scan, speed = x[:, :-1].unsqueeze(1), x[:, -1:]
        z = torch.cat([self.conv(scan).flatten(1), speed], dim=1)
        out = self.head(z)
        return torch.stack([torch.tanh(out[:, 0]), torch.sigmoid(out[:, 1])], dim=1)


class NeuralPolicy:
    """Wraps a trained LidarPolicy with the same .act(scan, speed) interface as FollowTheGap."""
    def __init__(self, path, speed_scale=1.0):
        self.net = LidarPolicy()
        self.net.load_state_dict(torch.load(path, map_location="cpu"))
        self.net.eval()
        self.speed_scale = speed_scale

    @torch.no_grad()
    def act(self, scan, speed):
        x = torch.from_numpy(policy_input(scan, speed)).unsqueeze(0)
        s, v = self.net(x)[0].tolist()
        return s * STEER_MAX, v * SPEED_MAX * self.speed_scale
