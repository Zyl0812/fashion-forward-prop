import torch
import torch.nn as nn
import torch.nn.functional as F


class Denoiser(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 8, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(8, 16, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(16, 32, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv_t1 = nn.ConvTranspose2d(32, 32, kernel_size=2, stride=2)
        self.conv_t2 = nn.ConvTranspose2d(32, 16, kernel_size=2, stride=2)
        self.conv_t3 = nn.ConvTranspose2d(16, 8, kernel_size=2, stride=2)
        self.conv_out = nn.Conv2d(8, 3, kernel_size=3, padding=1)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.pool(F.relu(self.conv3(x)))
        x = F.relu(self.conv_t1(x))
        x = F.relu(self.conv_t2(x))
        x = F.relu(self.conv_t3(x))
        return torch.sigmoid(self.conv_out(x))
