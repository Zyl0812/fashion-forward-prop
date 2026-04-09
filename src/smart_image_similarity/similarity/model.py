import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 16, 3, 1, 1)
        self.conv2 = nn.Conv2d(16, 32, 3, 1, 1)
        self.conv3 = nn.Conv2d(32, 64, 3, 1, 1)
        self.conv4 = nn.Conv2d(64, 128, 3, 1, 1)
        self.conv5 = nn.Conv2d(128, 256, 3, 1, 1)
        self.conv6 = nn.Conv2d(256, 512, 3, 1, 1)
        self.pool = nn.MaxPool2d(2, 2)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.pool(F.relu(self.conv3(x)))
        x = self.pool(F.relu(self.conv4(x)))
        x = self.pool(F.relu(self.conv5(x)))
        x = self.pool(F.relu(self.conv6(x)))
        return x.squeeze(-1).squeeze(-1)


class ConvDecoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv_t1 = nn.ConvTranspose2d(512, 256, 2, 2)
        self.conv_t2 = nn.ConvTranspose2d(256, 128, 2, 2)
        self.conv_t3 = nn.ConvTranspose2d(128, 64, 2, 2)
        self.conv_t4 = nn.ConvTranspose2d(64, 32, 2, 2)
        self.conv_t5 = nn.ConvTranspose2d(32, 16, 2, 2)
        self.conv_t6 = nn.ConvTranspose2d(16, 3, 2, 2)

    def forward(self, x):
        x = x.unsqueeze(-1).unsqueeze(-1)
        x = torch.relu(self.conv_t1(x))
        x = torch.relu(self.conv_t2(x))
        x = torch.relu(self.conv_t3(x))
        x = torch.relu(self.conv_t4(x))
        x = torch.relu(self.conv_t5(x))
        return torch.sigmoid(self.conv_t6(x))
