import torch.nn as nn


class Classifier(nn.Module):
    def __init__(self, n_classes=5):
        super().__init__()
        self.model = nn.Sequential(
            nn.Conv2d(3, 8, 3, 1, 1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(8, 16, 3, 1, 1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Flatten(),
            nn.Linear(4096, n_classes),
        )

    def forward(self, x):
        return self.model(x)
