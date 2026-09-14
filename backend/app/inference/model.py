import torch
import torch.nn as nn


class SpectraShieldCNN(nn.Module):
    def __init__(self):
        super(SpectraShieldCNN, self).__init__()

        self.conv1 = nn.Conv2d(
            1, 32, kernel_size=3, padding=1
        )
        self.bn1 = nn.BatchNorm2d(32)

        self.conv2 = nn.Conv2d(
            32, 64, kernel_size=3, padding=1
        )
        self.bn2 = nn.BatchNorm2d(64)

        self.conv3 = nn.Conv2d(
            64, 128, kernel_size=3, padding=1
        )
        self.bn3 = nn.BatchNorm2d(128)

        self.conv4 = nn.Conv2d(
            128, 256, kernel_size=3, padding=1
        )
        self.bn4 = nn.BatchNorm2d(256)

        self.pool = nn.MaxPool2d(2)

        self.adaptive_pool = nn.AdaptiveAvgPool2d((1, 1))

        self.dropout = nn.Dropout(0.4)

        self.fc = nn.Linear(256, 1)

    def forward(self, x):
        x = self.pool(
            torch.relu(self.bn1(self.conv1(x)))
        )

        x = self.pool(
            torch.relu(self.bn2(self.conv2(x)))
        )

        x = self.pool(
            torch.relu(self.bn3(self.conv3(x)))
        )

        x = torch.relu(
            self.bn4(self.conv4(x))
        )

        x = self.adaptive_pool(x)

        x = x.view(x.size(0), -1)

        x = self.dropout(x)

        x = self.fc(x)

        return x