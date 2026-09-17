import torch.nn as nn


class SpectraShieldCNN(nn.Module):
    def __init__(self):
        super(SpectraShieldCNN, self).__init__()

        self.features = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),      # 0
            nn.BatchNorm2d(32),                               # 1
            nn.ReLU(),                                        # 2
            nn.MaxPool2d(2),                                  # 3

            nn.Conv2d(32, 64, kernel_size=3, padding=1),     # 4
            nn.BatchNorm2d(64),                               # 5
            nn.ReLU(),                                        # 6
            nn.MaxPool2d(2),                                  # 7

            nn.Conv2d(64, 128, kernel_size=3, padding=1),    # 8
            nn.BatchNorm2d(128),                              # 9
            nn.ReLU(),                                        # 10
            nn.MaxPool2d(2),                                  # 11

            nn.Conv2d(128, 256, kernel_size=3, padding=1),   # 12
            nn.BatchNorm2d(256),                              # 13
            nn.ReLU(),                                        # 14
            nn.AdaptiveAvgPool2d((1, 1)),                    # 15
        )

        self.classifier = nn.Sequential(
            nn.Dropout(0.4),                                  # 0
            nn.Identity(),                                    # 1
            nn.Linear(256, 1),                                # 2
        )

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x