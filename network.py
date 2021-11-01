import torch.nn as nn


def convTwice(channels, kernel_size=3, padding=1):
    return nn.Sequential(
        nn.Conv2d(channels, channels, kernel_size, padding=padding),
        nn.ReLU(),
        nn.Conv2d(channels, channels, kernel_size, padding=padding),
    )


class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv_1_1 = nn.Sequential(
            nn.Conv2d(3, 16, 3, padding=1),
            nn.ReLU())
        self.conv_1_2 = convTwice(16)
        self.conv_1_3 = convTwice(16)
        self.conv_1_4 = convTwice(16)
        self.conv_2_1 = nn.Sequential(
            nn.Conv2d(16, 32, 3, padding=1, stride=2),
            nn.ReLU())
        self.conv_2_2 = convTwice(32)
        self.conv_2_3 = convTwice(32)
        self.conv_2_4 = convTwice(32)
        self.conv_3_1 = nn.Sequential(
            nn.Conv2d(32, 64, 3, padding=1, stride=2),
            nn.ReLU())
        self.conv_3_2 = convTwice(64)
        self.conv_3_3 = convTwice(64)
        self.conv_3_4 = convTwice(64)
        self.conv_4_1 = nn.Sequential(
            nn.ConvTranspose2d(64, 32, 4, padding=1, stride=2),
            nn.ReLU())
        self.conv_4_2 = convTwice(32)
        self.conv_4_3 = convTwice(32)
        self.conv_4_4 = convTwice(32)
        self.conv_5_1 = nn.Sequential(
            nn.ConvTranspose2d(32, 16, 4, padding=1, stride=2),
            nn.ReLU())
        self.conv_5_2 = convTwice(16)
        self.conv_5_3 = convTwice(16)
        self.conv_5_4 = convTwice(16)
        self.conv_6_1 = nn.Sequential(
            nn.Conv2d(16, 3, 3, padding=1),
            nn.Sigmoid()
        )

    def resnetBlock(self, x, layer):
        return x + layer(x)

    def forward(self, x):
        saved_layers = []
        x = self.conv_1_1(x)
        x = self.resnetBlock(x, self.conv_1_2)
        x = self.resnetBlock(x, self.conv_1_3)
        x = self.resnetBlock(x, self.conv_1_4)
        saved_layers.append(x.clone())
        x = self.conv_2_1(x)
        x = self.resnetBlock(x, self.conv_2_2)
        x = self.resnetBlock(x, self.conv_2_3)
        x = self.resnetBlock(x, self.conv_2_4)
        saved_layers.append(x.clone())
        x = self.conv_3_1(x)
        x = self.resnetBlock(x, self.conv_3_2)
        x = self.resnetBlock(x, self.conv_3_3)
        x = self.resnetBlock(x, self.conv_3_4)
        x = self.conv_4_1(x)
        x += saved_layers.pop()
        x = self.resnetBlock(x, self.conv_4_2)
        x = self.resnetBlock(x, self.conv_4_3)
        x = self.resnetBlock(x, self.conv_4_4)
        x = self.conv_5_1(x)
        x += saved_layers.pop()
        x = self.resnetBlock(x, self.conv_5_2)
        x = self.resnetBlock(x, self.conv_5_3)
        x = self.resnetBlock(x, self.conv_5_4)
        x = self.conv_6_1(x)
        return x
