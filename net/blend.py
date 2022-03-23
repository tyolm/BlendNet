import torch
import torch.nn as nn
from torchsummary import summary


class Blend(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(128, 256, 3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(256, 256, 2, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU()
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(256, 128, 2, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU()
        )
        self.conv4 = nn.Sequential(
            nn.Conv2d(128, 64, 2, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU()
        )
        self.conv5 = nn.Sequential(
            nn.Conv2d(64, 64, 2, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU()
        )

        self.output_block = nn.Sequential(
            nn.MaxPool2d(2),
            nn.Conv2d(64, 1, 1, padding=0),
            nn.Flatten()
        )


    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.conv5(x)
        x = self.output_block(x)
        return x


if __name__ == '__main__':
    x = torch.rand(10, 512, 21, 21).cuda()
    model = Blend().cuda()
    print(model(x).shape)
    summary(model, input_size=(512, 42, 42), batch_size=10)
