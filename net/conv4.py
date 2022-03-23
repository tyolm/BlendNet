import torch.nn as nn

def conv_block(in_channels, out_channels):
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, 3, padding=1),
        nn.BatchNorm2d(out_channels),
        nn.ReLU(),
        nn.MaxPool2d(2)
    )


class CNNEncoder(nn.Module):

    def __init__(self):
        super(CNNEncoder, self).__init__()
        # self.encoder = nn.Sequential(
        #     conv_block(x_dim, hid_dim),
        #     conv_block(hid_dim, hid_dim),
        #     conv_block(hid_dim, hid_dim),
        #     conv_block(hid_dim, z_dim),
        # )
        # self.out_channels = 1600
        self.conv_block1 = nn.Sequential(
            nn.Conv2d(3, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )
        self.conv_block2 = nn.Sequential(
            nn.Conv2d(64, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )

    def forward(self, x):
        x = self.conv_block1(x)
        out1 = x
        x = self.conv_block2(x)
        out2 = x
        x = self.conv_block2(x)
        out3 = x
        x = self.conv_block2(x)
        out4 = x
        # return x.view(x.size(0), -1)
        return out1, out2, out3, out4


if __name__ == '__main__':
    import torch

    x = torch.randn(10, 3, 84, 84)
    model = CNNEncoder()
    res = model(x)
    for i in res:
        print(i.shape)
    '''
    [10, 64, 42, 42]
    [10, 128, 21, 21]
    [10, 256, 10, 10]
    [10, 512, 5, 5]
    '''