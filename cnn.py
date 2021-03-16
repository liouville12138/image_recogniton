import torch

class Cnn(torch.nn.Module):
    def __init__(self, in_channels, out_channels):
        super(Cnn, self).__init__()
        conv1_out_channels = 3
        conv1_kernel_size = 5

        self.layer1 = torch.nn.Sequential(
            torch.nn.Conv2d(in_channels=in_channels,
                            out_channels=conv1_out_channels,
                            kernel_size=conv1_kernel_size,
                            stride=[1, 1],  # 卷积核移动步长
                            padding=[1, 1],  # 处理边界时填充0的数量, 默认为0(不填充).
                            dilation=[1, 1],  # 采样间隔数量, 默认为1, 无间隔采样.
                            groups=1,
                            bias=True),
            torch.nn.BatchNorm2d(conv1_out_channels),
            torch.nn.ReLU(inplace=True)
        )

        conv2_in_channels = conv1_out_channels
        conv2_kernel_size = 4
        self.layer2 = torch.nn.Sequential(
            torch.nn.MaxPool2d(kernel_size=2, stride=2)
        )

        self.layer3 = torch.nn.Sequential(
            torch.nn.Conv2d(in_channels=conv2_in_channels,
                            out_channels=out_channels,
                            kernel_size=conv2_kernel_size,
                            stride=[1, 1],  # 卷积核移动步长
                            padding=[1, 1],  # 处理边界时填充0的数量, 默认为0(不填充).
                            dilation=[1, 1],  # 采样间隔数量, 默认为1, 无间隔采样.
                            groups=1,
                            bias=True),
            torch.nn.BatchNorm2d(out_channels),
            torch.nn.ReLU(inplace=True)
        )

        self.layer4 = torch.nn.Sequential(
            torch.nn.MaxPool2d(kernel_size=2, stride=2)
        )

    def layer(self):
        layer = torch.nn.Sequential(
            self.layer1,
            self.layer2,
            self.layer3,
            self.layer4,
        )
        return layer

    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        return x
