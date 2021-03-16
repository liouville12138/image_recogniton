
import torch

class FullConnectNet(torch.nn.Module):
    def __init__(self, input_size, hidden_size, num_classes):
        super(FullConnectNet, self).__init__()
        self.layer1 = torch.nn.Sequential(
            torch.nn.Linear(input_size, hidden_size),
            torch.nn.ReLU(inplace=True)
        )
        self.layer2 = torch.nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        out = self.layer1(x)
        out = self.layer2(out)
        return out
