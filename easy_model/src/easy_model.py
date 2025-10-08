from torch import nn
from collections import OrderedDict

class EasyModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.model = nn.Sequential(OrderedDict([
            ('conv1',    nn.Conv2d(3, 32, 5, padding=2)),
            ('maxpool1', nn.MaxPool2d(2)),
            ('conv2',    nn.Conv2d(32, 32, 5, padding=2)),
            ('maxpool2', nn.MaxPool2d(2)),
            ('conv3',    nn.Conv2d(32, 64, 5, padding=2)),
            ('maxpool3', nn.MaxPool2d(2)),
            ('flatten',  nn.Flatten()),
            ('fc1',      nn.Linear(64 * 4 * 4, 64)),
            ('fc2',      nn.Linear(64, 10))
        ]))

    def forward(self, x):
        return self.model(x)