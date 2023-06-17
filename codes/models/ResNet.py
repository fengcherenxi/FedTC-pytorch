import torch.nn as nn
from torchvision import models

backbone = models.resnet18(pretrained=True)
class ResNet18(nn.Module):
    def __init__(self):
        super(ResNet18, self).__init__()
        self.backbone = backbone
        self.fcrelu = nn.ReLU()
        self.fc1 = nn.Linear(1000,10)

    def forward(self, x):
        y = self.backbone(x)
        y = self.fcrelu(y)
        y = self.fc1(y)
        return y
# ourModel = ResNet18()
# print(ourModel)