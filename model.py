from torch import nn
import torch.nn.functional as F
from torchvision import models

class LeftRightResnet18(nn.Module):
    '''
    Resnet18 model for Left-Right image classification
    '''
    def __init__(self, is_trained):
        super().__init__()
        self.resnet = models.resnet18(pretrained=is_trained)
        kernel_count = self.resnet.fc.in_features
        self.resnet.fc = nn.Sequential(nn.Linear(2560, 2),nn.Sigmoid())

    def forward(self, x):
        x = self.resnet(x)
        return x
