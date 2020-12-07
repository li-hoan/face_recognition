#构建特征提取网络和特征分类网络
from torchvision import models
from torch import nn
import torch
from arf import *

class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.layer1 = models.densenet121(pretrained=True)
        self.layer2 = Arc(1000,3)
    def forward(self,x):
        feature = self.layer1(x)
        cls = self.layer2(feature)
        return feature,cls
if __name__ == '__main__':
    data = torch.Tensor(1,3,100,100)
    net = Net()
    print(type(net(data)[0]))