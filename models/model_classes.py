#-*- coding:utf-8 -*-

import torchvision
from torch import nn, hub


class GoogleNet(nn.Module):
    def __init__(self):
        super(GoogleNet,self).__init__()
        self.layer1 = nn.Linear(1000,512)
        self.Relu1 = nn.ReLU()
        self.Dropout1 = nn.Dropout(p=0.5)
        self.layer2 = nn.Linear(512, 256)
        self.Relu2 = nn.ReLU()
        self.Dropout2 = nn.Dropout(p=0.5)
        self.layer3 = nn.Linear(256, 19)
        self.net = torchvision.models.googlenet(weights = torchvision.models.GoogLeNet_Weights.DEFAULT)
        for p in self.net.parameters():
            p.requires_grad=False

    def forward(self,x):
        return self.layer3(self.Dropout2(self.Relu2(self.layer2(self.Dropout1(self.Relu1(self.layer1(self.net(x))))))))
    
class ResNet50(nn.Module):
    def __init__(self):
        super(ResNet50,self).__init__()
        self.layer1 = nn.Linear(1000,512)
        self.Relu1 = nn.ReLU()
        self.Dropout1 = nn.Dropout(p=0.45)
        self.layer2 = nn.Linear(512, 256)
        self.Relu2 = nn.ReLU()
        self.Dropout2 = nn.Dropout(p=0.45)
        self.layer3 = nn.Linear(256, 19)
        self.net = torchvision.models.resnet50(weights = torchvision.models.ResNet50_Weights.DEFAULT)
        for p in self.net.parameters():
            p.requires_grad=False

    def forward(self,x):
        return self.layer3(self.Dropout2(self.Relu2(self.layer2(self.Dropout1(self.Relu1(self.layer1(self.net(x))))))))

class Vgg16(nn.Module):
    def __init__(self):
        super(Vgg16,self).__init__()
        self.layer1 = nn.Linear(1000,19)
        self.net = torchvision.models.vgg16(weights = torchvision.models.VGG16_Weights.DEFAULT)
        for p in self.net.parameters():
            p.requires_grad=False

    def forward(self,x):
        return (self.layer1(self.net(x)))
    
class Efficientnet(nn.Module):
    def __init__(self):
        super(Efficientnet, self).__init__()
        self.layer1 = nn.Linear(1000, 19)
        self.net = hub.load('NVIDIA/DeepLearningExamples:torchhub', 'nvidia_efficientnet_b0', pretrained=True)
        for p in self.net.parameters():
            p.requires_grad=False
        
    def foward(self,x):
        return (self.layer1(self.net(x)))