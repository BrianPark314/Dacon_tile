#-*- coding:utf-8 -*-

from torchvision import models
from torch import nn, hub


class GoogleNet(nn.Module):
    def __init__(self):
        super(GoogleNet,self).__init__()
        self.classifier = Classifier()
        self.net = models.googlenet(weights = models.GoogLeNet_Weights.DEFAULT)
        for p in self.net.parameters():
            p.requires_grad=False

    def forward(self,x):
        return self.classifier(self.net(x))
    
class ResNet50(nn.Module):
    def __init__(self):
        super(ResNet50,self).__init__()
        self.classifier = Classifier()
        self.net = models.resnet50(weights = models.ResNet50_Weights.DEFAULT)
        for p in self.net.parameters():
            p.requires_grad=False

    def forward(self,x):
        return self.classifier(self.net(x))

class Vgg16(nn.Module):
    def __init__(self):
        super(Vgg16,self).__init__()
        self.layer1 = nn.Linear(1000,19)
        self.net = models.vgg16(weights = models.VGG16_Weights.DEFAULT)
        for p in self.net.parameters():
            p.requires_grad=False

    def forward(self,x):
        return (self.layer1(self.net(x)))
    
class EfficientNet(nn.Module):
    def __init__(self):
        super(EfficientNet, self).__init__()
        self.classifier = Classifier()
        self.net = models.efficientnet_b0(weights = models.EfficientNet_B0_Weights.DEFAULT)
        for p in self.net.parameters():
            p.requires_grad=False
        
    def forward(self,x):
        return (self.classifier(self.net(x)))
    
class Classifier(nn.Module):
    def __init__(self):
        super(Classifier, self).__init__()
        self.layer1 = nn.Linear(1000,512)
        self.Relu1 = nn.ReLU()
        self.Dropout1 = nn.Dropout(p=0.5)
        self.layer2 = nn.Linear(512, 256)
        self.Relu2 = nn.ReLU()
        self.Dropout2 = nn.Dropout(p=0.5)
        self.layer3 = nn.Linear(256, 19)

    def forward(self, x):
        return self.layer3(self.Dropout2(self.Relu2(self.layer2(self.Dropout1(self.Relu1(self.layer1(x)))))))
