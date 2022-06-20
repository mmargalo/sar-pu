import torchvision
import torch
from torch import nn
from torchvision.models import resnet18, resnet34, resnet50, resnet101, resnet152,\
    resnext50_32x4d, resnext101_32x8d, wide_resnet50_2, wide_resnet101_2

model_dict = {
    '18': (resnet18, 512),
    '19': (resnet18, 512),
    '34': (resnet34, 512),
    '49': (resnet50, 2048), 
    '50': (resnet50, 2048), 
    '51': (resnet50, 2048), 
    '52': (resnet50, 2048), 
    '53': (resnet50, 2048), 
    '101': (resnet101, 2048),  
    '152': (resnet152, 2048),
    'next50': (resnext50_32x4d, 2048),
    'next101': (resnext101_32x8d, 2048), 
    'wide50': (wide_resnet50_2, 2048),
    'wide101': (wide_resnet101_2, 2048),
    
}

class ResnetFeat(torch.nn.Module):
    def __init__(self, type, class_count=3):
        super(ResnetFeat, self).__init__()
        assert type in model_dict.keys()

        self.class_count = class_count
        self.type = type
        resnet, feat_count = self.get_network(type)

        self.conv1 = resnet.conv1
        self.bn1 = resnet.bn1
        self.relu = resnet.relu
        self.maxpool = resnet.maxpool
        self.layer1 = resnet.layer1
        self.layer2 = resnet.layer2
        self.layer3 = resnet.layer3
        self.layer4 = resnet.layer4
        if self.type in ['49', '19']: self.layer4 = nn.Sequential() 
        
        #self.layer4 = resnet.layer4
        #self.avgpool = resnet.avgpool
    
    def get_network(self, type):
        model_func, feat_count = model_dict[type]
        return model_func(pretrained=True), feat_count

    def forward(self, x, s=None):

        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        if self.type not in ['49', '19']: 
            x = self.layer4(x)
        
        #x = self.avgpool(x)
            
        return x
        
        

    

