import torchvision
import torch
from torch import nn
from torchvision.models import resnet18, resnet34, resnet50, resnet101, resnet152,\
    resnext50_32x4d, resnext101_32x8d, wide_resnet50_2, wide_resnet101_2

model_dict = {
    '18': (resnet18, 512),
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
    'wide101': (wide_resnet101_2, 2048)
}

class Resnet(torch.nn.Module):
    def __init__(self, type, class_count=3):
        super(Resnet, self).__init__()
        assert type in model_dict.keys()

        self.class_count = class_count
        self.type = type
        resnet, feat_count = self.get_network(type)
        self.prop_count = feat_count

        self.conv1 = resnet.conv1
        self.bn1 = resnet.bn1
        self.relu = resnet.relu
        self.maxpool = resnet.maxpool
        self.layer1 = resnet.layer1
        self.layer2 = resnet.layer2
        self.layer3 = resnet.layer3
        self.layer4 = resnet.layer4
        self.avgpool = resnet.avgpool
        self.fc = torch.nn.Linear(feat_count, class_count)

        self.fc_classifier = nn.Sequential(
            nn.Dropout(),
            nn.Linear(feat_count, feat_count),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(feat_count, feat_count),
            nn.ReLU(inplace=True),
            nn.Linear(feat_count, class_count),
        )
        self.fc_propensity = nn.Sequential(
            nn.Dropout(),
            nn.Linear(self.prop_count, self.prop_count),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(self.prop_count, self.prop_count),
            nn.ReLU(inplace=True),
            nn.Linear(self.prop_count, class_count),
        )
    
    def get_network(self, type):
        model_func, feat_count = model_dict[type]
        return model_func(pretrained=True), feat_count

    def forward(self, x, s=None):
        # sigmoid = torch.nn.Sigmoid()

        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        #x = self.fc(x)


        if self.type == '51':
            # propensity for cancer is absolute
            propensity = self.fc_propensity(x)
            temp = s[:,2] * 100.
            temp[temp == 0] = -100.
            propensity[:,2] = torch.clone(temp)

            # propensity for normal is 0
            propensity[s == [0,0,0]] = torch.tensor([-100.,-100.,-100.]).float().cuda()

        elif self.type == '52':
            # propensity for cancer is absolute
            propensity = self.fc_propensity(x)
            temp = s[:,2] * 100.
            temp[temp == 0] = -100.
            propensity[:,2] = torch.clone(temp)

        elif self.type == '53':
             # propensity for normal is 0
            propensity = self.fc_propensity(x)
            propensity[s == [0,0,0]] = torch.tensor([-100.,-100.,-100.]).float().cuda()

        else:
            propensity = self.fc_propensity(x)
            
            
        return self.fc_classifier(x), propensity

        

    

