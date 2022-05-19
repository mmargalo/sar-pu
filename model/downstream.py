import torchvision
import torch
from torch import nn
from torchvision.models import resnet18, resnet34, resnet50, resnet101, resnet152,\
    resnext50_32x4d, resnext101_32x8d, wide_resnet50_2, wide_resnet101_2

    
class SimpleFc(torch.nn.Module):
    def __init__(self, feat_count, class_count=3):
        super(SimpleFc, self).__init__()

        self.class_count = class_count
        self.feat_count = int(feat_count)

        self.avgpool = torch.nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Sequential(
            nn.Dropout(),
            nn.Linear(self.feat_count, feat_count),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(self.feat_count, feat_count),
            nn.ReLU(inplace=True),
            nn.Linear(self.feat_count, class_count),
        )

    def forward(self, x):
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        return self.fc(x)
        
class SingleFc(torch.nn.Module):
    def __init__(self, feat_count, class_count=3):
        super(SingleFc, self).__init__()

        self.class_count = class_count
        self.feat_count = int(feat_count)

        self.avgpool = torch.nn.AdaptiveAvgPool2d((1, 1))
        self.fc = torch.nn.Linear(self.feat_count, class_count)
        

    def forward(self, x):
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        return self.fc(x)


class ConvFc(torch.nn.Module):
    def __init__(self, feat_count, class_count=3):
        super(ConvFc, self).__init__()

        self.class_count = class_count
        self.feat_count = int(feat_count)

        base = resnet18(pretrained=True)
        self.conv = base.layer4
        self.avgpool = torch.nn.AdaptiveAvgPool2d((1, 1))
        self.fc = torch.nn.Linear(self.feat_count, class_count)

    def forward(self, x):
        x = self.conv(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        return self.fc(x)

class ConvFc50(torch.nn.Module):
    def __init__(self, feat_count, class_count=3):
        super(ConvFc50, self).__init__()

        self.class_count = class_count
        self.feat_count = int(feat_count)

        base = resnet50(pretrained=True)
        self.conv = base.layer4
        self.avgpool = torch.nn.AdaptiveAvgPool2d((1, 1))
        self.fc = torch.nn.Linear(self.feat_count, class_count)

    def forward(self, x):
        x = self.conv(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        return self.fc(x)        


        

    

