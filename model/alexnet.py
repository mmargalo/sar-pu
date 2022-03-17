import torchvision
import torch
import torch.nn.functional as F
from torch import nn

class Alexnet(torch.nn.Module):
    def __init__(self, class_count=3):
        super(Alexnet, self).__init__()
        
        alexnet = torchvision.models.alexnet(pretrained=True)
        self.features = alexnet.features
        self.avgpool = alexnet.avgpool
        self.fc_classifier = nn.Sequential(
            nn.Dropout(),
            nn.Linear(256 * 6 * 6, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Linear(4096, class_count),
        )
        self.fc_propensity = nn.Sequential(
            nn.Dropout(),
            nn.Linear(256 * 6 * 6, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Linear(4096, class_count),
        )

    def forward(self, x, fc=None):
        # sigmoid = torch.nn.Sigmoid()
        x = self.features(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        # x = self.classifier(x)

        if fc == 'classifier':
            self.fc_classifier(x)
        elif fc == 'propensity':
            self.fc_propensity(x)
        else:
            self.fc_classifier(x), self.fc_propensity(x)

