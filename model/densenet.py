import torchvision
import torch
from torch import nn
import torch.nn.functional as F
from torchvision.models import densenet121, densenet169, densenet201, densenet161

modes = ['basic', 'full']
model_dict = {
    '121': (densenet121, 1024),
    '169': (densenet169, 1664),
    '201': (densenet201, 1920),
    '161': (densenet161, 2208)
}

class Densenet(torch.nn.Module):
    def __init__(self, type, class_count=3):
        super(Densenet, self).__init__()
        assert type in model_dict.keys()
        
        densenet, feat_count = self.get_network(type)

        self.features = densenet.features
        #self.fc_classifier = torch.nn.Linear(feat_count, class_count)
        #self.fc_propensity = torch.nn.Linear(feat_count, class_count)
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
            nn.Linear(feat_count, feat_count),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(feat_count, feat_count),
            nn.ReLU(inplace=True),
            nn.Linear(feat_count, class_count),
        )
    
    def get_network(self, type):
        model_func, feat_count = model_dict[type]
        return model_func(pretrained=True), feat_count

    def forward(self, x, fc=None):
        #sigmoid = torch.nn.Sigmoid()
        features = self.features(x)
        out = F.relu(features, inplace=True)

        out = F.adaptive_avg_pool2d(out, (1, 1))
        out = torch.flatten(out, 1)
        # out = self.classifier(out)

        if fc == 'classifier':
            return self.fc_classifier(out)
        elif fc == 'propensity':
            return self.fc_propensity(out)
        else:
            return self.fc_classifier(out), self.fc_propensity(out)

