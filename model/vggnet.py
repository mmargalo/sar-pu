import torchvision
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import vgg11, vgg11_bn, vgg13, vgg13_bn, vgg16, vgg16_bn,\
    vgg19, vgg19_bn

modes = ['basic', 'full']
model_dict = {
    '11': (vgg11, 4096),
    '11bn': (vgg11_bn, 4096),
    '13': (vgg13, 4096),
    '13bn': (vgg13_bn, 4096),
    '16': (vgg16, 4096),
    '16bn': (vgg16_bn, 4096),
    '19': (vgg19, 4096),
    '19bn': (vgg19_bn, 4096)
}

class Vggnet(torch.nn.Module):
    def __init__(self, type, class_count=3):
        super(Vggnet, self).__init__()
        assert type in model_dict.keys()
        
        vggnet, feat_count = self.get_network(type)

        self.features = vggnet.features
        self.avgpool = vggnet.avgpool
        self.fc_classifier = nn.Sequential(
            nn.Linear(512 * 7 * 7, 4096),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(feat_count, class_count)
        )
        self.fc_propensity = nn.Sequential(
            nn.Linear(512 * 7 * 7, 4096),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(feat_count, class_count)
        )
    
    def get_network(self, type):
        model_func, feat_count = model_dict[type]
        return model_func(pretrained=True), feat_count

    def forward(self, x, fc=None):
        return_tuple = []
        x = self.features(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        # x = self.classifier(x)

        if fc == 'classifier':
            return self.fc_classifier(x)
        elif fc == 'propensity':
            return self.fc_propensity(x)
        else:
            return self.fc_classifier(x), self.fc_propensity(x)

