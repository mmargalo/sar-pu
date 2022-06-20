from .resnet import Resnet
from .densenet import Densenet
from .alexnet import Alexnet
from .vggnet import Vggnet
from .resnet_feat import ResnetFeat
from .downstream import SimpleFc, SingleFc, ConvFc, ConvFc50
from csra import ResnetCSRA

model_dict = {
    'alexnet': Alexnet,
    'densenet': Densenet,
    'resnet': Resnet,
    'vggnet': Vggnet,
    'resnetfeat': ResnetFeat,
    'singlefc': SingleFc,
    'simplefc': SimpleFc,
    'convfc': ConvFc,
    'convfc50': ConvFc50,
    'csraresnet': ResnetCSRA
}

def get_model(base_model, type, class_count, **kwargs):
    base_model = base_model.lower()
    type = type.lower()

    assert base_model in model_dict.keys()
    model = model_dict[base_model]
    return model(type, class_count, kwargs)
