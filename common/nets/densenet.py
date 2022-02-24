import torch
import torch.nn as nn
from torchvision.models import densenet121, densenet161, densenet169, densenet201
from torchvision.models import DenseNet
from torch.nn import functional as F

model_urls = {
    'densenet121': 'https://download.pytorch.org/models/densenet121-a639ec97.pth',
    'densenet169': 'https://download.pytorch.org/models/densenet169-b2777c0a.pth',
    'densenet201': 'https://download.pytorch.org/models/densenet201-c1103571.pth',
    'densenet161': 'https://download.pytorch.org/models/densenet161-8d451a50.pth',
}
class DenseNetBackbone(nn.Module):

    def __init__(self, densenet_type):
	
        super(DenseNetBackbone, self).__init__()

        densenet_spec = {121: (densenet121, 32, (6, 12, 24, 16), 64, 'densenet121'),
		       161: (densenet161, 48, (6, 12, 36, 24), 96, 'densenet161'),
		       169: (densenet169,  32, (6, 12, 32, 32), 64, 'densenet169'),
		       201: (densenet201, 32, (6, 12, 48, 32), 64, 'densenet201'),}
        densenet, growth_rate, block_config,num_init_features, name = densenet_spec[densenet_type]
        self.feat_layer = densenet(pretrain = True).features()
        self.name = name
        self.inplanes = 64
        num_features = num_init_features
        for i, num_layers in enumerate(block_config):
            num_features = num_features + num_layers * growth_rate
            if i != len(block_config) - 1:
                num_features = num_features // 2

        self.conv1 = nn.Conv2d(num_features,2048, kernel_size=1, stride=1, padding=1,
                               bias=False)
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                # nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                nn.init.normal_(m.weight, mean=0, std=0.001)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x, skip_early=False):
        # densenet to get features 
        img_feats = self.feat_layer(x)

        # use 1*1 conv to make output dimension as 2048
        img_feats = self.conv1(img_feats)

        # x here to cooperate original codes. Delete it in future work
        return x,img_feats

    def init_weights(self):
        org_densenet = torch.utils.model_zoo.load_url(model_urls[self.name])
        # drop orginal resnet fc layer, add 'None' in case of no fc layer, that will raise error
        # org_resnet.pop('fc.weight', None)
        # org_resnet.pop('fc.bias', None)
        
        self.load_state_dict(org_densenet)
        print("Initialize densenet from model zoo")


