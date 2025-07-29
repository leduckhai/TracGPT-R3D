import torch.nn as nn
import torch.nn.functional as F
import torch 
from typing import List
from collections import OrderedDict

class _DenseLayer3D(nn.Sequential):
    def __init__(self, num_input_features, growth_rate, bn_size, drop_rate):
        super(_DenseLayer3D, self).__init__()
        self.add_module('norm1', nn.BatchNorm3d(num_input_features))
        self.add_module('relu1', nn.ReLU(inplace=True))
        self.add_module('conv1', nn.Conv3d(num_input_features, bn_size*growth_rate,
                                          kernel_size=1, stride=1, bias=False))
        self.add_module('norm2', nn.BatchNorm3d(bn_size*growth_rate))
        self.add_module('relu2', nn.ReLU(inplace=True))
        self.add_module('conv2', nn.Conv3d(bn_size*growth_rate, growth_rate,
                                          kernel_size=3, stride=1, padding=1, bias=False))
        self.drop_rate = drop_rate

    def forward(self, x):
        new_features = super(_DenseLayer3D, self).forward(x)
        if self.drop_rate > 0:
            new_features = F.dropout(new_features, p=self.drop_rate, training=self.training)
        return torch.cat([x, new_features], 1)

class DenseBlock3D(nn.Sequential):
    def __init__(self, num_layers, num_input_features, bn_size, growth_rate, drop_rate):
        super(DenseBlock3D, self).__init__()
        for i in range(num_layers):
            layer = _DenseLayer3D(num_input_features + i*growth_rate,
                                 growth_rate, bn_size, drop_rate)
            self.add_module('denselayer%d' % (i+1), layer)

class DenseNet3D(nn.Module):
    def __init__(self, growth_rate=32, block_config=(6,12,24,16),
                 num_init_features=64, bn_size=4, drop_rate=0, num_classes=1000):
        super(DenseNet3D, self).__init__()
        
        # First convolution
        self.features = nn.Sequential(OrderedDict([
            ('conv0', nn.Conv3d(1, num_init_features, kernel_size=7, stride=2, padding=3, bias=False)),
            ('norm0', nn.BatchNorm3d(num_init_features)),
            ('relu0', nn.ReLU(inplace=True)),
            ('pool0', nn.MaxPool3d(kernel_size=3, stride=2, padding=1)),
        ]))
        
        # Each denseblock
        num_features = num_init_features
        for i, num_layers in enumerate(block_config):
            block = DenseBlock3D(num_layers=num_layers,
                                num_input_features=num_features,
                                bn_size=bn_size,
                                growth_rate=growth_rate,
                                drop_rate=drop_rate)
            self.features.add_module('denseblock%d' % (i+1), block)
            num_features = num_features + num_layers*growth_rate
            if i != len(block_config)-1:
                trans = _Transition3D(num_features, num_features//2)
                self.features.add_module('transition%d' % (i+1), trans)
                num_features = num_features // 2
        
        # Final batch norm
        self.features.add_module('norm5', nn.BatchNorm3d(num_features))
        # print("num_features", num_features)
        # Linear layer
        self.classifier = nn.Linear(num_features, num_classes)
        
    def forward(self, x):
        features = self.features(x)
        out = F.relu(features, inplace=True)
        out = F.adaptive_avg_pool3d(out, (1,1,1))
        out = torch.flatten(out, 1)
        print("out shape", out.shape)
        out = self.classifier(out)
        print("out shape", out.shape)
        return out
class _Transition3D(nn.Sequential):
    def __init__(self, num_input_features, num_output_features):
        super(_Transition3D, self).__init__()
        self.add_module('norm', nn.BatchNorm3d(num_input_features))
        self.add_module('relu', nn.ReLU(inplace=True))
        self.add_module('conv', nn.Conv3d(num_input_features, num_output_features,
                                        kernel_size=1, stride=1, bias=False))
        self.add_module('pool', nn.AvgPool3d(kernel_size=2, stride=2))
    

if __name__=="__main__":
    inp=torch.randn(1,1,32,256,256)
    model=DenseNet3D()
    model.to("cuda")
    inp=inp.to("cuda")
    out=model(inp)
    print(out.shape)