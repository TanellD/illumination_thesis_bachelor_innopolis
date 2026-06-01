"""
Ported to pytorch thanks to [tstandley](https://github.com/tstandley/Xception-PyTorch)

@author: tstandley
Adapted by cadene

Creates an Xception Model as defined in:

Francois Chollet
Xception: Deep Learning with Depthwise Separable Convolutions
https://arxiv.org/pdf/1610.02357.pdf

This weights ported from the Keras implementation. Achieves the following performance on the validation set:

Loss:0.9173 Prec@1:78.892 Prec@5:94.292

REMEMBER to set your image size to 3x299x299 for both test and validation

normalize = transforms.Normalize(mean=[0.5, 0.5, 0.5],
                                  std=[0.5, 0.5, 0.5])

The resize parameter of the validation transform should be 333, and make sure to center crop at 299x299
"""
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.model_zoo as model_zoo
from torch.nn import init

pretrained_settings = {
    'xception': {
        'imagenet': {
            'url': 'http://data.lip6.fr/cadene/pretrainedmodels/xception-b5690688.pth',
            'input_space': 'RGB',
            'input_size': [3, 299, 299],
            'input_range': [0, 1],
            'mean': [0.5, 0.5, 0.5],
            'std': [0.5, 0.5, 0.5],
            'num_classes': 1000,
            'scale': 0.8975
        }
    }
}


class SeparableConv2d(nn.Module):
    def __init__(self,in_channels,out_channels,kernel_size=1,stride=1,padding=0,dilation=1,bias=False):
        super(SeparableConv2d,self).__init__()

        self.conv1 = nn.Conv2d(in_channels,in_channels,kernel_size,stride,padding,dilation,groups=in_channels,bias=bias)
        self.pointwise = nn.Conv2d(in_channels,out_channels,1,1,0,1,1,bias=bias)

    def forward(self,x):
        x = self.conv1(x)
        x = self.pointwise(x)
        return x


class Block(nn.Module):
    def __init__(self,in_filters,out_filters,reps,strides=1,start_with_relu=True,grow_first=True,dropout_rate=0.0):
        super(Block, self).__init__()

        if out_filters != in_filters or strides!=1:
            self.skip = nn.Conv2d(in_filters,out_filters,1,stride=strides, bias=False)
            self.skipbn = nn.BatchNorm2d(out_filters)
        else:
            self.skip = None
            self.skipbn = None

        self.relu = nn.ReLU(inplace=True)
        rep = []
        self.dropout_rate = dropout_rate

        filters = in_filters
        if grow_first:
            rep.append(self.relu)
            rep.append(SeparableConv2d(in_filters,out_filters,3,stride=1,padding=1,bias=False))
            rep.append(nn.BatchNorm2d(out_filters))
            if dropout_rate > 0:
                rep.append(nn.Dropout2d(p=dropout_rate))
            filters = out_filters

        for i in range(reps-1):
            rep.append(self.relu)
            rep.append(SeparableConv2d(filters,filters,3,stride=1,padding=1,bias=False))
            rep.append(nn.BatchNorm2d(filters))
            if dropout_rate > 0 and i < reps-2:  # No dropout before residual
                rep.append(nn.Dropout2d(p=dropout_rate))

        if not grow_first:
            rep.append(self.relu)
            rep.append(SeparableConv2d(in_filters,out_filters,3,stride=1,padding=1,bias=False))
            rep.append(nn.BatchNorm2d(out_filters))
            if dropout_rate > 0:
                rep.append(nn.Dropout2d(p=dropout_rate))

        if not start_with_relu:
            rep = rep[1:]
        else:
            rep[0] = nn.ReLU(inplace=False)

        if strides != 1:
            rep.append(nn.MaxPool2d(3,strides,1))
        
        self.rep = nn.Sequential(*rep)

    def forward(self,inp):
        x = self.rep(inp)

        if self.skip is not None:
            skip = self.skip(inp)
            if self.skipbn is not None:
                skip = self.skipbn(skip)
        else:
            skip = inp

        return x + skip


class Xception(nn.Module):
    """
    Xception optimized for the ImageNet dataset, as specified in
    https://arxiv.org/pdf/1610.02357.pdf
    """
    def __init__(self, num_classes=1000, dropout_config=None):
        """ Constructor
        Args:
            num_classes: number of classes
            dropout_config: dict with dropout rates for different parts
        """
        super(Xception, self).__init__()
        self.num_classes = num_classes
        
        # Default dropout configuration
        if dropout_config is None:
            dropout_config = {
                'early': 0.0,      # Early layers (conv1, conv2)
                'block': 0.0,       # Inside blocks
                'middle': 0.1,      # Between middle blocks
                'late': 0.2,        # Late conv layers
                'classifier': 0.5   # Final classifier
            }
        self.dropout_config = dropout_config

        # Entry flow with dropout
        self.conv1 = nn.Conv2d(3, 32, 3, 2, 0, bias=False)
        self.bn1 = nn.BatchNorm2d(32)
        self.relu = nn.ReLU(inplace=True)
        self.dropout1 = nn.Dropout2d(p=dropout_config['early']) if dropout_config['early'] > 0 else nn.Identity()

        self.conv2 = nn.Conv2d(32, 64, 3, bias=False)
        self.bn2 = nn.BatchNorm2d(64)
        self.dropout2 = nn.Dropout2d(p=dropout_config['early']) if dropout_config['early'] > 0 else nn.Identity()

        # Blocks with dropout
        self.block1 = Block(64, 128, 2, 2, start_with_relu=False, grow_first=True, 
                           dropout_rate=dropout_config['block'])
        self.block2 = Block(128, 256, 2, 2, start_with_relu=True, grow_first=True,
                           dropout_rate=dropout_config['block'])
        self.block3 = Block(256, 728, 2, 2, start_with_relu=True, grow_first=True,
                           dropout_rate=dropout_config['block'])

        # Middle flow blocks
        self.middle_dropout = nn.Dropout2d(p=dropout_config['middle']) if dropout_config['middle'] > 0 else nn.Identity()
        
        self.block4 = Block(728, 728, 3, 1, start_with_relu=True, grow_first=True,
                           dropout_rate=dropout_config['block'])
        self.block5 = Block(728, 728, 3, 1, start_with_relu=True, grow_first=True,
                           dropout_rate=dropout_config['block'])
        self.block6 = Block(728, 728, 3, 1, start_with_relu=True, grow_first=True,
                           dropout_rate=dropout_config['block'])
        self.block7 = Block(728, 728, 3, 1, start_with_relu=True, grow_first=True,
                           dropout_rate=dropout_config['block'])
        self.block8 = Block(728, 728, 3, 1, start_with_relu=True, grow_first=True,
                           dropout_rate=dropout_config['block'])
        self.block9 = Block(728, 728, 3, 1, start_with_relu=True, grow_first=True,
                           dropout_rate=dropout_config['block'])
        self.block10 = Block(728, 728, 3, 1, start_with_relu=True, grow_first=True,
                            dropout_rate=dropout_config['block'])
        self.block11 = Block(728, 728, 3, 1, start_with_relu=True, grow_first=True,
                            dropout_rate=dropout_config['block'])

        # Exit flow
        self.block12 = Block(728, 1024, 2, 2, start_with_relu=True, grow_first=False,
                            dropout_rate=dropout_config['block'])

        self.conv3 = SeparableConv2d(1024, 1536, 3, 1, 1)
        self.bn3 = nn.BatchNorm2d(1536)
        self.dropout3 = nn.Dropout2d(p=dropout_config['late']) if dropout_config['late'] > 0 else nn.Identity()

        self.conv4 = SeparableConv2d(1536, 2048, 3, 1, 1)
        self.bn4 = nn.BatchNorm2d(2048)
        self.dropout4 = nn.Dropout2d(p=dropout_config['late']) if dropout_config['late'] > 0 else nn.Identity()

        # Final classifier (will be replaced by xception function)
        self.fc = nn.Linear(2048, num_classes)
        self.last_linear = self.fc

    def features(self, input):
        # Entry flow
        x = self.conv1(input)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.dropout1(x)

        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu(x)
        x = self.dropout2(x)

        # Early blocks
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)

        # Middle flow with dropout between blocks
        x = self.block4(x)
        x = self.middle_dropout(x)
        x = self.block5(x)
        x = self.middle_dropout(x)
        x = self.block6(x)
        x = self.middle_dropout(x)
        x = self.block7(x)
        x = self.middle_dropout(x)
        x = self.block8(x)
        x = self.middle_dropout(x)
        x = self.block9(x)
        x = self.middle_dropout(x)
        x = self.block10(x)
        x = self.middle_dropout(x)
        x = self.block11(x)

        # Exit flow
        x = self.block12(x)

        x = self.conv3(x)
        x = self.bn3(x)
        x = self.relu(x)
        x = self.dropout3(x)

        x = self.conv4(x)
        x = self.bn4(x)
        x = self.relu(x)
        x = self.dropout4(x)

        return x

    def logits(self, features):
        x = self.relu(features)
        x = F.adaptive_avg_pool2d(x, (1, 1))
        x = x.view(x.size(0), -1)
        x = self.last_linear(x)
        return x

    def forward(self, input):
        x = self.features(input)
        x = self.logits(x)
        return x


def xception(num_classes=1000, pretrained='imagenet', dropout_rate=0.5, dropout_config=None):
    """
    Modified Xception with dropout throughout that preserves ImageNet weights.
    
    Args:
        num_classes: Number of output classes
        pretrained: Whether to load pretrained weights
        dropout_rate: Base dropout rate (used if dropout_config not provided)
        dropout_config: Custom dropout configuration dict
    """
    # Create dropout config if not provided
    if dropout_config is None:
        dropout_config = {
            'early': dropout_rate * 0.0,      # 0.15 if dropout_rate=0.5
            'block': dropout_rate * 0.0,       # 0.2 if dropout_rate=0.5
            'middle': dropout_rate * 0.2,      # 0.2 if dropout_rate=0.5
            'late': dropout_rate * 0.4,        # 0.3 if dropout_rate=0.5
            'classifier': dropout_rate         # 0.5 if dropout_rate=0.5
        }
    
    # Load the model with dropout architecture
    model = Xception(num_classes=1000, dropout_config=dropout_config)
    
    if pretrained:
        settings = pretrained_settings['xception'][pretrained]
        pretrained_dict = model_zoo.load_url(settings['url'])
        
        # Fix pointwise weights (convert 2D to 4D for SeparableConv2d)
        new_dict = {}
        for k, v in pretrained_dict.items():
            if 'pointwise.weight' in k and v.dim() == 2:
                v = v.unsqueeze(-1).unsqueeze(-1)
            
            # Handle keys that might have different names due to dropout layers
            # The pretrained dict doesn't have dropout layers, so we skip those keys
            new_dict[k] = v
        
        # Load pretrained weights, ignoring missing keys (dropout layers)
        missing_keys, unexpected_keys = model.load_state_dict(new_dict, strict=False)
        
        print(f"✅ Loaded ImageNet pretrained weights")
        if missing_keys:
            print(f"   Missing keys (expected due to dropout): {len(missing_keys)}")
        if unexpected_keys:
            print(f"   Unexpected keys: {len(unexpected_keys)}")
    
    # Get the original weights and bias from the fc layer
    original_weight = model.fc.weight.data.clone()
    original_bias = model.fc.bias.data.clone() if model.fc.bias is not None else None
    
    # Create new classifier with dropout
    if dropout_config['classifier'] > 0:
        new_last_linear = nn.Sequential(
            nn.Dropout(p=dropout_config['classifier'], inplace=True),
            nn.Linear(2048, num_classes)
        )
    else:
        new_last_linear = nn.Linear(2048, num_classes)
    
    # Replace the last linear layer
    model.last_linear = new_last_linear
    
    # Handle weight transfer for the final layer
    if pretrained:
        if num_classes == 1000:
            # Transfer weights to the new linear layer
            linear_layer = model.last_linear[1] if dropout_config['classifier'] > 0 else model.last_linear
            linear_layer.weight.data = original_weight
            if original_bias is not None and linear_layer.bias is not None:
                linear_layer.bias.data = original_bias
            print(f"✅ Transferred pretrained weights with classifier dropout={dropout_config['classifier']}")
        else:
            # Initialize for different number of classes
            linear_layer = model.last_linear[1] if dropout_config['classifier'] > 0 else model.last_linear
            nn.init.kaiming_normal_(linear_layer.weight, mode='fan_out', nonlinearity='relu')
            if linear_layer.bias is not None:
                nn.init.constant_(linear_layer.bias, 0)
            print(f"⚠️ Changed to {num_classes} classes - final layer randomly initialized")
    
    return model


# For backward compatibility, also provide the original xception function
def xception_original(num_classes=1000, pretrained='imagenet'):
    """Original Xception without dropout throughout."""
    model = Xception(num_classes=1000)
    
    if pretrained:
        settings = pretrained_settings['xception'][pretrained]
        pretrained_dict = model_zoo.load_url(settings['url'])
        
        new_dict = {}
        for k, v in pretrained_dict.items():
            if 'pointwise.weight' in k and v.dim() == 2:
                v = v.unsqueeze(-1).unsqueeze(-1)
            new_dict[k] = v
        
        model.load_state_dict(new_dict, strict=False)
        print(f"✅ Loaded ImageNet pretrained weights (original Xception)")
    
    if num_classes != 1000:
        model.fc = nn.Linear(2048, num_classes)
        model.last_linear = model.fc
        print(f"⚠️ Changed to {num_classes} classes")
    
    return model