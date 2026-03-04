'''
Purpose: Segmentation-style UNet built on torchvision backbones (e.g., ResNet-18/34) with skip connections and configurable upsampling.
Contents:
	•	get_backbone: Load a torchvision backbone and define skip/output feature names.
	•	UpsampleBlock: Parametric (transpose conv) or non-parametric (bilinear + conv) upsampling with optional skip concatenation.
	•	Unet: Decoder over a frozen or trainable backbone; exposes get_pretrained_parameters/get_random_initialized_parameters to split optimizers.
'''

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models

def get_backbone(name, pretrained=True):
    if name == 'resnet18':
        backbone = models.resnet18(pretrained=pretrained)
    elif name == 'resnet34':
        backbone = models.resnet34(pretrained=pretrained)
    else:
        raise NotImplemented('{} backbone model is not implemented so far.'.format(name))
    if name.startswith('resnet'):
        feature_names = [None, 'relu', 'layer1', 'layer2', 'layer3']
        backbone_output = 'layer4'
    else:
        raise NotImplemented('{} backbone model is not implemented so far.'.format(name))
    return backbone, feature_names, backbone_output

class UpsampleBlock(nn.Module):
    def __init__(self, ch_in, ch_out=None, skip_in=0, use_bn=True, parametric=False):
        super(UpsampleBlock, self).__init__()
        self.parametric = parametric
        ch_out = ch_in/2 if ch_out is None else ch_out
        if parametric:
            self.up = nn.ConvTranspose2d(in_channels=ch_in, out_channels=ch_out, kernel_size=(4, 4),
                                         stride=2, padding=1, output_padding=0, bias=(not use_bn))
            self.bn1 = nn.BatchNorm2d(ch_out) if use_bn else None
        else:
            self.up = None
            ch_in = ch_in + skip_in
            self.conv1 = nn.Conv2d(in_channels=ch_in, out_channels=ch_out, kernel_size=(3, 3),
                                   stride=1, padding=1, bias=(not use_bn))
            self.bn1 = nn.BatchNorm2d(ch_out) if use_bn else None
        self.relu = nn.ReLU(inplace=True)
        conv2_in = ch_out if not parametric else ch_out + skip_in
        self.conv2 = nn.Conv2d(in_channels=conv2_in, out_channels=ch_out, kernel_size=(3, 3),
                               stride=1, padding=1, bias=(not use_bn))
        self.bn2 = nn.BatchNorm2d(ch_out) if use_bn else None

    def forward(self, x, skip_connection=None):
        x = self.up(x) if self.parametric else F.interpolate(x, size=None, scale_factor=2, mode='bilinear',
                                                             align_corners=None)
        if self.parametric:
            x = self.bn1(x) if self.bn1 is not None else x
            x = self.relu(x)
        if skip_connection is not None:
            x = torch.cat([x, skip_connection], dim=1)
        if not self.parametric:
            x = self.conv1(x)
            x = self.bn1(x) if self.bn1 is not None else x
            x = self.relu(x)
        x = self.conv2(x)
        x = self.bn2(x) if self.bn2 is not None else x
        x = self.relu(x)
        return x

class Unet(nn.Module):
    def __init__(self, backbone_name='resnet50', pretrained=True, encoder_freeze=False,
                 classes=21, decoder_filters=(256, 128, 64, 32, 16), parametric_upsampling=True,
                 shortcut_features='default', decoder_use_batchnorm=True):
        super(Unet, self).__init__()
        self.backbone_name = backbone_name
        self.backbone, self.shortcut_features, self.bb_out_name = get_backbone(backbone_name, pretrained=pretrained)
        shortcut_chs, bb_out_chs = self.infer_skip_channels()
        if shortcut_features != 'default':
            self.shortcut_features = shortcut_features
        self.upsample_blocks = nn.ModuleList()
        decoder_filters = decoder_filters[:len(self.shortcut_features)]
        decoder_filters_in = [bb_out_chs] + list(decoder_filters[:-1])
        num_blocks = len(self.shortcut_features)
        for i, (filters_in, filters_out) in enumerate(zip(decoder_filters_in, decoder_filters)):
            self.upsample_blocks.append(UpsampleBlock(filters_in, filters_out,
                                                      skip_in=shortcut_chs[num_blocks-i-1],
                                                      parametric=parametric_upsampling,
                                                      use_bn=decoder_use_batchnorm))
        self.final_conv = nn.Conv2d(decoder_filters[-1], classes, kernel_size=(1, 1))
        if encoder_freeze:
            self.freeze_encoder()
        self.replaced_conv1 = False

    def freeze_encoder(self):
        for param in self.backbone.parameters():
            param.requires_grad = False

    def forward(self, *input):
        x, features = self.forward_backbone(*input)
        for skip_name, upsample_block in zip(self.shortcut_features[::-1], self.upsample_blocks):
            skip_features = features[skip_name]
            x = upsample_block(x, skip_features)
        x = self.final_conv(x)
        return x

    def forward_backbone(self, x):
        features = {None: None} if None in self.shortcut_features else dict()
        for name, child in self.backbone.named_children():
            x = child(x)
            if name in self.shortcut_features:
                features[name] = x
            if name == self.bb_out_name:
                break
        return x, features

    def infer_skip_channels(self):
        x = torch.zeros(1, 3, 224, 224)
        has_fullres_features = self.backbone_name.startswith('vgg') or self.backbone_name == 'unet_encoder'
        channels = [] if has_fullres_features else [0]
        for name, child in self.backbone.named_children():
            x = child(x)
            if name in self.shortcut_features:
                channels.append(x.shape[1])
            if name == self.bb_out_name:
                out_channels = x.shape[1]
                break
        return channels, out_channels

    def get_pretrained_parameters(self):
        for name, param in self.backbone.named_parameters():
            if not (self.replaced_conv1 and name == 'conv1.weight'):
                yield param

    def get_random_initialized_parameters(self):
        pretrained_param_names = set()
        for name, param in self.backbone.named_parameters():
            if not (self.replaced_conv1 and name == 'conv1.weight'):
                pretrained_param_names.add('backbone.{}'.format(name))
        for name, param in self.named_parameters():
            if name not in pretrained_param_names:
                yield param
