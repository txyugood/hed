import paddle
from paddle import ParamAttr
import paddle.nn as nn
import paddle.nn.functional as F
from paddle.nn import Conv2D, BatchNorm, Linear, Dropout
from paddle.nn import AdaptiveAvgPool2D, MaxPool2D, AvgPool2D
from utils import load_pretrained_model

__all__ = ["VGG11", "VGG13", "VGG16", "VGG19"]


class ConvBlock(nn.Layer):
    def __init__(self, input_channels, output_channels, groups, stage, name=None):
        super(ConvBlock, self).__init__()

        lr = 1.0
        if stage == 5:
            lr = 100.0

        self.groups = groups
        self._conv_1 = Conv2D(
            in_channels=input_channels,
            out_channels=output_channels,
            kernel_size=3,
            stride=1,
            padding=35 if stage == 1 else 1,
            weight_attr=ParamAttr(name=name + "1_weights",
                                  learning_rate=lr,
                                  regularizer=paddle.regularizer.L2Decay(2e-4)),
            bias_attr=ParamAttr(name=name + "1_bias",
                                learning_rate=lr * 2.0,
                                regularizer=paddle.regularizer.L2Decay(0)))
        if groups == 2 or groups == 3 or groups == 4:
            self._conv_2 = Conv2D(
                in_channels=output_channels,
                out_channels=output_channels,
                kernel_size=3,
                stride=1,
                padding=1,
                weight_attr=ParamAttr(name=name + "2_weights",
                                      learning_rate=lr,
                                      regularizer=paddle.regularizer.L2Decay(2e-4)),
                bias_attr=ParamAttr(name=name + "2_bias",
                                learning_rate=lr * 2.0,
                                regularizer=paddle.regularizer.L2Decay(0)))
        if groups == 3 or groups == 4:
            self._conv_3 = Conv2D(
                in_channels=output_channels,
                out_channels=output_channels,
                kernel_size=3,
                stride=1,
                padding=1,
                weight_attr=ParamAttr(name=name + "3_weights",
                                      learning_rate=lr,
                                      regularizer=paddle.regularizer.L2Decay(2e-4)
                                      ),
                bias_attr=ParamAttr(name=name + "3_bias",
                                learning_rate=lr * 2.0,
                                regularizer=paddle.regularizer.L2Decay(0)))
        if groups == 4:
            self._conv_4 = Conv2D(
                in_channels=output_channels,
                out_channels=output_channels,
                kernel_size=3,
                stride=1,
                padding=1,
                weight_attr=ParamAttr(name=name + "4_weights",
                                      learning_rate=lr,
                                      regularizer=paddle.regularizer.L2Decay(2e-4)
                                      ),
                bias_attr=ParamAttr(name=name + "4_bias",
                                learning_rate=lr * 2.0,
                                regularizer=paddle.regularizer.L2Decay(0)))

        self._pool = MaxPool2D(kernel_size=2, stride=2, padding=0)

    def forward(self, inputs):
        x = self._conv_1(inputs)
        x = F.relu(x)
        if self.groups == 2 or self.groups == 3 or self.groups == 4:
            x = self._conv_2(x)
            conv = x
            x = F.relu(x)
        if self.groups == 3 or self.groups == 4:
            x = self._conv_3(x)
            conv = x 
            x = F.relu(x)
        if self.groups == 4:
            x = self._conv_4(x)
            conv = x
            x = F.relu(x)
        pool = self._pool(x)
        return pool, conv


class VGGNet(nn.Layer):
    def __init__(self, layers=11, stop_grad_layers=0, pretrained=None):
        super(VGGNet, self).__init__()

        self.layers = layers
        self.stop_grad_layers = stop_grad_layers
        self.vgg_configure = {
            11: [1, 1, 2, 2, 2],
            13: [2, 2, 2, 2, 2],
            16: [2, 2, 3, 3, 3],
            19: [2, 2, 4, 4, 4]
        }
        assert self.layers in self.vgg_configure.keys(), \
            "supported layers are {} but input layer is {}".format(
                self.vgg_configure.keys(), layers)
        self.groups = self.vgg_configure[self.layers]

        self._conv_block_1 = ConvBlock(3, 64, self.groups[0], 1, name="conv1_")
        self._conv_block_2 = ConvBlock(64, 128, self.groups[1], 2, name="conv2_")
        self._conv_block_3 = ConvBlock(128, 256, self.groups[2], 3, name="conv3_")
        self._conv_block_4 = ConvBlock(256, 512, self.groups[3], 4, name="conv4_")
        self._conv_block_5 = ConvBlock(512, 512, self.groups[4], 5, name="conv5_")

        for idx, block in enumerate([
                self._conv_block_1, self._conv_block_2, self._conv_block_3,
                self._conv_block_4, self._conv_block_5
        ]):
            if self.stop_grad_layers >= idx + 1:
                for param in block.parameters():
                    param.trainable = False
        if pretrained is not None:
            load_pretrained_model(self, pretrained)

    def forward(self, inputs):
        feat_list = []
        pool, x = self._conv_block_1(inputs)
        feat_list.append(x)
        pool, x = self._conv_block_2(pool)
        feat_list.append(x)
        pool, x = self._conv_block_3(pool)
        feat_list.append(x)
        pool, x = self._conv_block_4(pool)
        feat_list.append(x)
        pool, x = self._conv_block_5(pool)
        feat_list.append(x)
        return feat_list


def VGG11(**args):
    model = VGGNet(layers=11, **args)
    return model


def VGG13(**args):
    model = VGGNet(layers=13, **args)
    return model


def VGG16(**args):
    model = VGGNet(layers=16, **args)
    return model


def VGG19(**args):
    model = VGGNet(layers=19, **args)
    return model

import numpy as np

def export_weight_names(net):
    print(net.state_dict().keys())
    with open('paddle_vgg.txt', 'w') as f:
        for key in net.state_dict().keys():
            f.write(key + '\n')
if __name__ == '__main__':
    model = VGG16()
    export_weight_names(model)
