from math import ceil

import numpy as np
import paddle
import paddle.nn as nn
from paddle import ParamAttr
from paddle.nn import Conv2D, Conv2DTranspose
import paddle.nn.functional as F

from utils import load_pretrained_model
from vgg import VGG16

class HED(nn.Layer):
    def __init__(self, pretrained=None, backbone_pretrained=None):
        super(HED, self).__init__()
        self.backbone = VGG16(pretrained=backbone_pretrained)
        self.head = HEDHead()
        if pretrained is not None:
            load_pretrained_model(self, pretrained)

    def forward(self, inputs):
        y = self.backbone(inputs)
        y = self.head(y, inputs.shape)
        return y


class HEDHead(nn.Layer):
    def __init__(self):
        super(HEDHead, self).__init__()
        lr = 0.01
        self.dsn_conv1 = Conv2D(in_channels=64, out_channels=1, kernel_size=1, stride=1, padding=0,
                                weight_attr=ParamAttr(name="score_dns1_weights",
                                                      learning_rate=lr,
                                                      regularizer=paddle.regularizer.L2Decay(2e-4),
                                                      initializer=nn.initializer.Constant(value=0)),
                                bias_attr=ParamAttr(name="score_dns1_bias",
                                                    learning_rate=lr * 2.0,
                                                    regularizer=paddle.regularizer.L2Decay(0),
                                                    initializer=nn.initializer.Constant(value=0))
                                )
        self.dsn_conv2 = Conv2D(in_channels=128, out_channels=1, kernel_size=1, stride=1, padding=0,
                                weight_attr=ParamAttr(name="score_dns2_weights",
                                                      learning_rate=lr,
                                                      regularizer=paddle.regularizer.L2Decay(2e-4),
                                                      initializer=nn.initializer.Constant(value=0)),
                                bias_attr=ParamAttr(name="score_dns2_bias",
                                                    learning_rate=lr * 2.0,
                                                    regularizer=paddle.regularizer.L2Decay(0),
                                                    initializer=nn.initializer.Constant(value=0))
                                )
        self.dsn_conv3 = Conv2D(in_channels=256, out_channels=1, kernel_size=1, stride=1, padding=0,
                                weight_attr=ParamAttr(name="score_dns3_weights",
                                                      learning_rate=lr,
                                                      regularizer=paddle.regularizer.L2Decay(2e-4),
                                                      initializer=nn.initializer.Constant(value=0)),
                                bias_attr=ParamAttr(name="score_dns3_bias",
                                                    learning_rate=lr * 2.0,
                                                    regularizer=paddle.regularizer.L2Decay(0),
                                                    initializer=nn.initializer.Constant(value=0))
                                )
        self.dsn_conv4 = Conv2D(in_channels=512, out_channels=1, kernel_size=1, stride=1, padding=0,
                                weight_attr=ParamAttr(name="score_dns4_weights",
                                                      learning_rate=lr,
                                                      regularizer=paddle.regularizer.L2Decay(2e-4),
                                                      initializer=nn.initializer.Constant(value=0)),
                                bias_attr=ParamAttr(name="score_dns4_bias",
                                                    learning_rate=lr * 2.0,
                                                    regularizer=paddle.regularizer.L2Decay(0),
                                                    initializer=nn.initializer.Constant(value=0))
                                )
        self.dsn_conv5 = Conv2D(in_channels=512, out_channels=1, kernel_size=1, stride=1, padding=0,
                                weight_attr=ParamAttr(name="score_dns5_weights",
                                                      learning_rate=lr,
                                                      regularizer=paddle.regularizer.L2Decay(2e-4),
                                                      initializer=nn.initializer.Constant(value=0)),
                                bias_attr=ParamAttr(name="score_dns5_bias",
                                                    learning_rate=lr * 2.0,
                                                    regularizer=paddle.regularizer.L2Decay(0),
                                                    initializer=nn.initializer.Constant(value=0))
                                )

        self.combine = Conv2D(in_channels=5, out_channels=1, kernel_size=1, stride=1, padding=0,
                              weight_attr=ParamAttr(name="score_combine_weights",
                                                    learning_rate=0.001,
                                                    regularizer=paddle.regularizer.L2Decay(2e-4),
                                                    initializer=nn.initializer.Constant(value=0.2)),
                              bias_attr=ParamAttr(name="score_combine_bias",
                                                  learning_rate=0.001 * 2.0,
                                                  regularizer=paddle.regularizer.L2Decay(0),
                                                  initializer=nn.initializer.Constant(value=0))
                              )

    def forward(self, inputs, ori_shape):
        score1 = self.dsn_conv1(inputs[0])
        score2 = self.dsn_conv2(inputs[1])
        score3 = self.dsn_conv3(inputs[2])
        score4 = self.dsn_conv4(inputs[3])
        score5 = self.dsn_conv5(inputs[4])

        score2 = F.interpolate(score2, ori_shape[2:], mode='bilinear')
        score3 = F.interpolate(score3, ori_shape[2:], mode='bilinear')
        score4 = F.interpolate(score4, ori_shape[2:], mode='bilinear')
        score5 = F.interpolate(score5, ori_shape[2:], mode='bilinear')

        scores = [score1, score2, score3, score4, score5]
        return [self.combine(paddle.concat(scores, axis=1))] + scores

if __name__ == '__main__':
    model = HED()
    y = model(paddle.rand([1,3,224,224]))
    pass

