
import numpy as np
import paddle
import paddle.nn as nn
from paddle import ParamAttr
from paddle.nn import Conv2D, Conv2DTranspose
import paddle.nn.functional as F

from utils import load_pretrained_model
from vgg import VGG16


def upsample_filt(size):
    factor = (size + 1) // 2
    if size % 2 == 1:
        center = factor - 1
    else:
        center = factor - 0.5
    og = np.ogrid[:size, :size]
    return (1 - abs(og[0] - center) / factor) * \
           (1 - abs(og[1] - center) / factor)

def weight_init(module):
    for n, m in module.named_children():
        if isinstance(m, nn.Conv2DTranspose):
            i, o, h, w = m.weight.shape
            if i != o:
                print('input + output channels need to be the same')
                raise
            if h != w:
                print('filters need to be square')
                raise
            filt = upsample_filt(h)
            filt = filt[np.newaxis, np.newaxis, :, :]
            initializer = nn.initializer.Assign(value=filt)
            initializer(m.weight, m.weight.block)
            m.weight.optimize_attr['learning_rate'] = 0.0
            m.bias.optimize_attr['learning_rate'] = 0.0
            pass






class HED(nn.Layer):
    def __init__(self, pretrained=None, backbone_pretrained=None):
        super(HED, self).__init__()
        self.backbone = VGG16(pretrained=backbone_pretrained)
        self.head = HEDHead()
        for p in self.backbone.parameters():
            p.optimize_attr['learning_rate'] /= 10.0
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
                                                      regularizer=paddle.regularizer.L2Decay(2e-4)),
                                bias_attr=ParamAttr(name="score_dns1_bias",
                                                    learning_rate=lr * 2.0,
                                                    regularizer=paddle.regularizer.L2Decay(0))
                                )
        self.dsn_conv2 = Conv2D(in_channels=128, out_channels=1, kernel_size=1, stride=1, padding=0,
                                weight_attr=ParamAttr(name="score_dns2_weights",
                                                      learning_rate=lr,
                                                      regularizer=paddle.regularizer.L2Decay(2e-4)),
                                bias_attr=ParamAttr(name="score_dns2_bias",
                                                    learning_rate=lr * 2.0,
                                                    regularizer=paddle.regularizer.L2Decay(0))
                                )
        self.dsn_conv3 = Conv2D(in_channels=256, out_channels=1, kernel_size=1, stride=1, padding=0,
                                weight_attr=ParamAttr(name="score_dns3_weights",
                                                      learning_rate=lr,
                                                      regularizer=paddle.regularizer.L2Decay(2e-4)),
                                bias_attr=ParamAttr(name="score_dns3_bias",
                                                    learning_rate=lr * 2.0,
                                                    regularizer=paddle.regularizer.L2Decay(0))
                                )
        self.dsn_conv4 = Conv2D(in_channels=512, out_channels=1, kernel_size=1, stride=1, padding=0,
                                weight_attr=ParamAttr(name="score_dns4_weights",
                                                      learning_rate=lr,
                                                      regularizer=paddle.regularizer.L2Decay(2e-4)),
                                bias_attr=ParamAttr(name="score_dns4_bias",
                                                    learning_rate=lr * 2.0,
                                                    regularizer=paddle.regularizer.L2Decay(0))
                                )
        self.dsn_conv5 = Conv2D(in_channels=512, out_channels=1, kernel_size=1, stride=1, padding=0,
                                weight_attr=ParamAttr(name="score_dns5_weights",
                                                      learning_rate=lr,
                                                      regularizer=paddle.regularizer.L2Decay(2e-4)),
                                bias_attr=ParamAttr(name="score_dns5_bias",
                                                    learning_rate=lr * 2.0,
                                                    regularizer=paddle.regularizer.L2Decay(0))
                                )

        self.devcon2 = Conv2DTranspose(in_channels=1, out_channels=1, kernel_size=4, stride=2)
        self.devcon3 = Conv2DTranspose(in_channels=1, out_channels=1, kernel_size=8, stride=4)
        self.devcon4 = Conv2DTranspose(in_channels=1, out_channels=1, kernel_size=16, stride=8)
        self.devcon5 = Conv2DTranspose(in_channels=1, out_channels=1, kernel_size=32, stride=16)

        self.combine = Conv2D(in_channels=5, out_channels=1, kernel_size=1, stride=1, padding=0)

        weight_init(self)
    def forward(self, inputs, ori_shape):
        score1 = self.dsn_conv1(inputs[0])
        score2 = self.dsn_conv2(inputs[1])
        score3 = self.dsn_conv3(inputs[2])
        score4 = self.dsn_conv4(inputs[3])
        score5 = self.dsn_conv5(inputs[4])

        # score1 = F.interpolate(score1, ori_shape[2:], mode='bilinear')
        # score2 = F.interpolate(score2, ori_shape[2:], mode='bilinear')
        # score3 = F.interpolate(score3, ori_shape[2:], mode='bilinear')
        # score4 = F.interpolate(score4, ori_shape[2:], mode='bilinear')
        # score5 = F.interpolate(score5, ori_shape[2:], mode='bilinear')

        score2 = self.devcon2(score2)
        score3 = self.devcon3(score3)
        score4 = self.devcon4(score4)
        score5 = self.devcon5(score5)

        offset_h = int((score1.shape[2] - ori_shape[2]) / 2 + 0.5)
        offset_w = int((score1.shape[3] - ori_shape[3]) / 2 + 0.5)
        score1 = paddle.crop(score1, ori_shape[0:1] + [1] + ori_shape[2:], [0, 0, offset_h, offset_w])
        offset_h = int((score2.shape[2] - ori_shape[2]) / 2 + 0.5)
        offset_w = int((score2.shape[3] - ori_shape[3]) / 2 + 0.5)
        score2 = paddle.crop(score2, ori_shape[0:1] + [1] + ori_shape[2:], [0, 0, offset_h, offset_w])
        offset_h = int((score3.shape[2] - ori_shape[2]) / 2 + 0.5)
        offset_w = int((score3.shape[3] - ori_shape[3]) / 2 + 0.5)
        score3 = paddle.crop(score3, ori_shape[0:1] + [1] + ori_shape[2:], [0, 0, offset_h, offset_w])
        offset_h = int((score4.shape[2] - ori_shape[2]) / 2 + 0.5)
        offset_w = int((score4.shape[3] - ori_shape[3]) / 2 + 0.5)
        score4 = paddle.crop(score4, ori_shape[0:1] + [1] + ori_shape[2:], [0, 0, offset_h, offset_w])
        offset_h = int((score5.shape[2] - ori_shape[2]) / 2 + 0.5)
        offset_w = int((score5.shape[3] - ori_shape[3]) / 2 + 0.5)
        score5 = paddle.crop(score5, ori_shape[0:1] + [1] + ori_shape[2:], [0, 0, offset_h, offset_w])

        scores = [score1, score2, score3, score4, score5]
        return [
            self.combine(paddle.concat(scores, axis=1)),
            score1,score2,score3,score4,score5
            ]

if __name__ == '__main__':
    model = HED()
    y = model(paddle.rand([1,3,224,224]))
    pass

