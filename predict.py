import glob
import os
import argparse

import cv2
import numpy as np
import paddle
import paddle.nn.functional as F

from hed_model import HED
from utils import load_pretrained_model


def tensor_flip(x, flip):
    """Flip tensor according directions"""
    if flip[0]:
        x = x[:, :, :, ::-1]
    if flip[1]:
        x = x[:, :, ::-1, :]
    return x


def parse_args():
    parser = argparse.ArgumentParser(description='Model training')
    # params of testing

    parser.add_argument(
        '--pretrained_model',
        dest='pretrained_model',
        help='The directory for pretrained model',
        type=str,
        default='/Users/alex/Downloads/model_hed.pdparams')
    parser.add_argument(
        '--dataset',
        dest='dataset',
        help='The directory for test dataset',
        type=str,
        default='/Users/alex/baidu/HED-BSDS/test/')
    parser.add_argument(
        '--save_dir',
        dest='save_dir',
        help='The directory for saving the model snapshot',
        type=str,
        default='./output/result')

    return parser.parse_args()


def infer(args):
    model = HED()
    load_pretrained_model(model, args.pretrained_model)
    model.eval()
    target_path = args.save_dir
    test_path = args.dataset
    image_list = glob.glob(test_path + "*.jpg")
    nimgs = len(image_list)
    print("totally {} images".format(nimgs))
    for i in range(nimgs):
        img = image_list[i]
        img = cv2.imread(img).astype(np.float32)
        h, w, _ = img.shape
        edge = np.zeros((h, w), np.float32)
        scales = [0.5, 1, 1.5]
        flips = [[False, False]]

        for s in scales:
            for flip in flips:
                h1, w1 = int(s * h), int(s * w)
                img1 = cv2.resize(img, (w1, h1), interpolation=cv2.INTER_CUBIC)
                img1 = img1 - np.array((104.00699, 116.66876762, 122.67891434))
                img1 = np.transpose(img1, [2, 0, 1])
                img1 = img1[np.newaxis, :, :, :].astype(np.float32)
                img1 = tensor_flip(img1, flip)
                img1 = paddle.to_tensor(img1)
                edge1 = model(img1)[0]
                edge1 = F.sigmoid(edge1)
                edge1 = tensor_flip(edge1, flip)
                edge1 = edge1.numpy()[0, 0, :, :]
                edge += cv2.resize(
                    edge1, (w, h),
                    interpolation=cv2.INTER_CUBIC).astype(np.float32)
        edge /= (len(scales) * len(flips))
        edge = edge / edge.max()
        edge *= 255.0
        fn, ext = os.path.splitext(image_list[i])
        fn = fn.split('/')[-1]

        cv2.imwrite(os.path.join(target_path, "{}".format(fn) + '.png'), edge)
        print("Saving to '" + os.path.join(target_path, image_list[i][0:-4]) +
              "', Processing %d of %d..." % (i + 1, nimgs))


if __name__ == '__main__':
    args = parse_args()
    infer(args)
