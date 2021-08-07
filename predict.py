import os
import glob

import cv2
import scipy
import numpy as np
import paddle
import paddle.nn.functional as F


from hed_model import HED
from utils import load_pretrained_model
from transforms import Compose, Normalize
from transforms import functional
tiny = True

if tiny:
    target_path = 'output/data/tiny-edges'
else:
    target_path = 'output/result'

if not os.path.exists(target_path):
    os.makedirs(target_path)


def tensor_flip(x, flip):
    """Flip tensor according directions"""
    if flip[0]:
        x = x[:, :, :, ::-1]
    if flip[1]:
        x = x[:, :, ::-1, :]
    return x

model = HED()
load_pretrained_model(model, "/Users/alex/Downloads/model_hed.pdparams")
model.eval()
test_path = "/Users/alex/baidu/HED-BSDS/test/"
if tiny:
    image_list = [os.path.join(test_path, fn) for fn in ['2018.jpg', '3063.jpg', '5096.jpg', '14092.jpg', '35049.jpg',
                                                         '6046.jpg', '41085.jpg', '28083.jpg', '101084.jpg',
                                                         '246009.jpg']]
else:
    image_list = glob.glob(test_path + "*.jpg")
nimgs = len(image_list)
print("totally {} images".format(nimgs))
for i in range(nimgs):
    img = image_list[i]
    img = cv2.imread(img).astype(np.float32)
    h, w, _ = img.shape
    edge = np.zeros((h, w), np.float32)
    scales = [0.5, 1, 1.5]
    # flips = [[False, False], [True, False], [False, True],[True, True]]

    # scales = [1]
    flips = [[False, False]]
    # scales = [1]

    for s in scales:
        for flip in flips:
            h1, w1 = int(s * h), int(s * w)
            img1 = cv2.resize(img, (w1, h1), interpolation=cv2.INTER_CUBIC)
            img1 = img1 - np.array((104.00699, 116.66876762, 122.67891434))
            img1 = np.transpose(img1, [2, 0, 1])
            img1 = img1[np.newaxis, :, :, :].astype(np.float32)
            img1 = tensor_flip(img1, flip)
            img1 = paddle.to_tensor(img1)
            # img1 = F.interpolate(img1, (h1, w1), mode='bilinear').astype('float32')
            edge1 = model(img1)[0]
            # edge1 = F.interpolate(edge1, (h, w), mode='bilinear')
            edge1 = F.sigmoid(edge1)
            edge1 = tensor_flip(edge1, flip)
            edge1 = edge1.numpy()[0, 0, :, :]
            # edge += edge1.astype(np.float32)
            # edge1 = edge1 / edge1.max()
            edge += cv2.resize(edge1, (w, h), interpolation=cv2.INTER_CUBIC).astype(np.float32)
    edge /= (len(scales) * len(flips))
    edge = edge / edge.max()
    edge *= 255.0
    fn, ext = os.path.splitext(image_list[i])
    fn = fn.split('/')[-1]

    cv2.imwrite(os.path.join(target_path, "{}".format(fn) + '.png'), edge)
    print("Saving to '" + os.path.join(target_path, image_list[i][0:-4]) + "', Processing %d of %d..." % (i + 1, nimgs))
