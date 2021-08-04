import os
import glob

import cv2
import numpy as np
import paddle
import paddle.nn.functional as F

from hed_model import HED
from utils import load_pretrained_model
from transforms import Compose, Normalize

target_path = 'output/result'
if not os.path.exists(target_path):
    os.makedirs(target_path)

model = HED()
load_pretrained_model(model, "/Users/alex/Downloads/model_hed.pdparams")
model.eval()
test_path = "/Users/alex/baidu/HED-BSDS/test/"
image_list = glob.glob(test_path + "*.jpg")
transforms = Compose([Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
for i, image_im in enumerate(image_list):
    img = cv2.imread(image_im)
    img, _ = transforms(img)
    img = paddle.to_tensor(img[np.newaxis, :, :, :].astype('float32'))
    logits = model(img)

    save_image = F.sigmoid(logits[0][0])
    save_image = paddle.transpose(save_image, [1, 2, 0])
    save_image = paddle.squeeze(save_image, axis=-1)
    save_image = save_image.numpy() * 255.0
    save_image = save_image.astype('uint8')
    cv2.imwrite(os.path.join(target_path, image_im.split('/')[-1].split('.')[0] + ".png"),save_image)
    print("{}/{}".format(i+1, len(image_list)))
    pass

pass