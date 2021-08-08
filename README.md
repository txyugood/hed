# 1.简介
本项目基于PaddlePaddle复现《 Holistically-Nested Edge Detection》论文。改论文提出了一种使用卷积神经网络来检测边缘的方法。并超越原文精度，达到ODS=0.787。

论文地址：

[https://arxiv.org/abs/1504.06375](https://arxiv.org/abs/1504.06375)

参考项目:

[https://github.com/sniklaus/pytorch-hed](https://github.com/sniklaus/pytorch-heds) 精度:ODS=0.774

[https://github.com/s9xie/hed](https://github.com/s9xie/hed) 精度:ODS=0.782 (原文项目)

[https://github.com/zeakey/hed](https://github.com/zeakey/hed) 精度:ODS=0.779



# 2.数据集下载

HED-BSDS:

[https://aistudio.baidu.com/aistudio/datasetdetail/103495](https://aistudio.baidu.com/aistudio/datasetdetail/103495)

# 3.环境

PaddlePaddle >= 2.0.0

python >= 3.6

# 4.VGG预训练模型

模型下载地址：

链接: [https://pan.baidu.com/s/1etmgEGtbhwxMECwIRkL1Lg](https://pan.baidu.com/s/1etmgEGtbhwxMECwIRkL1Lg)

密码: uo0e

# 3.训练

```
python train.py --iters 100000 --batch_size 10 --learning_rate 0.0001 --save_interval 1000 --pretrained_model vgg.pdparams --dataset HED-BSDS
```

上述命令中pretrained_model和dataset改为实际地址。

# 4.测试

```
python predict.py --pretrained_model hed_model.pdparams --dataset HED-BSDS/test --save_dir output/result
```
上述命令中pretrained_model为训练结果模型，dataset为测试图片路径。

训练结果模型下载地址：

链接: [https://pan.baidu.com/s/1VXnrHCu9Wb7zAiOTsb0vFw](https://pan.baidu.com/s/1VXnrHCu9Wb7zAiOTsb0vFw)

密码: pocu

# 5.验证模型

预测结果需要使用另外一个项目进行评估。

评估项目地址:

[https://github.com/zeakey/edgeval](https://github.com/zeakey/edgeval)

运行环境 Matlab 2014a


本项目评估结果：

![](https://github.com/txyugood/hed/blob/main/images/hed_result.png?raw=true)

预测结果：

![](https://github.com/txyugood/hed/blob/main/images/388067.jpg?raw=true)

![](https://github.com/txyugood/hed/blob/main/images/388067.png?raw=true)


# 6.总结

本论文发布时间较早，应该是比较早使用卷积神经网络来做图像边缘检测的项目。论文中使用了VGG16网络做为backbone。然后分别从VGG的5个部分做了5个分支，输出不同的边缘图，最后用一个卷积将5个边缘图融合，作为最终结果。

训练策略方面，文中将VGG的前4部分学习率的倍率设置为1，第五部分的学习率倍率是前四部分的100倍。这意味着，前四部分主要进行微调，第五部分需要重新学习，主要用来输出边缘图像。同时最后融合部分的卷积层的学习率的倍率设置为0.001，同时初始化值为0.2，这意味着，用较低的学习率来微调融合权重。

在训练过程中，论文使用SDG优化器，使用StepDecay动态调整学习率，学习率为1e-6。但是通过大量测试，效果并不理想，所以在本项目中将学习率设置为1e-4,同时使用Warmup和PolynomialDecay的方式动态的调整学习率，总迭代次数为100000次。最终评测结果0.787，超过了原文精度以及其他pytoch和caffe版本的复现项目。
