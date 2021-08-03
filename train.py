import os
import time

import paddle
import paddle.nn.functional as F
from dataset import Dataset
from hed_model import HED
from transforms import Normalize
from timer import TimeAverager, calculate_eta
import logger

save_dir = 'output'
iters = 100000
save_interval = 1000
batch_size = 1
transforms = [
    Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
]
dataset = Dataset(transforms=transforms,
                  dataset_root="/Users/alex/baidu/HED-BSDS",
                  train_path="/Users/alex/baidu/HED-BSDS/train_pair.lst")

batch_sampler = paddle.io.DistributedBatchSampler(
    dataset, batch_size=batch_size, shuffle=True, drop_last=True)

loader = paddle.io.DataLoader(
    dataset,
    batch_sampler=batch_sampler,
    num_workers=0,
    return_list=True,
)
iter = 0
iters_per_epoch = len(batch_sampler)
log_iters = 1
avg_loss = 0.0
avg_loss_list = []
reader_cost_averager = TimeAverager()
batch_cost_averager = TimeAverager()
model = HED(backbone_pretrained='/Users/alex/Downloads/VGG16_pretrained.pdparams')
learning_rate = paddle.optimizer.lr.StepDecay(learning_rate=1e-6, step_size=10000, gamma=0.1)
optimizer = paddle.optimizer.Momentum(learning_rate=learning_rate, parameters=model.parameters(), weight_decay=2e-4)

batch_start = time.time()
while iter < iters:
    for data in loader:
        iter += 1
        if iter > iters:
            break
        images = data[0]
        labels = data[1]
        logits_list = model(images)
        loss_list = []
        for logits in logits_list:
            loss = F.binary_cross_entropy_with_logits(logits, labels)
            loss_list.append(loss)
        loss = sum(loss_list)
        loss.backward()
        optimizer.step()
        model.clear_gradients()

        lr = optimizer.get_lr()

        # update lr
        if isinstance(optimizer, paddle.distributed.fleet.Fleet):
            lr_sche = optimizer.user_defined_optimizer._learning_rate
        else:
            lr_sche = optimizer._learning_rate
        if isinstance(lr_sche, paddle.optimizer.lr.LRScheduler):
            lr_sche.step()

        avg_loss += loss.numpy()[0]
        if not avg_loss_list:
            avg_loss_list = [l.numpy() for l in loss_list]
        else:
            for i in range(len(loss_list)):
                avg_loss_list[i] += loss_list[i].numpy()
        batch_cost_averager.record(
            time.time() - batch_start, num_samples=batch_size)

        if (iter) % log_iters == 0:
            avg_loss /= log_iters
            avg_loss_list = [l[0] / log_iters for l in avg_loss_list]
            remain_iters = iters - iter
            avg_train_batch_cost = batch_cost_averager.get_average()
            avg_train_reader_cost = reader_cost_averager.get_average()
            eta = calculate_eta(remain_iters, avg_train_batch_cost)
            logger.info(
                "[TRAIN] epoch: {}, iter: {}/{}, loss: {:.4f}, lr: {:.6f}, batch_cost: {:.4f}, reader_cost: {:.5f}, ips: {:.4f} samples/sec | ETA {}"
                    .format((iter - 1) // iters_per_epoch + 1, iter, iters,
                            avg_loss, lr, avg_train_batch_cost,
                            avg_train_reader_cost,
                            batch_cost_averager.get_ips_average(), eta))

            avg_loss = 0.0
            avg_loss_list = []
            reader_cost_averager.reset()
            batch_cost_averager.reset()

        if (iter % save_interval == 0 or iter == iters):
            current_save_dir = os.path.join(save_dir,
                                            "iter_{}".format(iter))
            if not os.path.isdir(current_save_dir):
                os.makedirs(current_save_dir)
            paddle.save(model.state_dict(),
                        os.path.join(current_save_dir, 'model.pdparams'))
            paddle.save(optimizer.state_dict(),
                        os.path.join(current_save_dir, 'model.pdopt'))
        batch_start = time.time()

