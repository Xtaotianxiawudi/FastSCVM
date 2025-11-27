import timeit
from datetime import datetime
import numpy as np
import pytz
import torch.nn.functional as F
import tqdm
from utils.Utils import *

bceloss = nn.BCEWithLogitsLoss()
diceloss = DiceLoss()

# ------------------
# 获取当前优化器的学习率
# ------------------
def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group['lr']


# ----------------
# 根据多项式更新学习率
# ----------------
def lr_poly(base_lr, iter, max_iter, power):
    return base_lr * ((1 - float(iter) / max_iter) ** (power))

# -----------------------------
# 使用lr_poly更新新的学习率到优化器中
# -----------------------------
def adjust_learning_rate(optimizer, learning_rate, i_iter, max_iter, power):
    """Sets the learning rate to the initial LR divided by 5 at 60th, 120th and 160th epochs"""
    lr = lr_poly(learning_rate, i_iter, max_iter, power)
    optimizer.param_groups[0]['lr'] = lr
    return lr


class Trainer(object):

    def __init__(self, cuda, model_gen, optimizer_gen,
                 train_loader, validation_loader, out, max_epoch, stop_epoch=None,
                 lr_gen=1e-3, lr_decrease_rate=0.1, interval_validate=None, batch_size=8, warmup_epoch=10,
                 ):
        self.cuda = cuda
        self.warmup_epoch = warmup_epoch
        self.model_gen = model_gen  # generator
        self.optim_gen = optimizer_gen
        self.lr_gen = lr_gen
        self.lr_decrease_rate = lr_decrease_rate
        self.batch_size = batch_size
        self.train_loader = train_loader
        self.validation_loader = validation_loader
        self.time_zone = 'Asia/Hong_Kong'
        self.timestamp_start = datetime.now(pytz.timezone(self.time_zone))  # 开始训练的时间戳
        self.out = out

        if interval_validate is None:
            self.interval_validate = 10
        else:
            self.interval_validate = interval_validate

        self.log_headers = [
            'epoch',
            'iteration',
            'train/loss_seg',
            'train/loss_adv',
            'train/loss_disc',
            'valid/loss_seg',
            'valid/cup_dice',
            'valid/disc_dice',
            'elapsed_time',
        ]
        # -----------------------------------
        # 将log_headers表头信息写入out中的log.scv
        # -----------------------------------
        if not osp.exists(osp.join(self.out, 'log.csv')):
            with open(osp.join(self.out, 'log.csv'), 'w') as f:
                f.write(','.join(self.log_headers) + '\n')
        self.epoch = 0
        self.iteration = 0
        self.max_epoch = max_epoch
        self.stop_epoch = max_epoch
        self.best_disc_dice = 0.0
        self.running_seg_loss = 0.0
        self.best_epoch = -1
        self.best_loss = np.inf  # 正无穷

    def validate(self):
        self.model_gen.eval()
        val_loss = 0
        val_cup_dice = 0
        val_disc_dice = 0
        metrics = []
        with torch.no_grad():
            for batch_idx, sample in tqdm.tqdm(
                    enumerate(self.validation_loader), total=len(self.validation_loader),
                    desc='Valid iteration=%d' % self.iteration, ncols=80,
                    leave=False):
                data = sample['image']  # torch.Size([12, 3, 512, 512])
                target_map = sample['map']  # torch.Size([12, 2, 512, 512])
                target_boundary = sample['boundary']  # torch.Size([12, 1, 512, 512])
                if self.cuda:
                    data, target_map, target_boundary = data.cuda(), target_map.cuda(), target_boundary.cuda()
                predictions = self.model_gen(data)
                # ----------------------------------
                # 验证集损失  采用的是二分类的交叉熵损失计算
                # ----------------------------------
                loss_seg1_1 = bceloss(predictions, target_map) # mask
                loss = loss_seg1_1
                # loss = F.binary_cross_entropy_with_logits(predictions, target_map)
                loss_data = loss.data.item()
                val_loss += loss_data
                # -----------------------------------------------------------------
                # 计算 当前批次的 视杯 视盘 dice值，并累加当前epoch所有批次的 视杯 视盘 dice值
                # 用的是metrics 下的 dice_coeff_2label(prediction, target)
                # -------------------------------------------------------
                dice_cup, dice_disc = dice_coeff_2label(predictions, target_map)
                # print(predictions.shape) torch.Size([12, 2, 256, 256])
                # print(target_map.shape)  torch.Size([12, 2, 256, 256])
                val_cup_dice += dice_cup
                val_disc_dice += dice_disc
            # -----------------------------------------------
            # 计算验证集  当前epoch 的平均验证损失 平均验证视杯dice 平均视盘dice
            # -----------------------------------------------
            val_loss /= len(self.validation_loader)
            val_cup_dice /= len(self.validation_loader)
            val_disc_dice /= len(self.validation_loader)

            metrics.append((val_loss, val_cup_dice, val_disc_dice))
            metrics = np.mean(metrics, axis=0)  # 感觉是多余的，本来就只有一个平均值
            # --------------------------------------------------------------------
            # 在验证集中用加权视杯视盘比确定最佳的epoch，此外因为视杯比较难分割，所以占的比重更大
            # 如果是最佳权重，即可写入到out目录下
            # --------------------------------------------------------------------
            mean_dice = (val_cup_dice * 2 + val_disc_dice) / 3
            is_best = mean_dice > self.best_disc_dice  # 判断是否是最好的模型
            if is_best:
                self.best_epoch = self.epoch + 1  # 如果是 记录下来
                self.best_disc_dice = mean_dice   # 并更新最佳的mean_dice
            torch.save({
                'epoch': self.epoch,
                'iteration': self.iteration,
                'arch': self.model_gen.__class__.__name__,
                'optim_state_dict': self.optim_gen.state_dict(),  # 优化器状态参数
                'model_state_dict': self.model_gen.state_dict(),  # 模型状态参数
                'learning_rate_gen': get_lr(self.optim_gen),
                # 'best_mean_dice': self.best_mean_dice,
            }, osp.join(self.out, 'checkpoint_%d.pth.tar' % self.epoch))
            # ----------------------------------------------------------------------
            # 将metrics(val_loss, val_cup_dice, val_disc_dice)数据写入out+log.scv文件中
            # 注意:此时写入的是   一个  epoch的 metrics 数据
            # ----------------------------------------------------------------------
            with open(osp.join(self.out, 'log.csv'), 'a') as f:
                elapsed_time = (
                        datetime.now(pytz.timezone(self.time_zone)) -
                        self.timestamp_start).total_seconds()
                log = [self.epoch, self.iteration] + [''] * 3 + metrics.tolist() + [elapsed_time] + [
                    'best model epoch: %d' % self.best_epoch]  # 获取所有数据
                log = map(str, log)  # 将log转换为str
                f.write(','.join(log) + '\n')  # 执行写入操作

    def train_epoch(self):
        # smooth = 1e-7
        self.model_gen.train()
        loss_adv_data = 0
        loss_disc_data = 0
        max_iteration = self.stop_epoch * len(self.train_loader)
        # validation_loader = enumerate(self.validation_loader)
        start_time = timeit.default_timer()
        for batch_idx, sample in tqdm.tqdm(
                enumerate(self.train_loader), total=len(self.train_loader),
                desc='Train epoch=%d' % self.epoch, ncols=80, leave=False):
            metrics = []
            # -----------------------------------
            # 迭代次数 是 所有epoch的batch次数累加之和
            # -----------------------------------
            iteration = batch_idx + self.epoch * len(self.train_loader)
            # --------------------------------
            # 执行以下方法将对优化器中的学习率进行更新
            # --------------------------------
            _ = adjust_learning_rate(self.optim_gen, self.lr_gen, iteration, max_iteration, 0.9)
            self.iteration = iteration
            self.optim_gen.zero_grad()
            for param in self.model_gen.parameters():
                # 训练权重
                param.requires_grad = True
            imageS = sample['image'].cuda()
            target_map = sample['map'].cuda()
            oS = self.model_gen(imageS)
            # ---------------
            # 二元交叉熵损失计算
            # ---------------
            loss_seg1_1 = bceloss(oS, target_map)  # mask
            loss_seg = loss_seg1_1
            self.running_seg_loss += loss_seg.item()
            loss_seg_data = loss_seg.data.item()
            loss_seg.backward(retain_graph=True)

            if self.epoch > self.warmup_epoch:
                self.optim_gen.step()
                for param in self.model_gen.parameters():
                    param.requires_grad = False

            # -----------------------------------------------------
            # 将loss_seg_data写入到log.csv中
            # 注意:loss_seg_data指的是当前epoch   一个batch  的训练损失
            # -----------------------------------------------------
            metrics.append((loss_seg_data, loss_adv_data, loss_disc_data))
            metrics = np.mean(metrics, axis=0)

            with open(osp.join(self.out, 'log.csv'), 'a') as f:
                elapsed_time = (
                        datetime.now(pytz.timezone(self.time_zone)) -
                        self.timestamp_start).total_seconds()
                log = [self.epoch, self.iteration] + metrics.tolist() + [''] * 3 + [elapsed_time]
                log = map(str, log)
                f.write(','.join(log) + '\n')
        # -----------------------------------------------------
        # 将 running_seg_loss 打印到控制台
        # 注意:running_seg_loss 指的是当前epoch 所有批次的平均训练损失
        # -----------------------------------------------------
        self.running_seg_loss /= len(self.train_loader)

        stop_time = timeit.default_timer()
        print('\n[Epoch: %d] lr:%f,  Average segLoss: %f, '
              'Execution time: %.5f' %
              (self.epoch, get_lr(self.optim_gen), self.running_seg_loss,
               stop_time - start_time))

    def train(self):
        for epoch in tqdm.trange(self.epoch, self.max_epoch,
                                 desc='Train', ncols=80):
            self.epoch = epoch
            self.train_epoch()  # one epoch
            if self.stop_epoch == self.epoch:
                print('Stop epoch at %d' % self.stop_epoch)
                break
            if (self.epoch + 1) % self.interval_validate == 0:
                self.validate()
