import torch
import numpy as np

bce = torch.nn.BCEWithLogitsLoss(reduction='none')


def _upscan(f):
    for i, fi in enumerate(f):
        if fi == np.inf: continue
        for j in range(1, i + 1):
            x = fi + j * j
            if f[i - j] < x: break
            f[i - j] = x


def dice_coefficient_numpy(binary_segmentation, binary_gt_label):
    """
    Compute the Dice coefficient between two binary segmentation.
    Dice coefficient is defined as here: https://en.wikipedia.org/wiki/S%C3%B8rensen%E2%80%93Dice_coefficient
    Input:
        binary_segmentation: binary 2D numpy array representing the region of interest as segmented by the algorithm
        binary_gt_label: binary 2D numpy array representing the region of interest as provided in the database
    Output:
        dice_value: Dice coefficient between the segmentation and the ground truth
    """

    # turn all variables to booleans, just in case
    binary_segmentation = np.asarray(binary_segmentation, dtype=np.bool)
    binary_gt_label = np.asarray(binary_gt_label, dtype=np.bool)
    # compute the intersection
    intersection = np.logical_and(binary_segmentation, binary_gt_label)

    # compute the TP+FP+FN
    OR_logical = np.logical_or(binary_segmentation, binary_gt_label)
    # same for the tp+fp+fn
    OR_logical = float(np.sum(OR_logical.flatten()))

    # count the number of True pixels in the binary segmentation
    segmentation_pixels = float(np.sum(binary_segmentation.flatten()))
    # same for the ground truth
    gt_label_pixels = float(np.sum(binary_gt_label.flatten()))
    # same for the intersection
    intersection = float(np.sum(intersection.flatten()))

    # compute the Dice coefficient
    dice_value = (2 * intersection + 1.0) / (1.0 + segmentation_pixels + gt_label_pixels)

    # compute the JACCARD
    JACCARD = intersection / OR_logical

    # return it
    # return dice_value, JACCARD
    return dice_value


def dice_coeff(pred, target):
    """This definition generalize to real valued pred and target vector.
    This should be differentiable.
    pred: tensor with first dimension as batch
    target: tensor with first dimension as batch
    """

    target = target.data.cpu()
    pred = torch.sigmoid(pred)
    pred = pred.data.cpu()
    pred[pred > 0.5] = 1
    pred[pred <= 0.5] = 0

    return dice_coefficient_numpy(pred, target)


def dice_coeff_2label(pred, target):
    """
    This definition generalize to real valued pred and target vector.
    This should be differentiable.
    pred: tensor with first dimension as batch
    target: tensor with first dimension as batch
    """

    target = target.data.cpu()
    pred = torch.sigmoid(pred)
    pred = pred.data.cpu()
    pred[pred > 0.5] = 1
    pred[pred <= 0.5] = 0
    # print target.shape
    # pred[:, 0, ...] 等价于 pred[:, 0, :, :]   target[:, 0, ...] 等价于 target[:, 0, :, :]
    # dice_coefficient_numpy(pred[:, 0, ...], target[:, 0, ...]) 返回第一个类别的dice，根据to_multilabel 映射关系，返回 视杯的dice
    # dice_coefficient_numpy(pred[:, 1, ...], target[:, 1, ...]) 返回第二个类别的dice，根据to_multilabel 映射关系，返回 原视盘区域+原视杯区域 的dice
    return dice_coefficient_numpy(pred[:, 0, ...], target[:, 0, ...]), dice_coefficient_numpy(pred[:, 1, ...],
                                                                                              target[:, 1, ...])


import torch


def compute_metrics(pred, target, threshold=0.8):
    """
    计算二分类任务（单通道标签）的 Dice、SE、SP、ACC

    参数:
    - pred: 模型原始输出 [B, 1, H, W]，未经过 sigmoid
    - target: 标签 [B, 1, H, W]，值为 0 或 1
    """
    pred = torch.sigmoid(pred)
    pred_bin = (pred > threshold).float()

    # 去除通道维，变成 [B, H, W]
    pred_fg = pred_bin[:, 0, :, :]
    target_fg = target[:, 0, :, :]

    # 背景 = 1 - 前景
    pred_bg = 1 - pred_fg
    target_bg = 1 - target_fg

    # TP, TN, FP, FN
    TP = (pred_fg * target_fg).sum().item()
    TN = (pred_bg * target_bg).sum().item()
    FP = (pred_fg * target_bg).sum().item()
    FN = (pred_bg * target_fg).sum().item()

    SE = TP / (TP + FN + 1e-8)
    SP = TN / (TN + FP + 1e-8)
    ACC = (TP + TN) / (TP + TN + FP + FN + 1e-8)

    # Dice
    intersection = (pred_fg * target_fg).sum()
    union = pred_fg.sum() + target_fg.sum()
    dice = (2. * intersection) / (union + 1e-8)
    dice = dice.item()

    return dice, SE, SP, ACC


def dice_loss(logits, targets, eps=1e-6):
    """
    logits: raw output from model, shape [B, 1, H, W] or [B, H, W]
    targets: ground truth mask, same shape as logits, binary {0,1}
    """
    probs = torch.sigmoid(logits)
    probs = probs.contiguous().view(-1)
    targets = targets.contiguous().view(-1)

    intersection = (probs * targets).sum()
    union = probs.sum() + targets.sum()
    dice = (2. * intersection + eps) / (union + eps)
    return 1 - dice


import torch
import torch.nn as nn
import torch.nn.functional as F


class DiceLoss(nn.Module):
    def __init__(self, smooth=1.0):
        super(DiceLoss, self).__init__()
        self.smooth = smooth

    def forward(self, pred, target):
        """
        pred: Tensor of shape [B, C, H, W], typically raw logits
        target: Tensor of shape [B, C, H, W], one-hot or soft label
        """
        # 先将 pred 做 softmax（适用于多类）或 sigmoid（二分类）处理
        # 如果是二分类建议用 sigmoid(pred)
        pred = torch.sigmoid(pred)
        # 扁平化所有维度，但保留 batch 和 class
        pred_flat = pred.contiguous().view(pred.size(0), pred.size(1), -1)
        target_flat = target.contiguous().view(target.size(0), target.size(1), -1)

        intersection = (pred_flat * target_flat).sum(2)
        dice_score = (2. * intersection + self.smooth) / (
                pred_flat.sum(2) + target_flat.sum(2) + self.smooth
        )

        # 对所有类取平均，再对 batch 求平均
        dice_loss = 1 - dice_score.mean()

        return dice_loss
