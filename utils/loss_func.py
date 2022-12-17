# Loss functions for 3D-GCNN

import numpy as np
import torch
import torch.nn as nn
from scipy.spatial import cKDTree


class NewDiceLoss(nn.Module):
    def __init__(self):
        super(NewDiceLoss, self).__init__()

    def forward(self, input, target, beta=3):
        smooth = 10e-4
        input_flat = input.view(-1)
        target_flat = target.view(-1)

        if torch.max(target) == 0:
            target_flat = 1 - target_flat
            input_flat = 1 - input_flat

        tp = input_flat * target_flat
        fp = input_flat * (1 - target_flat)
        fn = (1 - input_flat) * target_flat
        loss = 1 - ((2 * tp.sum(0) + smooth) / (2 * tp.sum(0) + fp.sum(0) + beta * fn.sum(0) + smooth))

        return loss


def cal_dice_loss(input, target):
    smooth = 10e-4
    tp = np.sum(input * target)
    fp = np.sum(input * (1 - target))
    fn = np.sum((1 - input) * target)
    loss = 1 - (2 * tp + smooth) / (2 * tp + fp + fn + smooth)

    return loss


def hausdorff_95(submission, groundtruth, spacing=np.array([1.6, 0.739, 0.739])):
    # There are more efficient algorithms for hausdorff distance than brute force, however, brute force is sufficient for datasets of this size.
    submission_points = spacing * np.array(np.where(submission), dtype=np.uint16).T
    submission_kdtree = cKDTree(submission_points)

    groundtruth_points = spacing * np.array(np.where(groundtruth), dtype=np.uint16).T
    groundtruth_kdtree = cKDTree(groundtruth_points)

    distances1, _ = submission_kdtree.query(groundtruth_points)
    distances2, _ = groundtruth_kdtree.query(submission_points)
    return max(np.quantile(distances1, 0.95), np.quantile(distances2, 0.95))
