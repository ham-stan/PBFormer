import numpy as np
import torch
from sklearn.metrics import confusion_matrix
from math import cos, pi


def square_distance(src, dst):
    """
    Calculate Euclid distance between each two points.
    src^T * dst = xn * xm + yn * ym + zn * zm；
    sum(src^2, dim=-1) = xn*xn + yn*yn + zn*zn;
    sum(dst^2, dim=-1) = xm*xm + ym*ym + zm*zm;
    dist = (xn - xm)^2 + (yn - ym)^2 + (zn - zm)^2
         = sum(src**2,dim=-1)+sum(dst**2,dim=-1)-2*src^T*dst
    Input:
        src: source points, [B, N, C]
        dst: target points, [B, M, C]
    Output:
        dist: per-point square distance, [B, N, M]
    """
    b, n, _ = src.shape
    _, m, _ = dst.shape
    dist = -2 * torch.matmul(src, dst.permute(0, 2, 1))
    dist += torch.sum(src ** 2, -1).view(b, n, 1)
    dist += torch.sum(dst ** 2, -1).view(b, 1, m)
    return dist


def knn_point(nsample, xyz, new_xyz):
    """
    Input:
        nsample: max sample number in local region
        xyz: all points, [B, N, C]
        new_xyz: query points, [B, S, C]
    Return:
        group_idx: grouped points index, [B, S, nsample]
    """
    sqrdists = square_distance(new_xyz, xyz)
    _, group_idx = torch.topk(sqrdists, nsample, dim=-1, largest=False, sorted=False)
    return group_idx


def index_points(points, idx):
    """
    Input:
        points: input points data, [B, N, C]
        idx: sample index data, [B, S]
    Return:
        new_points:, indexed points data, [B, S, C]
    """
    device = points.device
    b = points.shape[0]
    view_shape = list(idx.shape)
    view_shape[1:] = [1] * (len(view_shape) - 1)
    repeat_shape = list(idx.shape)
    repeat_shape[0] = 1
    batch_indices = torch.arange(b, dtype=torch.long).to(device).view(view_shape).repeat(repeat_shape)
    new_points = points[batch_indices, idx, :]
    return new_points


def adjust_learning_rate(optimizer, current_epoch, max_epoch, lr_min=1e-5, lr_max=5e-4, wp=True):
    warmup_epoch = 10 if wp else 0
    if current_epoch < warmup_epoch:
        lr = lr_max * current_epoch / warmup_epoch
    else:
        lr = lr_min + (lr_max - lr_min) * (1 + cos(pi * (current_epoch - warmup_epoch)/(max_epoch - warmup_epoch))) / 2
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def warmup(optimizer, current_epoch, lr_min, lr_max, wp=True):
    warmup_epoch = 10 if wp else 0
    lr = 1e-4
    if current_epoch < warmup_epoch:
        lr = lr_min + lr_max * current_epoch / warmup_epoch
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def scale_translate(pc, scale_low=2. / 3., scale_high=3. / 2., translate_range=0.2):
    bsize, _, _, _ = pc.shape
    for i in range(bsize):
        xyz1 = np.random.uniform(low=scale_low, high=scale_high, size=[3])
        xyz2 = np.random.uniform(low=-translate_range, high=translate_range, size=[3])

        pc[i, :, :, 0:3] = torch.mul(pc[i, :, :, 0:3], torch.from_numpy(xyz1).float().cuda()) + torch.from_numpy(
            xyz2).float().cuda()

    return pc


class IoUCalculator:
    def __init__(self, num_classes=3):
        self.gt_classes = [0 for _ in range(num_classes)]
        self.positive_classes = [0 for _ in range(num_classes)]
        self.true_positive_classes = [0 for _ in range(num_classes)]
        self.num_classes = num_classes

    def add_data(self, logits, labels):
        # logits = end_points['valid_logits']
        # labels = end_points['valid_labels']
        pred = logits.max(dim=1)[1]
        # print(pred.shape)
        # print(pred)
        # print(labels.shape)
        pred_valid = pred.detach().cpu().numpy().reshape(-1, 1).squeeze()
        # pred_valid = np.rint(pred_valid)
        # print(pred_valid.shape)
        labels_valid = labels.detach().cpu().numpy().reshape(-1, 1).squeeze()
        # labels_valid = np.rint(labels_valid)
        # print(labels_valid.shape)

        val_total_correct = 0
        val_total_seen = 0

        correct = np.sum(pred_valid == labels_valid)
        val_total_correct += correct
        val_total_seen += len(labels_valid)

        conf_matrix = confusion_matrix(y_true=labels_valid, y_pred=pred_valid,
                                       labels=np.arange(0, self.num_classes, 1))
        self.gt_classes += np.sum(conf_matrix, axis=1)
        self.positive_classes += np.sum(conf_matrix, axis=0)
        self.true_positive_classes += np.diagonal(conf_matrix)

    def compute_iou(self):
        iou_list = []
        for n in range(0, self.num_classes, 1):
            if float(self.gt_classes[n] + self.positive_classes[n] - self.true_positive_classes[n]) != 0:
                iou = self.true_positive_classes[n] / float(self.gt_classes[n] + self.positive_classes[n] -
                                                            self.true_positive_classes[n])
                iou_list.append(iou)
            else:
                iou_list.append(0.0)
        mean_iou = sum(iou_list) / float(self.num_classes)
        return mean_iou, iou_list
