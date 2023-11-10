from torch.utils.data import Dataset
from helper_ply import read_ply
import numpy as np
import torch
from sklearn.metrics import confusion_matrix


class Change3D(Dataset):
    def __init__(self, txt_path='data/Change3D/', transform=True):
        fn = open(txt_path, 'r')
        points = []
        for line in fn:
            line = line.rstrip()
            words = line.split()
            points.append((words[0], words[1], int(words[2])))
            self.points = points
        self.transform = transform

    def __getitem__(self, item):
        fn1, fn2, label = self.points[item]
        data1 = read_ply(fn1)
        data2 = read_ply(fn2)
        x1 = np.stack((data1['x'], data1['y'], data1['z'], data1['red'], data1['green'], data1['blue']), axis=1)
        x2 = np.stack((data2['x'], data2['y'], data2['z'], data2['red'], data2['green'], data2['blue']), axis=1)
        # if self.transform:
        #     x1 = self.scale_translate(x1)
        #     x2 = self.scale_translate(x2)
        if self.transform:
            x1, x2 = scale_translate(x1, x2)

        return x1, x2, label

    def __len__(self):
        return len(self.points)

    @staticmethod
    def scale_translate(x, scale_low=2. / 3., scale_high=3. / 2., translate_range=0.2):
        print(x.shape)
        x = x.reshape(1, -1, 6)
        x = torch.from_numpy(x).cuda()
        print(x.shape)
        b, _, _ = x.shape
        for i in range(b):
            xyz1 = np.random.uniform(low=scale_low, high=scale_high, size=[3])
            xyz2 = np.random.uniform(low=-translate_range, high=translate_range, size=[3])
            x[i, :, 0:3] = torch.mul(x[i, :, 0:3],
                                     torch.from_numpy(xyz1).float().cuda()) + torch.from_numpy(xyz2).float().cuda()
        return x


def scale_translate(x1, x2, scale_low=2. / 3., scale_high=3. / 2., translate_range=0.2):
    bsize, _, _ = x1.shape
    for i in range(bsize):
        xyz1 = np.random.uniform(low=scale_low, high=scale_high, size=[3])
        xyz2 = np.random.uniform(low=-translate_range, high=translate_range, size=[3])

        x1[i, :, 0:3] = torch.mul(x1[i, :, 0:3], torch.from_numpy(xyz1).float().cuda()) + torch.from_numpy(
            xyz2).float().cuda()
        x2[i, :, 0:3] = torch.mul(x2[i, :, 0:3], torch.from_numpy(xyz1).float().cuda()) + torch.from_numpy(
            xyz2).float().cuda()

    return x1, x2


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
        # print(labels)
        pred_valid = pred.detach().cpu().numpy().reshape(-1, 1).squeeze()
        print(pred_valid)
        print('----------------')
        # pred_valid = np.rint(pred_valid)
        # print(pred_valid.shape)
        labels_valid = labels.detach().cpu().numpy().reshape(-1, 1).squeeze()
        # labels_valid = np.rint(labels_valid)
        print(labels_valid)
        labels_valid = [labels_valid]
        print(len(labels_valid))
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


def calc_miou(pre, correct, mode='train'):
    if mode == 'train':
        d = [350, 102, 58, 60, 20]  # 5: {'nochange': 0, 'removed': 1, 'added': 2, 'change': 3, 'color_change': 4}
    elif mode == 'test':
        d = [88, 26, 16, 14, 6]
    else:
        raise ValueError('Error mode!!!')
    iou_list = []
    for i in range(5):
        t = correct[i] / (pre[i] + d[i] - correct[i])
        iou_list.append(t)
    mean_iou = sum(iou_list) / 5.0
    return mean_iou, iou_list
