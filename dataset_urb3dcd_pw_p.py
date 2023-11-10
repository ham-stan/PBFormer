import numpy as np
from helper_ply import read_ply
import torch.utils.data
import torch
import pickle


class Dataset(torch.utils.data.Dataset):
    def __init__(self, split='train_mini', seq_len=512):
        super(Dataset, self).__init__()
        self.split = split
        self.seq_len = seq_len
        if split == 'train_mini':
            print('TRAIN MINI DATASET')
            self.root = [['data/Urb3DCD/1/TrainMini-1a/LyonN10/pointcloud0.ply',
                          'data/Urb3DCD/1/TrainMini-1a/LyonN10/pointcloud0_KDTree.pkl',
                          'data/Urb3DCD/1/TrainMini-1a/LyonN10/pointcloud1.ply',
                          'data/Urb3DCD/1/TrainMini-1a/LyonN10/pointcloud1_KDTree.pkl',
                          ]
                         ]
        elif split == 'train_normal':
            print('TRAIN NORMAL DATASET')
            self.root = [['data/Urb3DCD/1/TrainNormal-1b/LyonN4/pointcloud0.ply',
                          'data/Urb3DCD/1/TrainNormal-1b/LyonN4/pointcloud0_KDTree.pkl',
                          'data/Urb3DCD/1/TrainNormal-1b/LyonN4/pointcloud1.ply',
                          'data/Urb3DCD/1/TrainNormal-1b/LyonN4/pointcloud1_KDTree.pkl',
                          ],
                         ['data/Urb3DCD/1/TrainNormal-1b/LyonN10/pointcloud0.ply',
                          'data/Urb3DCD/1/TrainNormal-1b/LyonN10/pointcloud0_KDTree.pkl',
                          'data/Urb3DCD/1/TrainNormal-1b/LyonN10/pointcloud1.ply',
                          'data/Urb3DCD/1/TrainNormal-1b/LyonN10/pointcloud1_KDTree.pkl',
                          ],
                         ['data/Urb3DCD/1/TrainNormal-1b/LyonN12/pointcloud0.ply',
                          'data/Urb3DCD/1/TrainNormal-1b/LyonN12/pointcloud0_KDTree.pkl',
                          'data/Urb3DCD/1/TrainNormal-1b/LyonN12/pointcloud1.ply',
                          'data/Urb3DCD/1/TrainNormal-1b/LyonN12/pointcloud1_KDTree.pkl',
                          ],
                         ['data/Urb3DCD/1/TrainNormal-1b/LyonN13/pointcloud0.ply',
                          'data/Urb3DCD/1/TrainNormal-1b/LyonN13/pointcloud0_KDTree.pkl',
                          'data/Urb3DCD/1/TrainNormal-1b/LyonN13/pointcloud1.ply',
                          'data/Urb3DCD/1/TrainNormal-1b/LyonN13/pointcloud1_KDTree.pkl',
                          ],
                         ['data/Urb3DCD/1/TrainNormal-1b/LyonN15/pointcloud0.ply',
                          'data/Urb3DCD/1/TrainNormal-1b/LyonN15/pointcloud0_KDTree.pkl',
                          'data/Urb3DCD/1/TrainNormal-1b/LyonN15/pointcloud1.ply',
                          'data/Urb3DCD/1/TrainNormal-1b/LyonN15/pointcloud1_KDTree.pkl',
                          ],
                         ['data/Urb3DCD/1/TrainNormal-1b/LyonN16/pointcloud0.ply',
                          'data/Urb3DCD/1/TrainNormal-1b/LyonN16/pointcloud0_KDTree.pkl',
                          'data/Urb3DCD/1/TrainNormal-1b/LyonN16/pointcloud1.ply',
                          'data/Urb3DCD/1/TrainNormal-1b/LyonN16/pointcloud1_KDTree.pkl',
                          ],
                         ['data/Urb3DCD/1/TrainNormal-1b/LyonN17/pointcloud0.ply',
                          'data/Urb3DCD/1/TrainNormal-1b/LyonN17/pointcloud0_KDTree.pkl',
                          'data/Urb3DCD/1/TrainNormal-1b/LyonN17/pointcloud1.ply',
                          'data/Urb3DCD/1/TrainNormal-1b/LyonN17/pointcloud1_KDTree.pkl',
                          ],
                         ['data/Urb3DCD/1/TrainNormal-1b/LyonN18/pointcloud0.ply',
                          'data/Urb3DCD/1/TrainNormal-1b/LyonN18/pointcloud0_KDTree.pkl',
                          'data/Urb3DCD/1/TrainNormal-1b/LyonN18/pointcloud1.ply',
                          'data/Urb3DCD/1/TrainNormal-1b/LyonN18/pointcloud1_KDTree.pkl',
                          ],
                         ['data/Urb3DCD/1/TrainNormal-1b/LyonN19/pointcloud0.ply',
                          'data/Urb3DCD/1/TrainNormal-1b/LyonN19/pointcloud0_KDTree.pkl',
                          'data/Urb3DCD/1/TrainNormal-1b/LyonN19/pointcloud1.ply',
                          'data/Urb3DCD/1/TrainNormal-1b/LyonN19/pointcloud1_KDTree.pkl',
                          ],
                         ['data/Urb3DCD/1/TrainNormal-1b/LyonN24/pointcloud0.ply',
                          'data/Urb3DCD/1/TrainNormal-1b/LyonN24/pointcloud0_KDTree.pkl',
                          'data/Urb3DCD/1/TrainNormal-1b/LyonN24/pointcloud1.ply',
                          'data/Urb3DCD/1/TrainNormal-1b/LyonN24/pointcloud1_KDTree.pkl',
                          ],
                         ]
        elif split == 'train_large':
            print('TRAIN LARGE DATASET')
            self.root = [['data/Urb3DCD/1/TrainLarge-1c/LyonN/pointcloud0.ply', '',
                          'data/Urb3DCD/1/TrainLarge-1c/LyonN/pointcloud1.ply', '',
                          ],
                         [

                         ],

                         ]
        elif split == 'val':
            print('VALIDATION DATASET')
            self.root = [['data/Urb3DCD/1/Val/Lyon2_NorthDown/pointcloud0.ply',
                          'data/Urb3DCD/1/Val/Lyon2_NorthDown/pointcloud0_KDTree.pkl',
                          'data/Urb3DCD/1/Val/Lyon2_NorthDown/pointcloud1.ply',
                          'data/Urb3DCD/1/Val/Lyon2_NorthDown/pointcloud1_KDTree.pkl',
                          ],

                         ]
        elif split == 'test':
            print('TEST DATASET')
            self.root = [['data/Urb3DCD/1/Test/LyonS/pointCloud0.ply',
                          'data/Urb3DCD/1/Test/LyonS/pointcloud0_KDTree.pkl',
                          'data/Urb3DCD/1/Test/LyonS/pointcloud1.ply',
                          'data/Urb3DCD/1/Test/LyonS/pointcloud1_KDTree.pkl',
                          ],
                         ['data/Urb3DCD/1/Test/LyonS1/pointcloud0.ply',
                          'data/Urb3DCD/1/Test/LyonS1/pointcloud0_KDTree.pkl',
                          'data/Urb3DCD/1/Test/LyonS1/pointcloud1.ply',
                          'data/Urb3DCD/1/Test/LyonS1/pointcloud1_KDTree.pkl',
                          ],
                         ['data/Urb3DCD/1/Test/LyonS2/pointcloud0.ply',
                          'data/Urb3DCD/1/Test/LyonS2/pointcloud0_KDTree.pkl',
                          'data/Urb3DCD/1/Test/LyonS2/pointcloud1.ply',
                          'data/Urb3DCD/1/Test/LyonS2/pointcloud1_KDTree.pkl',
                          ],

                         ]
        else:
            raise ValueError('Error Split!')
        self.date1_points = {'train_mini': [], 'train_normal': [], 'train_large': [], 'val': [], 'test': []}
        self.date1_tree = {'train_mini': [], 'train_normal': [], 'train_large': [], 'val': [], 'test': []}
        self.date1_labels = {'train_mini': [], 'train_normal': [], 'train_large': [], 'val': [], 'test': []}
        self.date2_points = {'train_mini': [], 'train_normal': [], 'train_large': [], 'val': [], 'test': []}
        self.date2_tree = {'train_mini': [], 'train_normal': [], 'train_large': [], 'val': [], 'test': []}
        self.date2_labels = {'train_mini': [], 'train_normal': [], 'train_large': [], 'val': [], 'test': []}
        self.load_ply(self.split, self.root)

        self.cind = [0, 0]
        self.pind = [0, 0]
        self.d1_len = []
        self.d2_len = []
        for i in range(len(self.date1_points[self.split])):
            self.d1_len.append(self.date1_points[self.split][i].shape[0])
            self.d2_len.append(self.date2_points[self.split][i].shape[0])

    def load_ply(self, split, root):
        for rt in root:
            data_1 = read_ply(rt[0])
            data_2 = read_ply(rt[2])
            points_1 = np.stack((data_1['x'], data_1['y'], data_1['z']), axis=1)
            points_2 = np.stack((data_2['x'], data_2['y'], data_2['z']), axis=1)
            self.date1_points[split].append(points_1)
            self.date2_points[split].append(points_2)
            labels_1 = data_1['label_ch']
            labels_2 = data_2['label_ch']
            # if split != 'test':
            #     labels_1 = data_1['label_ch']
            #     labels_2 = data_2['label_ch']
            # else:
            #     labels_1 = None
            #     labels_2 = None
            self.date1_labels[split].append(labels_1)
            self.date2_labels[split].append(labels_2)

            with open(rt[1], 'rb') as f:
                search_tree_1 = pickle.load(f)
            self.date1_tree[split].append(search_tree_1)
            with open(rt[3], 'rb') as f:
                search_tree_2 = pickle.load(f)
            self.date2_tree[split].append(search_tree_2)
        print('loading finished')

    def one_pair(self, split, center_points, center_labels, cind, pind):
        center_point = center_points[split][cind][pind].reshape(1, -1)
        label = center_labels[split][cind][pind].reshape(-1)

        query_seq_1 = self.date1_tree[split][cind].query(center_point, k=self.seq_len)[1][0]
        query_seq_1 = self.shuffle_idx(query_seq_1)
        seq_1 = self.date1_points[split][cind][query_seq_1]
        seq_1 = seq_1 - center_point  # seq_len 3
        query_seq_2 = self.date2_tree[split][cind].query(center_point, k=self.seq_len)[1][0]
        query_seq_2 = self.shuffle_idx(query_seq_2)
        seq_2 = self.date2_points[split][cind][query_seq_2]
        seq_2 = seq_2 - center_point  # seq_len 3
        pair = np.stack((seq_1, seq_2), axis=0)  # 2 seq_len 3

        return pair, label

    def __len__(self):
        num2 = 0
        for i in range(len(self.date2_points[self.split])):
            num2 += self.date2_points[self.split][i].shape[0]
        return num2

    def __getitem__(self, item):
        if self.cind[1] >= len(self.date2_points[self.split]):
            self.pind[1] = 0
            self.cind[1] = 0
        # if self.cind[1] < len(self.date2_points[self.split]):
        pair, label = self.one_pair(self.split, self.date2_points, self.date2_labels, self.cind[1], self.pind[1])
        self.pind[1] += 1
        # if self.pind[0] >= 64:
        if self.pind[1] >= self.d2_len[self.cind[1]]:
            self.pind[1] = 0
            self.cind[1] += 1
        # else:
        #     self.pind[1] = 0
        #     self.cind[1] = 0

        return pair, label

    @staticmethod
    def shuffle_idx(x):
        idx = np.arange(len(x))
        np.random.shuffle(idx)
        return x[idx]


if __name__ == '__main__':
    dataset = Dataset('val')
    print(dataset.__len__())
    print(dataset[0][0].shape)
    print(dataset[0][1].shape)
    dl = torch.utils.data.DataLoader(dataset, batch_size=8, drop_last=True)
    print(len(dl))
    for idd, y in enumerate(dl):
        print('---------------------------------------')
        print(idd)
        print(y[0].shape, '-----', y[1].shape)
        if idd == 200:
            break
