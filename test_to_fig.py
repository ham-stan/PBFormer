import torch
import numpy as np
# from dataset_to_fig import Dataset
# from to_fig_3 import Dataset
from to_fig_4 import Dataset
import argparse
from torch.utils.data import DataLoader
import os
import datetime
from helper_ply import read_ply, write_ply
from function import IoUCalculator


def parse_args():
    # PARAMETERS
    parser = argparse.ArgumentParser('training')
    parser.add_argument('--batch_size', default=32, type=int, help='batch size in training')

    parser.add_argument('--log_dir', default='output', type=str, help='experiment root')
    parser.add_argument('--seq_len', default=256, type=int, help='length of sequence')

    return parser.parse_args()


def main(args):
    log_dir = args.log_dir
    if not os.path.exists(log_dir):
        os.mkdir(log_dir)
    log_fout = open(os.path.join(log_dir, 'log_test_to_fig_' +
                                 str(datetime.datetime.now().strftime('%Y-%m-%d_%H-%M')) + '.txt'), 'a')

    def log_string(out_str):
        log_fout.write(out_str + '\n')
        log_fout.flush()
        print(out_str)

    log_string('Load dataset.')
    test_data = Dataset('test')
    print(test_data.__len__())
    test_dataloader = DataLoader(test_data, batch_size=args.batch_size, shuffle=True, drop_last=True)
    print(len(test_dataloader))

    log_string('Load model.')
    model_path = 'model/8505.pth'
    model = torch.load(model_path)
    model = model.to("cuda")
    model.eval()

    log_string('Start testing.')

    preds = []
    preds = np.array(preds)
    ll = test_data.__len__() - len(test_dataloader) * args.batch_size
    lac = np.zeros((ll,))

    iou_cal = IoUCalculator(num_classes=3)
    with torch.no_grad():
        for idx, (pair, label) in enumerate(test_dataloader):
            pair, label = pair.type(torch.FloatTensor).to("cuda"), label.type(torch.LongTensor).squeeze().to("cuda")
            logit = model(pair).squeeze()
            iou_cal.add_data(logits=logit, labels=label)
            pred = logit.max(dim=1)[1]
            pred_valid = pred.detach().cpu().numpy().reshape(-1, 1).squeeze()
            # print(pred_valid.shape)
            # print(pred_valid.dtype)
            # p = np.array(pred_valid)
            # print(p)
            # print(p.shape)

            preds = np.hstack((preds, pred_valid))
            if idx % 100 == 0:
                print(idx)
                # break
        mean_iou, iou_list = iou_cal.compute_iou()
        log_string('mean IoU:{:.2f}'.format(mean_iou * 100))
    preds = np.hstack((preds, lac))
    print('preds.shape----', preds.shape)

    c_0 = np.array([67, 1, 84])
    c_1 = np.array([0, 150, 128])
    c_2 = np.array([255, 208, 0])
    points = read_ply('data/Urb3DCD/4/Test/LyonS/pointCloud1.ply')
    cloud = np.stack((points['x'], points['y'], points['z']), axis=1)
    n = preds.shape[0]
    print('cloud.shape----', cloud.shape)
    # print(cloud.shape[0])

    # cloud_ = np.ones((367, 3))
    color = [0, 0, 0]
    color = np.array(color)
    for i in range(n):
        if preds[i] == 0:
            color = np.vstack((color, c_0))
        if preds[i] == 1:
            color = np.vstack((color, c_1))
        if preds[i] == 2:
            color = np.vstack((color, c_2))
        # 0  r 67  g 1  b 84
        # 1  r 0  g 150  b 128
        # 2  r 255  g 208  b 0
    print('color.shape----', color.shape)

    write_ply('result/result_4.ply', (cloud, color[1:, :], preds), ['x', 'y', 'z', 'red', 'green', 'blue', 'class'])


if __name__ == "__main__":

    arg = parse_args()
    main(arg)
