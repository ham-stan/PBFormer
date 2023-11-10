from dataset_urb3dcd_pw_p import Dataset
import torch
from torch.utils.data import DataLoader
import argparse
import os
import datetime
from function import IoUCalculator


def parse_args():
    # PARAMETERS
    parser = argparse.ArgumentParser('training')
    parser.add_argument('--batch_size', default=12, type=int, help='batch size in training')

    parser.add_argument('--log_dir', default='output', type=str, help='experiment root')
    parser.add_argument('--seq_len', default=256, type=int, help='length of sequence')

    return parser.parse_args()


def main(args):
    log_dir = args.log_dir
    if not os.path.exists(log_dir):
        os.mkdir(log_dir)
    log_fout = open(os.path.join(log_dir, 'log_train_' +
                                 str(datetime.datetime.now().strftime('%Y-%m-%d_%H-%M')) + '.txt'), 'a')

    def log_string(out_str):
        log_fout.write(out_str + '\n')
        log_fout.flush()
        print(out_str)

    log_string('Load dataset.')
    test_data = Dataset('test')
    test_dataloader = DataLoader(test_data, batch_size=args.batch_size, shuffle=True, drop_last=True)
    log_string('Load model.')
    model_path = 'output/naive_pcdT.pth'
    model = torch.load(model_path)
    model = model.to("cuda")
    model.eval()
    log_string('Start testing.')
    iou_cal = IoUCalculator(num_classes=3)
    with torch.no_grad():
        for idx, (pair, label) in enumerate(test_dataloader):
            pair, label = pair.type(torch.FloatTensor).to("cuda"), label.type(torch.LongTensor).squeeze().to("cuda")
            logit = model(pair).squeeze()
            iou_cal.add_data(logits=logit, labels=label)
            if idx % 100 == 0:
                print('batch:{}/{}'.format(idx, len(test_dataloader)))
        mean_iou, iou_list = iou_cal.compute_iou()
        log_string('mean IoU:{:.2f}'.format(mean_iou * 100))
        s = 'IoU:'
        for iou_tmp in iou_list:
            s += '{:5.2f} '.format(100 * iou_tmp)
        log_string(s)
    log_string('Test over.')


if __name__ == "__main__":
    arg = parse_args()
    main(arg)
