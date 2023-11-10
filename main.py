from dataset_pw_p import Dataset
from model_pw_p import PCDTransformer
import torch
from torch.utils.data import DataLoader
import torch.optim
import torch.nn.functional as f
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
import argparse
import os
import datetime
from function import scale_translate, warmup, IoUCalculator


def parse_args():
    # PARAMETERS
    parser = argparse.ArgumentParser('training')
    parser.add_argument('--batch_size', default=4, type=int, help='batch size in training')
    parser.add_argument('--epoch', default=200, type=int, help='number of epoch in training')
    parser.add_argument('--learning_rate', default=1e-3, type=float, help='learning rate in training')
    parser.add_argument('--weight_decay', default=5e-2, type=float, help='learning rate in training')
    parser.add_argument('--log_dir', default='output', type=str, help='experiment root')
    parser.add_argument('--seq_len', default=64, type=int, help='length of sequence')

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
    train_data = Dataset(split='train', seq_len=args.seq_len)
    val_data = Dataset(split='val', seq_len=args.seq_len)
    train_dataloader = DataLoader(train_data, batch_size=args.batch_size, shuffle=True, drop_last=True)
    val_dataloader = DataLoader(val_data, batch_size=args.batch_size, shuffle=True, drop_last=True)
    log_string('Load model.')
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    net = PCDTransformer().to(device)
    optimizer = torch.optim.AdamW(net.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    scheduler = CosineAnnealingWarmRestarts(optimizer, T_0=5, T_mult=2)
    log_string('Start training.')
    for epoch in range(args.epoch):
        net.train()
        iters = len(train_dataloader)
        iou_calc = IoUCalculator(num_classes=3)
        for idx, (pair, label) in enumerate(train_dataloader):
            pair, label = pair.type(torch.FloatTensor).to(device), label.type(torch.LongTensor).squeeze().to(device)

            pair = scale_translate(pair)
            optimizer.zero_grad()
            logit = net(pair).squeeze()
            loss = f.cross_entropy(logit, label)
            loss.backward()
            optimizer.step()

            if epoch < 10:
                warmup(optimizer=optimizer, current_epoch=epoch, lr_min=1e-4, lr_max=args.learning_rate, wp=True)
            else:
                scheduler.step(epoch - 10 + idx/iters)

            if idx % 10 == 0:
                with torch.no_grad():
                    log_string('epoch:{}/{}, batch:{}/{}, loss={:g}, lr={}'.format(epoch+1, args.epoch, idx,
                                                                                   len(train_dataloader), loss.item(),
                                                                                   optimizer.param_groups[0]['lr']))
            iou_calc.add_data(logits=logit, labels=label)

        mean_iou, iou_list = iou_calc.compute_iou()
        log_string('mean IoU:{:.2f}'.format(mean_iou * 100))
        s = 'IoU:'
        for iou_tmp in iou_list:
            s += '{:5.2f} '.format(100 * iou_tmp)
        log_string(s)

        if epoch % 10 == 0:
            log_string('Start validating.')
            net.eval()
            iou_cal = IoUCalculator(num_classes=3)
            with torch.no_grad():
                for _, (pair, label) in enumerate(val_dataloader):
                    pair, label = pair.type(torch.FloatTensor).to(device), label.type(torch.LongTensor).squeeze().to(
                        device)
                    logit = net(pair).squeeze()
                    iou_cal.add_data(logits=logit, labels=label)
                mean_iou, iou_list = iou_cal.compute_iou()
                log_string('mean IoU:{:.2f}'.format(mean_iou * 100))
                s = 'IoU:'
                for iou_tmp in iou_list:
                    s += '{:5.2f} '.format(100 * iou_tmp)
                log_string(s)
            log_string('Validate over.')

    torch.save(net, 'output/naive_pcdT.pth')


if __name__ == '__main__':
    arg = parse_args()
    main(arg)
