from dataset_change3d_pw_p import Change3D, scale_translate, calc_miou
from model_pw_p import AsymmetryPCDT, AsymmetrySiamPT
import torch
from torch.utils.data import DataLoader
import torch.optim
import torch.nn.functional as f
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
import argparse
import os
import datetime
from function import warmup


def parse_args():
    # PARAMETERS
    parser = argparse.ArgumentParser('training')
    parser.add_argument('--batch_size', default=1, type=int, help='batch size in training')
    parser.add_argument('--epoch', default=15, type=int, help='number of epoch in training')
    parser.add_argument('--learning_rate', default=1e-4, type=float, help='learning rate in training')
    parser.add_argument('--weight_decay', default=5e-2, type=float, help='learning rate in training')
    parser.add_argument('--log_dir', default='output', type=str, help='experiment root')
    parser.add_argument('--seq_len', default=256, type=int, help='length of sequence')

    return parser.parse_args()


def main(args):
    log_dir = args.log_dir
    if not os.path.exists(log_dir):
        os.mkdir(log_dir)
    log_fout = open(os.path.join(log_dir, 'log_train_change3d_' +
                                 str(datetime.datetime.now().strftime('%Y-%m-%d_%H-%M')) + '.txt'), 'a')

    def log_string(out_str):
        log_fout.write(out_str + '\n')
        log_fout.flush()
        print(out_str)

    log_string('Load dataset.')
    train_data = Change3D(txt_path='data/Change3D/Train/train_filelist.txt', transform=False)
    test_data = Change3D(txt_path='data/Change3D/Test/test_filelist.txt', transform=False)
    train_dataloader = DataLoader(train_data, batch_size=args.batch_size, shuffle=True, drop_last=True)
    test_dataloader = DataLoader(test_data, batch_size=args.batch_size, shuffle=True, drop_last=True)
    log_string('Load model.')
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # net = AsymmetryPCDT().to(device)
    net = AsymmetrySiamPT().to(device)
    optimizer = torch.optim.AdamW(net.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    scheduler = CosineAnnealingWarmRestarts(optimizer, T_0=5, T_mult=2)
    log_string('Start training.')
    for epoch in range(args.epoch):
        net.train()
        iters = len(train_dataloader)
        correct = [0, 0, 0, 0, 0]
        pre = [0, 0, 0, 0, 0]

        for idx, (x1, x2, label) in enumerate(train_dataloader):
            # print(x1.shape)
            x1 = x1.type(torch.FloatTensor).to(device)
            x2 = x2.type(torch.FloatTensor).to(device)
            # label = label.type(torch.LongTensor).squeeze().to(device)
            label = label.type(torch.LongTensor).to(device)
            x1, x2 = scale_translate(x1, x2)
            # print(label.shape)
            # print(label)
            optimizer.zero_grad()
            logit = net(x1, x2).squeeze(0)
            # print(logit.shape)
            loss = f.cross_entropy(logit, label)
            loss.backward()
            optimizer.step()

            if epoch < 10:
                warmup(optimizer=optimizer, current_epoch=epoch, lr_min=1e-5, lr_max=args.learning_rate, wp=True)
            else:
                scheduler.step(epoch - 10 + idx/iters)

            if idx % 100 == 0:
                with torch.no_grad():
                    log_string('epoch:{}/{}, batch:{}/{}, loss={:g}, lr={}'.format(epoch+1, args.epoch, idx,
                                                                                   len(train_dataloader), loss.item(),
                                                                                   optimizer.param_groups[0]['lr']))
            pred = logit.max(dim=1)[1]
            pre[pred.item()] += 1
            if pred.item() == label.item():
                correct[pred.item()] += 1
        mean_iou, iou_list = calc_miou(pre=pre, correct=correct, mode='test')

        log_string('mean IoU:{:.2f}'.format(mean_iou * 100))
        s = 'IoU:'
        for iou_tmp in iou_list:
            s += '{:5.2f} '.format(100 * iou_tmp)
        log_string(s)

        if epoch % 1 == 0:
            log_string('Start testing.')
            net.eval()
            correct = [0, 0, 0, 0, 0]
            pre = [0, 0, 0, 0, 0]
            with torch.no_grad():
                for _, (x1, x2, label) in enumerate(test_dataloader):
                    x1 = x1.type(torch.FloatTensor).to(device)
                    x2 = x2.type(torch.FloatTensor).to(device)
                    label = label.type(torch.LongTensor).to(device)
                    logit = net(x1, x2).squeeze(0)
                    pred = logit.max(dim=1)[1]
                    pre[pred.item()] += 1
                    if pred.item() == label.item():
                        correct[pred.item()] += 1
                mean_iou, iou_list = calc_miou(pre=pre, correct=correct, mode='test')
                log_string('mean IoU:{:.2f}'.format(mean_iou * 100))
                s = 'IoU:'
                for iou_tmp in iou_list:
                    s += '{:5.2f} '.format(100 * iou_tmp)
                log_string(s)
            log_string('Test over.')

    torch.save(net, 'output/' + str(datetime.datetime.now().strftime('%Y-%m-%d_%H-%M')) + '.pth')


if __name__ == '__main__':
    arg = parse_args()
    main(arg)
