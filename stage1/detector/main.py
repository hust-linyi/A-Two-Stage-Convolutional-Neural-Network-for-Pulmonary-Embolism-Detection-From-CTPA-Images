import argparse
import os
import time
import numpy as np
import data
from importlib import import_module
import shutil
from utils import *
import sys

sys.path.append('../')
from split_combine import SplitComb
import torch
from torch.nn import DataParallel
from torch.backends import cudnn
from torch.utils.data import DataLoader
from torch import optim
from torch.autograd import Variable
from tensorboardX import SummaryWriter
from layers import acc

parser = argparse.ArgumentParser(description='PyTorch PE Detector')
parser.add_argument('--model', '-m', metavar='MODEL', default='base',
                    help='model')
parser.add_argument('--data', '-d', metavar='DATA', default='debug',
                    help='data')
parser.add_argument('--config', '-c', default='config_training', type=str)
parser.add_argument('-j', '--workers', default=30, type=int, metavar='N',
                    help='number of data loading workers (default: 32)')
parser.add_argument('--epochs', default=100, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                    help='manual epoch number (useful on restarts)')
parser.add_argument('-b', '--batch-size', default=16, type=int,
                    metavar='N', help='mini-batch size (default: 16)')
parser.add_argument('--lr', '--learning-rate', default=0.01, type=float,
                    metavar='LR', help='initial learning rate')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                    help='momentum')
parser.add_argument('--weight-decay', '--wd', default=1e-3, type=float,
                    metavar='W', help='weight decay (default: 1e-4)')
parser.add_argument('--save-freq', default='1', type=int, metavar='S',
                    help='save frequency')
parser.add_argument('--resume', default='', type=str, metavar='PATH',
                    help='path to latest checkpoint (default: none)')
parser.add_argument('--save-dir', default='', type=str, metavar='SAVE',
                    help='directory to save checkpoint (default: none)')
parser.add_argument('--test', default=0, type=int, metavar='TEST',
                    help='1 do test evaluation, 0 not')
parser.add_argument('--testthresh', default=-8, type=float,
                    help='threshod for get pbb')
parser.add_argument('--split', default=8, type=int, metavar='SPLIT',
                    help='In the test phase, split the image to 8 parts')
parser.add_argument('--gpu', default='all', type=str, metavar='N',
                    help='use gpu')
parser.add_argument('--n_test', default=1, type=int, metavar='N',
                    help='number of gpu for test')


def main():
    global args
    args = parser.parse_args()
    config_training = import_module(args.config)
    config_training = config_training.config
    torch.manual_seed(0)

    model = import_module(args.model)
    config, net, loss, get_pbb = model.get_model()
    start_epoch = args.start_epoch
    save_dir = args.save_dir

    if args.resume:
        checkpoint = torch.load(args.resume)
        net.load_state_dict(checkpoint['state_dict'])
    if start_epoch == 0:
        start_epoch = 1
    if not save_dir:
        exp_id = time.strftime('%m%d-%H%M', time.localtime())
        if args.test == 1:
            save_dir = os.path.join('results/test', args.resume.split('/')[-2], args.resume.split('/')[-1])
            print(args.resume)
            print(save_dir)
        else:
            save_dir = os.path.join('results/train', args.data + args.model + '-' + exp_id)
    else:
        save_dir = os.path.join('results', save_dir)

    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    logfile = os.path.join(save_dir, 'log')
    if args.test != 1:
        sys.stdout = Logger(logfile)
        pyfiles = [f for f in os.listdir('./') if f.endswith('.py')]
        for f in pyfiles:
            shutil.copy(f, os.path.join(save_dir, f))
    n_gpu = setgpu(args.gpu)
    args.n_gpu = n_gpu
    net = net.cuda()
    loss = loss.cuda()
    cudnn.benchmark = True # True
    net = DataParallel(net)
    traindatadir = config_training['train_preprocess_result_path']
    valdatadir = config_training['val_preprocess_result_path']
    testdatadir = config_training['test_preprocess_result_path']
    trainfilelist = [t for t in config_training['trainfilelist'] if t not in config_training['black_list']]
    testfilelist = [t for t in config_training['testfilelist'] if t not in config_training['black_list']]
    valfilelist = [t for t in config_training['testfilelist'] if t not in config_training['black_list']]

    if args.test == 1:
        margin = 32
        sidelen = 144
        import data
        split_comber = SplitComb(sidelen, config['max_stride'], config['stride'], margin, config['pad_value'])
        dataset = data.DataBowl3Detector(
            testdatadir,
            testfilelist,
            config,
            phase='test',
            split_comber=split_comber)
        test_loader = DataLoader(
            dataset,
            batch_size=1,
            shuffle=False,
            num_workers=args.workers,
            collate_fn=data.collate,
            pin_memory=False)

        for i, (data, target, coord, nzhw) in enumerate(test_loader):  # check data consistency
            if i >= len(testfilelist) / args.batch_size:
                break

        test(test_loader, net, get_pbb, save_dir, config)
        return
    # net = DataParallel(net)
    import data
    print(len(trainfilelist))
    dataset = data.DataBowl3Detector(
        traindatadir,
        trainfilelist,
        config,
        phase='train')
    train_loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.workers,
        pin_memory=True)

    dataset = data.DataBowl3Detector(
        valdatadir,
        valfilelist,
        config,
        phase='val')
    val_loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.workers,
        pin_memory=True)

    for i, (data, target, coord) in enumerate(train_loader):  # check data consistency
        # print target.shape
        if i >= len(trainfilelist) / args.batch_size:
            break
    
    for i, (data, target, coord) in enumerate(val_loader):  # check data consistency
        if i >= len(valfilelist) / args.batch_size:
            break

    optimizer = torch.optim.SGD(
        net.parameters(),
        args.lr,
        momentum=0.9,
        weight_decay=args.weight_decay)

    def get_lr(epoch):
        if epoch <= args.epochs * 1 / 3:  # 0.5:
            lr = args.lr
        elif epoch <= args.epochs * 2 / 3:  # 0.8:
            lr = 0.1 * args.lr
        elif epoch <= args.epochs * 0.8:
            lr = 0.05 * args.lr
        else:
            lr = 0.01 * args.lr
        return lr

    writer = SummaryWriter()
    for epoch in range(start_epoch, args.epochs + 1):
        train(train_loader, net, loss, epoch, optimizer, get_lr, args.save_freq, save_dir, writer)
        validate(val_loader, net, loss, writer, epoch)


def train(data_loader, net, loss, epoch, optimizer, get_lr, save_freq, save_dir, writer):
    start_time = time.time()
    torch.cuda.empty_cache()

    net.train()
    lr = get_lr(epoch)
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

    metrics = []

    for i, (data, target, coord) in enumerate(data_loader):
        data = Variable(data.cuda(async=True))
        target = Variable(target.cuda(async=True))
        coord = Variable(coord.cuda(async=True))

        output = net(data, coord)
        loss_output = loss(output, target)
        optimizer.zero_grad()
        loss_output[0].backward()
        optimizer.step()

        loss_output[0] = loss_output[0].item()
        metrics.append(loss_output)

    if epoch % args.save_freq == 0:
        state_dict = net.module.state_dict()
        for key in state_dict.keys():
            state_dict[key] = state_dict[key].cpu()

        torch.save({
            'epoch': epoch,
            'save_dir': save_dir,
            'state_dict': state_dict,
            'args': args},
            os.path.join(save_dir, '%03d.ckpt' % epoch))

    end_time = time.time()

    metrics = np.asarray(metrics, np.float32)
    print('Epoch %03d (lr %.5f)' % (epoch, lr))
    print('Train:      tpr %3.2f, total pos %d, total neg %d, time %3.2f' % (
        100.0 * np.sum(metrics[:, 6]) / np.sum(metrics[:, 7]),
        np.sum(metrics[:, 7]),
        np.sum(metrics[:, 9]),
        end_time - start_time))
    print('loss %2.4f, classify loss %2.4f, regress loss %2.4f, %2.4f, %2.4f, %2.4f' % (
        np.mean(metrics[:, 0]),
        np.mean(metrics[:, 1]),
        np.mean(metrics[:, 2]),
        np.mean(metrics[:, 3]),
        np.mean(metrics[:, 4]),
        np.mean(metrics[:, 5])))
    print
    writer.add_scalar('Train/loss', np.mean(metrics[:, 0]), epoch)
    writer.add_scalar('Train/classify loss', np.mean(metrics[:, 1]), epoch)
    writer.add_scalar('Train/regress loss', np.mean(metrics[:, 2] + metrics[:, 3] + metrics[:, 4] + metrics[:, 5]), epoch)
    writer.add_scalar('Train/tpr', 100.0 * np.sum(metrics[:, 6]) / np.sum(metrics[:, 7]), epoch)
    #writer.add_scalar('Train/fpr', 100.0 * np.sum(metrics[:, 8]) / np.sum(metrics[:, 9]), epoch)
    if epoch == args.epochs:
        writer.export_scalars_to_json(save_dir + '/all_scalars.json')
        writer.close()


def validate(data_loader, net, loss, writer, epoch):
    start_time = time.time()

    net.eval()

    metrics = []
    with torch.no_grad():
        for i, (data, target, coord) in enumerate(data_loader):
            data = Variable(data.cuda(async=True))
            target = Variable(target.cuda(async=True))
            coord = Variable(coord.cuda(async=True))

            output = net(data, coord)
            loss_output = loss(output, target, train=False)

            loss_output[0] = loss_output[0].item()
            metrics.append(loss_output)
    end_time = time.time()

    metrics = np.asarray(metrics, np.float32)
    print('Validation: tpr %3.2f, total pos %d, total neg %d, time %3.2f' % (
        100.0 * np.sum(metrics[:, 6]) / np.sum(metrics[:, 7]),
        np.sum(metrics[:, 7]),
        np.sum(metrics[:, 9]),
        end_time - start_time))
    print('loss %2.4f, classify loss %2.4f, regress loss %2.4f, %2.4f, %2.4f, %2.4f' % (
        np.mean(metrics[:, 0]),
        np.mean(metrics[:, 1]),
        np.mean(metrics[:, 2]),
        np.mean(metrics[:, 3]),
        np.mean(metrics[:, 4]),
        np.mean(metrics[:, 5])))
    print
    print

    writer.add_scalar('Val/loss', np.mean(metrics[:, 0]), epoch)
    writer.add_scalar('Val/classify loss', np.mean(metrics[:, 1]), epoch)
    writer.add_scalar('Val/regress loss', np.mean(metrics[:, 2] + metrics[:, 3] + metrics[:, 4] + metrics[:, 5]), epoch)
    writer.add_scalar('Val/tpr', 100.0 * np.sum(metrics[:, 6]) / np.sum(metrics[:, 7]), epoch)
    #writer.add_scalar('Val/fpr', 100.0 * np.sum(metrics[:, 8]) / np.sum(metrics[:, 9]), epoch)

def test(data_loader, net, get_pbb, save_dir, config):
    start_time = time.time()
    save_dir = os.path.join(save_dir, 'bbox')
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    print(save_dir)
    net.eval()
    namelist = []
    split_comber = data_loader.dataset.split_comber
    for i_name, (data, target, coord, nzhw) in enumerate(data_loader):
        s = time.time()
        target = [np.asarray(t, np.float32) for t in target]
        lbb = target[0]
        nzhw = nzhw[0]
        name = data_loader.dataset.filenames[i_name].split('/')[-1].split('_clean')[0]  # .split('-')[0]  wentao change
        data = data[0][0]
        coord = coord[0][0]
        isfeat = False
        if 'output_feature' in config:
            if config['output_feature']:
                isfeat = True
        n_per_run = args.n_test
        print(data.size())
        splitlist = range(0, len(data) + 1, n_per_run)
        if splitlist[-1] != len(data):
            splitlist.append(len(data))
        outputlist = []
        featurelist = []

        for i in range(len(splitlist) - 1):
            with torch.no_grad():
                input = Variable(data[splitlist[i]:splitlist[i + 1]]).cuda()
                inputcoord = Variable(coord[splitlist[i]:splitlist[i + 1]]).cuda()
                if isfeat:
                    output, feature = net(input, inputcoord)
                    featurelist.append(feature.data.cpu().numpy())
                else:
                    output = net(input, inputcoord)
            outputlist.append(output.data.cpu().numpy())
        output = np.concatenate(outputlist, 0)
        output = split_comber.combine(output, nzhw=nzhw)
        if isfeat:
            feature = np.concatenate(featurelist, 0).transpose([0, 2, 3, 4, 1])[:, :, :, :, :, np.newaxis]
            feature = split_comber.combine(feature, sidelen)[..., 0]

        thresh = args.testthresh  # -8 #-3
        print 'pbb thresh', thresh
        pbb, mask = get_pbb(output, thresh, ismask=True)
        if isfeat:
            feature_selected = feature[mask[0], mask[1], mask[2]]
            np.save(os.path.join(save_dir, name + '_feature.npy'), feature_selected)
        # tp,fp,fn,_ = acc(pbb,lbb,0,0.1,0.1)
        # print([len(tp),len(fp),len(fn)])
        print([i_name, name])
        e = time.time()
        np.save(os.path.join(save_dir, name + '_pbb.npy'), pbb)
        np.save(os.path.join(save_dir, name + '_lbb.npy'), lbb)
    np.save(os.path.join(save_dir, 'namelist.npy'), namelist)
    end_time = time.time()

    print('elapsed time is %3.2f seconds' % (end_time - start_time))
    print
    print


#def singletest(data, net, config, splitfun, combinefun, n_per_run, margin=64, isfeat=False):
#    z, h, w = data.size(2), data.size(3), data.size(4)
#    print(data.size())
#    data = splitfun(data, config['max_stride'], margin)
#    data = Variable(data.cuda(async=True), volatile=True, requires_grad=False)
#    splitlist = range(0, args.split + 1, n_per_run)
#    outputlist = []
#    featurelist = []
#    for i in range(len(splitlist) - 1):
#        if isfeat:
#            output, feature = net(data[splitlist[i]:splitlist[i + 1]])
#            featurelist.append(feature)
#        else:
#            output = net(data[splitlist[i]:splitlist[i + 1]])
#        output = output.data.cpu().numpy()
#        outputlist.append(output)
#
#    output = np.concatenate(outputlist, 0)
#    output = combinefun(output, z / config['stride'], h / config['stride'], w / config['stride'])
#    if isfeat:
#        feature = np.concatenate(featurelist, 0).transpose([0, 2, 3, 4, 1])
#        feature = combinefun(feature, z / config['stride'], h / config['stride'], w / config['stride'])
#        return output, feature
#    else:
#        return output


if __name__ == '__main__':
    main()
