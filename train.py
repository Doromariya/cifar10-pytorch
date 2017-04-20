#!/usr/bin/env python
# coding: utf-8
#
# Author:   Kazuto Nakashima
# URL:      https://github.com/kazuto1011
# Created:  2017-04-20


from __future__ import print_function
import argparse
import os.path as osp
from datetime import datetime
import tensorboard_logger as logger
import torch
import torch.nn.functional as F
import torch.optim as optim
import torchvision
from torchvision import transforms
from torch.autograd import Variable
from models import VGG, ResNetCifar10


parser = argparse.ArgumentParser(
    description='Place Categorization on Sparse MPO')
parser.add_argument('--batch-size', type=int, default=128, metavar='N')
parser.add_argument('--test-batch-size', type=int, default=128, metavar='N')
parser.add_argument('--epochs', type=int, default=500, metavar='N')
parser.add_argument('--no-cuda', action='store_true', default=False)
parser.add_argument('--seed', type=int, default=1, metavar='S')
parser.add_argument('--log-interval', type=int, default=50, metavar='N')
parser.add_argument('--save-interval', type=int, default=50, metavar='N')
parser.add_argument('--fold', type=int, default=0)
parser.add_argument('--optimizer', type=str, default='momentum_sgd')
parser.add_argument('--model', type=str, required=True)
parser.add_argument('--lr', type=float, default=0.1)
parser.add_argument('--lr-decay', type=float, default=0.1)
parser.add_argument('--lr-decay-rate', type=float, default=100)
parser.add_argument('--weight-decay', type=float, default=1e-4)
parser.add_argument('--tensorboard', action='store_true')
args = parser.parse_args()
args.cuda = not args.no_cuda and torch.cuda.is_available()

print('Arguments')
for arg in vars(args):
    print('{0:20s}: {1}'.format(arg.rjust(20), getattr(args, arg)))


class average_meter(object):

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def train(epoch, model, optimizer, loader):
    losses = average_meter()
    accuracy = average_meter()

    model.train()
    for batch_idx, (data, target) in enumerate(loader):
        if args.cuda:
            data, target = data.cuda(), target.cuda()
        data, target = Variable(data), Variable(target)

        output = model(data)
        loss = F.nll_loss(output, target)
        losses.update(loss.data[0], data.size(0))

        pred = output.data.max(1)[1]
        prec = pred.eq(target.data).cpu().sum()
        accuracy.update(float(prec) / data.size(0), data.size(0))

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if batch_idx % args.log_interval == 0:
            print('Train Epoch: {}\t'
                  'Batch: [{:5d}/{:5d} ({:3.0f}%)]\t'
                  'Loss: {:.6f}'.format(
                      epoch, batch_idx * len(data), len(loader.dataset),
                      100. * batch_idx / len(loader), losses.val))

    if args.tensorboard:
        logger.log_value('train_loss', losses.avg, epoch)
        logger.log_value('train_accuracy', accuracy.avg, epoch)


def test(epoch, model, optimizer, loader):
    losses = average_meter()
    accuracy = average_meter()

    model.eval()
    for data, target in loader:
        if args.cuda:
            data, target = data.cuda(), target.cuda()
        data, target = Variable(data, volatile=True), Variable(
            target, volatile=True)

        output = model(data)
        loss = F.nll_loss(output, target)
        losses.update(loss.data[0], data.size(0))

        pred = output.data.max(1)[1]
        prec = pred.eq(target.data).cpu().sum()
        accuracy.update(float(prec) / data.size(0), data.size(0))

    print('\nTest: Average loss: {:.4f}, Accuracy: {}/{} ({:.2f}%)\n'.format(
        losses.avg, int(accuracy.sum), len(loader.dataset), 100. * accuracy.avg))

    if args.tensorboard:
        logger.log_value('test_loss', losses.avg, epoch)
        logger.log_value('test_accuracy', accuracy.avg, epoch)


def main():

    if args.tensorboard:
        logger.configure(
            osp.join('log',
                     'model_{}'.format(args.model),
                     'batchsize_{}'.format(args.batch_size),
                     'optimizer_{}'.format(args.optimizer),
                     'lr_{}'.format(args.lr),
                     'weightdecay_{}'.format(args.weight_decay),
                     datetime.now().isoformat()), flush_secs=1)

    normalize = transforms.Normalize(
        (0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))

    train_loader = torch.utils.data.DataLoader(
        torchvision.datasets.CIFAR10(
            root='./data', train=True, download=True,
            transform=transforms.Compose([
                transforms.RandomCrop(32, padding=4),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                normalize,
            ])), batch_size=128, shuffle=True, num_workers=2)

    test_loader = torch.utils.data.DataLoader(
        torchvision.datasets.CIFAR10(
            root='./data', train=False, download=True,
            transform=transforms.Compose([
                transforms.ToTensor(),
                normalize,
            ])), batch_size=128, shuffle=False, num_workers=2)

    model = {
        'vgg': VGG(),
        'resnet20': ResNetCifar10(n_block=3),
        'resnet32': ResNetCifar10(n_block=4),
        'resnet44': ResNetCifar10(n_block=5),
        'resnet56': ResNetCifar10(n_block=6),
        'resnet110': ResNetCifar10(n_block=18),
    }.get(args.model)

    if args.cuda:
        model.cuda()

    optimizer = {
        'adam': optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay),
        'momentum_sgd': optim.SGD(model.parameters(), args.lr, momentum=0.9, weight_decay=args.weight_decay),
        'nesterov_sgd': optim.SGD(model.parameters(), args.lr, momentum=0.9, weight_decay=args.weight_decay, nesterov=True),
    }.get(args.optimizer)

    for epoch in range(1, args.epochs + 1):
        if args.optimizer is not 'adam':
            lr = args.lr * (args.lr_decay ** (epoch // args.lr_decay_rate))
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr

        train(epoch, model, optimizer, train_loader)
        test(epoch, model, optimizer, test_loader)

        if epoch % args.save_interval == 0:
            torch.save(model.state_dict(),
                       'log/{}_epoch{}.model'.format(args.model, epoch))


if __name__ == '__main__':
    main()
