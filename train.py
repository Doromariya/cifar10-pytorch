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

import torch
import torch.nn as nn
import torch.optim as optim
import torch.optim.lr_scheduler as lrs
from tensorboardX import SummaryWriter
from torch.autograd import Variable
from torchnet.meter import AverageValueMeter

import torchvision
from models import VGG, ResNetCifar10
from torchvision import transforms


def train(epoch, model, criterion, optimizer, loader, writer, args):
    loss_meter = AverageValueMeter()
    accuracy_meter = AverageValueMeter()

    model.train()
    for batch_idx, (data, target) in enumerate(loader):
        if args.cuda:
            data, target = data.cuda(), target.cuda()
        data = Variable(data)
        target = Variable(target)

        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss_meter.add(loss.data[0], data.size(0))

        prediction = output.data.max(1)[1]
        accuracy = prediction.eq(target.data).cpu().sum()
        accuracy_meter.add(float(accuracy) / data.size(0), data.size(0))

        loss.backward()
        optimizer.step()

        if batch_idx % args.log_interval == 0:
            print('Train Epoch: {}\t'
                  'Batch: [{:5d}/{:5d} ({:3.0f}%)]\t'
                  'Loss: {:.6f}'.format(
                      epoch, batch_idx * len(data), len(loader.dataset),
                      100. * batch_idx / len(loader), loss.data[0]))

    writer.add_scalar('train_loss', loss_meter.value()[0], epoch)
    writer.add_scalar('train_accuracy_meter', accuracy_meter.value()[0], epoch)


def val(epoch, model, criterion, loader, writer, args):
    loss_meter = AverageValueMeter()
    accuracy_meter = AverageValueMeter()

    model.eval()
    for data, target in loader:
        if args.cuda:
            data, target = data.cuda(), target.cuda()
        data = Variable(data, volatile=True)
        target = Variable(target, volatile=True)

        output = model(data)
        loss = criterion(output, target)
        loss_meter.add(loss.data[0], data.size(0))

        prediction = output.data.max(1)[1]
        accuracy = prediction.eq(target.data).cpu().sum()
        accuracy_meter.add(float(accuracy) / data.size(0), data.size(0))

    print('\nTest: Average loss: {:.4f}\t'
          'Accuracy: {}/{} ({:.2f}%)\n'.format(
              loss_meter.value()[0], int(accuracy_meter.sum),
              len(loader.dataset), 100. * accuracy_meter.value()[0]))

    writer.add_scalar('test_loss', loss_meter.value()[0], epoch)
    writer.add_scalar('test_accuracy_meter', accuracy_meter.value()[0], epoch)

    return loss_meter.value()[0]


def main(args):

    torch.manual_seed(args.seed)
    if args.cuda:
        torch.cuda.manual_seed(args.seed)

    writer = SummaryWriter()

    normalize = transforms.Normalize(
        (0.4914, 0.4822, 0.4465),
        (0.2023, 0.1994, 0.2010))

    train_loader = torch.utils.data.DataLoader(
        torchvision.datasets.CIFAR10(
            root='./data', train=True, download=True,
            transform=transforms.Compose([
                transforms.RandomCrop(32, padding=4),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                normalize,
            ])),
        batch_size=args.batch_size, shuffle=True, num_workers=2)

    test_loader = torch.utils.data.DataLoader(
        torchvision.datasets.CIFAR10(
            root='./data', train=False, download=True,
            transform=transforms.Compose([
                transforms.ToTensor(),
                normalize,
            ])),
        batch_size=args.test_batch_size, shuffle=False, num_workers=2)

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
        'adam': optim.Adam(
            model.parameters(), lr=args.lr,
            weight_decay=args.weight_decay),
        'momentum_sgd': optim.SGD(
            model.parameters(), lr=args.lr,
            weight_decay=args.weight_decay, momentum=0.9),
        'nesterov_sgd': optim.SGD(
            model.parameters(), lr=args.lr,
            weight_decay=args.weight_decay, momentum=0.9, nesterov=True),
    }.get(args.optimizer)

    criterion = nn.CrossEntropyLoss()
    if args.cuda:
        criterion.cuda()

    if args.lr_factor != 1.0:
        scheduler = lrs.ReduceLROnPlateau(
            optimizer, mode='min', factor=args.lr_factor,
            patience=args.lr_patience, verbose=True)

    best_loss = 1e10
    patience = args.patience

    for epoch in range(1, args.epochs + 1):
        train(epoch, model, criterion, optimizer, train_loader, writer, args)
        val_loss = val(epoch, model, criterion, test_loader, writer, args)

        if val_loss < best_loss:
            torch.save({'epoch': epoch,
                        'arch': args.model,
                        'weight': model.state_dict(),
                        'optimizer': optimizer.state_dict()},
                       osp.join(args.logdir, 'checkpoint_best.pth.tar'))
            best_loss = val_loss
            patience = args.patience
        else:
            patience -= 1
            if patience == 0:
                print('=> Early stopping at {} epochs'.format(epoch))
                break

        # Reduce learning rate when the loss plateau
        if args.lr_factor != 1.0:
            scheduler.step(val_loss)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Image Classification on CIFAR-10')
    parser.add_argument('--batch_size', type=int, default=128, metavar='N')
    parser.add_argument('--test_batch_size', type=int,
                        default=128, metavar='N')
    parser.add_argument('--epochs', type=int, default=500, metavar='N')
    parser.add_argument('--no_cuda', action='store_true', default=False)
    parser.add_argument('--seed', type=int, default=1, metavar='S')
    parser.add_argument('--log_interval', type=int, default=50, metavar='N')
    parser.add_argument('--fold', type=int, default=0)
    parser.add_argument('--optimizer', type=str, default='momentum_sgd')
    parser.add_argument('--model', type=str, required=True)
    parser.add_argument('--lr', type=float, default=0.1)
    parser.add_argument('--lr-patience', type=int, default=5)
    parser.add_argument('--lr-factor', type=float, default=1.0)
    parser.add_argument('--weight_decay', type=float, default=1e-4)
    parser.add_argument('--patience', type=int, default=10)
    parser.add_argument('--logdir', type=str, default='log')
    args = parser.parse_args()
    args.cuda = not args.no_cuda and torch.cuda.is_available()

    print('Arguments')
    for arg in vars(args):
        print('{0:20s}: {1}'.format(arg.rjust(20), getattr(args, arg)))

    main(args)
