#!/usr/bin/env python
# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import argparse
import builtins
import math
import os
import random
import shutil
import time
import warnings

import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
import torchvision.transforms as transforms
import torchvision.datasets as datasets

import simsiam.loader
import simsiam.builder

from mymodel_spital import spital_ViT  # 导入自定义的 ViT 模型


def main():
    # 使用硬参数代替 parser
    class Args:
        def __init__(self):
            self.data = './dataset_spital'  # 数据集路径
            self.workers = 32
            self.epochs = 1
            self.start_epoch = 0
            self.arch = 'vit'
            self.batch_size = 64
            self.lr = 0.05
            self.momentum = 0.9
            self.weight_decay = 1e-4
            self.print_freq = 10
            self.resume = ''
            self.gpu = 0  # 使用 GPU 0
            self.dim = 128
            self.pred_dim = 64
            self.fix_pred_lr = False
            self.seed = None  # 添加 seed 属性

    args = Args()

    if args.seed is not None:
        random.seed(args.seed)
        torch.manual_seed(args.seed)
        cudnn.deterministic = True
        warnings.warn('You have chosen to seed training. '
                      'This will turn on the CUDNN deterministic setting, '
                      'which can slow down your training considerably! '
                      'You may see unexpected behavior when restarting '
                      'from checkpoints.')

    # 单 GPU 训练
    main_worker(args)


def main_worker(args):
    # 创建模型
    print("=> creating model")
    encoder = spital_ViT(
        image_size=224,
        patch_size=32,
        num_classes=1000,
        dim=128,
        depth=6,
        heads=16,
        mlp_dim=2048,
        dropout=0.1,
        emb_dropout=0.1
    )
    model = simsiam.builder.SimSiam(
        base_encoder=encoder,
        dim=args.dim,
        pred_dim=args.pred_dim
    )

    # 将模型移动到 GPU
    model = model.cuda(args.gpu)

    # 定义损失函数和优化器
    criterion = nn.CosineSimilarity(dim=1).cuda(args.gpu)

    init_lr = args.lr * args.batch_size / 256
    optimizer = torch.optim.SGD(model.parameters(), init_lr,
                                momentum=args.momentum,
                                weight_decay=args.weight_decay)

    # 可选：从检查点恢复
    if args.resume:
        if os.path.isfile(args.resume):
            print("=> loading checkpoint '{}'".format(args.resume))
            checkpoint = torch.load(args.resume, map_location='cuda:{}'.format(args.gpu))
            args.start_epoch = checkpoint['epoch']
            model.load_state_dict(checkpoint['state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            print("=> loaded checkpoint '{}' (epoch {})"
                  .format(args.resume, checkpoint['epoch']))
        else:
            print("=> no checkpoint found at '{}'".format(args.resume))

    cudnn.benchmark = True

    # 数据加载
    traindir = os.path.join(args.data, 'train')
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])

    augmentation = [
        transforms.RandomResizedCrop(224, scale=(0.2, 1.)),
        transforms.RandomApply([
            transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)
        ], p=0.8),
        transforms.RandomGrayscale(p=0.2),
        transforms.RandomApply([simsiam.loader.GaussianBlur([.1, 2.])], p=0.5),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        normalize
    ]

    train_dataset = datasets.ImageFolder(
        traindir,
        simsiam.loader.TwoCropsTransform(transforms.Compose(augmentation)))

    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True,
        num_workers=args.workers, pin_memory=True, drop_last=True)

    for epoch in range(args.start_epoch, args.epochs):
        adjust_learning_rate(optimizer, init_lr, epoch, args)

        # 训练一个周期
        train(train_loader, model, criterion, optimizer, epoch, args)

        save_checkpoint({
            'epoch': epoch + 1,
            'arch': args.arch,
            'state_dict': model.state_dict(),
            'optimizer': optimizer.state_dict(),
        }, is_best=False, filename='first_spital_checkpoint_{:04d}.pth.tar'.format(epoch))


def train(train_loader, model, criterion, optimizer, epoch, args):
    batch_time = AverageMeter('Time', ':6.3f')
    data_time = AverageMeter('Data', ':6.3f')
    losses = AverageMeter('Loss', ':.4f')
    progress = ProgressMeter(
        len(train_loader),
        [batch_time, data_time, losses],
        prefix="Epoch: [{}]".format(epoch))

    # 切换到训练模式
    model.train()

    end = time.time()
    for i, (images, _) in enumerate(train_loader):
        # 记录数据加载时间
        data_time.update(time.time() - end)

        images[0] = images[0].cuda(args.gpu, non_blocking=True)
        images[1] = images[1].cuda(args.gpu, non_blocking=True)

        # 计算输出和损失
        p1, p2, z1, z2 = model(x1=images[0], x2=images[1])
        print(z1.shape)
        print(z2.shape)
        print(p1.shape)
        print(z2.shape)
        loss = -(criterion(p1, z2).mean() + criterion(p2, z1).mean()) * 0.5

        losses.update(loss.item(), images[0].size(0))

        # 计算梯度并执行 SGD 步骤
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # 记录批处理时间
        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.print_freq == 0:
            progress.display(i)


def save_checkpoint(state, is_best, filename='checkpoint.pth.tar'):
    torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename, 'model_best.pth.tar')


class AverageMeter(object):
    def __init__(self, name, fmt=':f'):
        self.name = name
        self.fmt = fmt
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

    def __str__(self):
        fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
        return fmtstr.format(**self.__dict__)


class ProgressMeter(object):
    def __init__(self, num_batches, meters, prefix=""):
        self.batch_fmtstr = self._get_batch_fmtstr(num_batches)
        self.meters = meters
        self.prefix = prefix

    def display(self, batch):
        entries = [self.prefix + self.batch_fmtstr.format(batch)]
        entries += [str(meter) for meter in self.meters]
        print('\t'.join(entries))

    def _get_batch_fmtstr(self, num_batches):
        num_digits = len(str(num_batches // 1))
        fmt = '{:' + str(num_digits) + 'd}'
        return '[' + fmt + '/' + fmt.format(num_batches) + ']'


def adjust_learning_rate(optimizer, init_lr, epoch, args):
    cur_lr = init_lr * 0.5 * (1. + math.cos(math.pi * epoch / args.epochs))
    for param_group in optimizer.param_groups:
        param_group['lr'] = cur_lr


if __name__ == '__main__':
    main()