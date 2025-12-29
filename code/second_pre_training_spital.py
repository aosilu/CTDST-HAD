#!/usr/bin/env python
# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.

import argparse
import builtins
import math
import os
import random
import shutil
import time
import warnings
import copy  # 新增导入

import torch
import torch.nn as nn
import torch.nn.functional as F  # 新增导入
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
import torchvision.transforms as transforms
import torchvision.datasets as datasets

import simsiam.loader
import simsiam.builder
from mymodel_spital import spital_ViT


# === 新增1：EWC类 ===
class EWC:
    def __init__(self, model, old_model, fisher_dataset, lambda_0=1e5, beta=5):
        self.model = model
        self.old_model = copy.deepcopy(old_model)
        self.old_model.eval()
        self.lambda_0 = lambda_0
        self.beta = beta
        self.fisher = {}
        self.old_params = {}

        # 存储旧参数
        for name, param in self.old_model.named_parameters():
            param.requires_grad_(False)
            self.old_params[name] = param.detach().clone()

        # 计算Fisher矩阵
        self._compute_fisher(fisher_dataset)

    def _compute_fisher(self, dataset, samples=1000):
        self.model.train()
        fisher_loader = torch.utils.data.DataLoader(
            dataset, batch_size=64, shuffle=False, num_workers=4)

        for name, param in self.model.named_parameters():
            self.fisher[name] = torch.zeros_like(param.data)

        for i, (x, _) in enumerate(fisher_loader):
            if i >= samples: break
            x = x.cuda(non_blocking=True)
            self.model.zero_grad()

            # 使用SimSiam特征相似度作为损失
            z1 = self.model.encoder(x)
            z2 = self.model.encoder(x)
            loss = 1 - F.cosine_similarity(z1, z2).mean()
            loss.backward()

            for name, param in self.model.named_parameters():
                if param.grad is not None:
                    self.fisher[name] += param.grad.pow(2) / samples

    def penalty(self, x):
        with torch.no_grad():
            z_old = F.normalize(self.old_model.encoder(x), dim=1)
            z_new = F.normalize(self.model.encoder(x), dim=1)
            s_t = F.cosine_similarity(z_old, z_new).mean().item()
            lambda_t = self.lambda_0 * math.exp(-self.beta * s_t)

        loss = 0
        for name, param in self.model.named_parameters():
            if name in self.fisher:
                loss += (self.fisher[name] * (param - self.old_params[name]).pow(2)).sum()
        return lambda_t * loss


def main():
    # === 参数部分保持原样 ===
    class Args:
        def __init__(self):
            self.data = './dataset_second_spital'
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
            self.gpu = 0
            self.dim = 128
            self.pred_dim = 64
            self.fix_pred_lr = False
            self.seed = None
            self.pretrained = './first_spital_checkpoint_0000.pth.tar'
            self.ewc_lambda = 1  # 新增EWC参数
            self.ewc_beta = 5  # 新增EWC参数

    args = Args()
    if args.seed is not None:
        random.seed(args.seed)
        torch.manual_seed(args.seed)
        cudnn.deterministic = True

    main_worker(args)


def main_worker(args):
    # === 模型初始化保持原样 ===
    print("=> creating model")
    encoder = spital_ViT(
        image_size=224, patch_size=32, num_classes=1000,
        dim=128, depth=6, heads=16, mlp_dim=2048,
        dropout=0.1, emb_dropout=0.1
    )
    model = simsiam.builder.SimSiam(
        base_encoder=encoder,
        dim=args.dim,
        pred_dim=args.pred_dim
    ).cuda(args.gpu)

    # === 新增2：EWC初始化 ===
    ewc = None
    if args.pretrained and os.path.isfile(args.pretrained):
        print("=> loading checkpoint '{}'".format(args.pretrained))
        checkpoint = torch.load(args.pretrained, map_location='cuda:{}'.format(args.gpu))
        model.load_state_dict(checkpoint['state_dict'], strict=False)
        '''
        # 重置 spectral_ViT 的权重
        def weight_reset(m):
            if hasattr(m, 'reset_parameters'):
                m.reset_parameters()
        model.apply(weight_reset)
            '''

        # 准备旧任务数据（使用相同的transform）
        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                         std=[0.229, 0.224, 0.225])
        old_dataset = datasets.ImageFolder(
            os.path.join(args.data, 'train'),
            transforms.Compose([
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                normalize
            ]))

        print("=> Initializing EWC...")
        ewc = EWC(model, model, old_dataset, args.ewc_lambda, args.ewc_beta)

    # === 数据加载和训练循环保持原样 ===
    cudnn.benchmark = True

    # 数据增强
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
    augmentation = [
        transforms.RandomResizedCrop(224, scale=(0.2, 1.)),
        transforms.RandomApply([transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)], p=0.8),
        transforms.RandomGrayscale(p=0.2),
        transforms.RandomApply([simsiam.loader.GaussianBlur([.1, 2.])], p=0.5),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        normalize
    ]

    train_dataset = datasets.ImageFolder(
        os.path.join(args.data, 'train'),
        simsiam.loader.TwoCropsTransform(transforms.Compose(augmentation)))

    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True,
        num_workers=args.workers, pin_memory=True, drop_last=True)

    # 优化器
    criterion = nn.CosineSimilarity(dim=1).cuda(args.gpu)
    init_lr = args.lr * args.batch_size / 256
    optimizer = torch.optim.SGD(model.parameters(), init_lr,
                                momentum=args.momentum,
                                weight_decay=args.weight_decay)

    # === 修改训练循环 ===
    def train(train_loader, model, criterion, optimizer, epoch, args, ewc=None):
        batch_time = AverageMeter('Time', ':6.3f')
        data_time = AverageMeter('Data', ':6.3f')
        losses = AverageMeter('Loss', ':.4f')
        progress = ProgressMeter(
            len(train_loader),
            [batch_time, data_time, losses],
            prefix="Epoch: [{}]".format(epoch))

        model.train()
        end = time.time()
        for i, (images, _) in enumerate(train_loader):
            data_time.update(time.time() - end)
            images = [img.cuda(args.gpu, non_blocking=True) for img in images]

            # 原始SimSiam计算
            p1, p2, z1, z2 = model(x1=images[0], x2=images[1])
            loss = -(criterion(p1, z2).mean() + criterion(p2, z1).mean()) * 0.5

            # === 新增3：EWC正则项 ===
            if ewc is not None:
                loss += ewc.penalty(images[0])

            # 原始优化步骤
            losses.update(loss.item(), images[0].size(0))
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            batch_time.update(time.time() - end)
            end = time.time()

            if i % args.print_freq == 0:
                progress.display(i)

    # 训练循环调用
    for epoch in range(args.start_epoch, args.epochs):
        adjust_learning_rate(optimizer, init_lr, epoch, args)
        train(train_loader, model, criterion, optimizer, epoch, args, ewc)

        save_checkpoint({
            'epoch': epoch + 1,
            'arch': args.arch,
            'state_dict': model.state_dict(),
            'optimizer': optimizer.state_dict(),
        }, is_best=False, filename='second_spital_checkpoint_{:04d}.pth.tar'.format(epoch))


# === 保持原有辅助函数不变 ===
def save_checkpoint(state, is_best, filename='checkpoint.pth.tar'):
    torch.save(state, filename)


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