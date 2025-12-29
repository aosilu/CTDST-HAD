#!/usr/bin/env python
# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import argparse
import builtins
import math
import pickle
import matplotlib.pyplot as plt
import matplotlib
import os
import random
import shutil
import time
import warnings
import scipy.io as sio
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
import torch.utils.data.distributed
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import numpy as np
from sklearn.decomposition import PCA
import cv2  # 用于双线性插值
# Import your custom ViT model
from mymodel_spital import spital_ViT
from mymodel_spectral import spectral_ViT
import simsiam.loader_spectral
import simsiam.builder_spectral
from joint_model import combinedModel
'''
class FocalLoss(nn.Module):
    def __init__(self, alpha=0.25, gamma=2):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, inputs, targets):
        # inputs: 模型输出的logits，形状为 (N,)
        # targets: 真实标签，形状为 (N,)，值为0或1

        # 计算每个样本的 pt 和对应的 alpha_t
        # 使用 logits 计算 pt，而不是概率值
        #pt = torch.sigmoid(inputs)  # 将 logits 转换为概率
        pt = torch.where(targets == 1, inputs, 1 - inputs)
        alpha_t = torch.where(targets == 1, self.alpha, 1.0 - self.alpha)

        # 计算 Focal Loss
        focal_loss = -alpha_t * (1 - pt) ** self.gamma * torch.log(pt + 1e-7)

        return focal_loss.mean()
'''
class FocalLoss(nn.Module):
    def __init__(self, alpha=0.25, gamma=2):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, inputs, targets):
        # inputs: 模型输出的预测概率，形状为 (N,)，每个值表示属于类别1的概率
        # targets: 真实标签，形状为 (N,)，每个值为0或1

        # 计算 Focal Loss
        pt = inputs.where(targets == 1, 1 - inputs)  # 对每个样本找到对应的 pt
        focal_loss = -self.alpha * (1 - pt) ** self.gamma * torch.log(pt + 1e-7)

        return focal_loss.mean()


class SubsampledDataset(torch.utils.data.Dataset):
    def __init__(self, dataset, step_size, label_index=2):
        self.dataset = dataset
        self.step_size = step_size
        self.label_index = label_index

        self.indices = []
        for i in range(len(dataset)):
            # 取第一個元素作為標籤
            label = dataset[i][label_index].view(-1)[0].item()
            if label == 1:
                self.indices.append(i)
            elif label == 0 and i % step_size == 0:
                self.indices.append(i)

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        return self.dataset[self.indices[idx]]

def lowresiez(img, th):
    r = np.size(img, 0)
    c = np.size(img, 1)
    bands = np.size(img, 2)
    img_new = np.empty((int(r*th), int(c*th), bands))
    for i in range(0, bands):
        cc = img[:, :, i]
        kk = cv2.resize(cc, dsize=None, fx=th, fy=th, interpolation=cv2.INTER_NEAREST)
        img_new[:, :, i] = kk
    return img_new

def main():
    # 使用硬参数代替 parser
    class Args:
        def __init__(self):
            #self.data = './dataset_spectral'  # 数据集路径
            self.workers = 32
            self.epochs = 1
            self.start_epoch = 0
            self.arch = 'vit'
            self.batch_size = 1
            self.lr = 0.05
            self.momentum = 0.9
            self.weight_decay = 1e-4
            self.print_freq = 10
            self.resume = ''
            self.gpu = 0  # 使用 GPU 0
            #self.dim = 2048
            #self.pred_dim = 512
            self.fix_pred_lr = False
            self.seed = None  # 添加 seed 属性
            self.pretrained_spital = './second_spital_checkpoint_0000.pth.tar'  # 预训练模型路径
            self.pretrained_spectral = './spectral_checkpoint_0000.pth.tar'

    args = Args()

    if args.seed is not None:
        random.seed(args.seed)
        torch.manual_seed(args.seed)
        cudnn.deterministic = True
        warnings.warn('You have chosen to seed training. This will turn on the CUDNN deterministic setting, which can slow down your training considerably! You may see unexpected behavior when restarting from checkpoints.')

    if args.gpu is not None:
        warnings.warn('You have chosen a specific GPU. This will completely disable data parallelism.')

    # 创建模型
    print("=> creating model '{}'".format(args.arch))
    model_spital = spital_ViT(
        image_size=224,
        patch_size=32,
        num_classes=1000,
        dim=1024,
        depth=6,
        heads=16,
        mlp_dim=2048,
        dropout=0.1,
        emb_dropout=0.1
    )
    model_spectral = spectral_ViT(
        spectral_band=89,
        spectral_num_patches=40,
        spectral_num_classes=1,
        spectral_dim=64,
        spectral_depth=5,
        spectral_heads=6,
        spectral_mlp_dim=8,
        spectral_dropout=0.1,
        spectral_emb_dropout=0.1,
        spectral_mode='ViT'
    )

    # 加载预训练权重
    if args.pretrained_spital:
        if os.path.isfile(args.pretrained_spital):
            print("=> loading checkpoint '{}'".format(args.pretrained_spital))
            checkpoint = torch.load(args.pretrained_spital, map_location="cpu")
            model_spital.load_state_dict(checkpoint['state_dict'], strict=False)
            '''
            # 重置 spectral_ViT 的权重
            def weight_reset(m):
                if hasattr(m, 'reset_parameters'):
                    m.reset_parameters()
            model_spital.apply(weight_reset)
            '''
            print("=> loaded pre-trained model '{}'".format(args.pretrained_spital))
        else:
            print("=> no checkpoint found at '{}'".format(args.pretrained_spital))


    if args.pretrained_spectral:
        if os.path.isfile(args.pretrained_spectral):
            print("=> loading checkpoint '{}'".format(args.pretrained_spectral))
            checkpoint = torch.load(args.pretrained_spectral, map_location="cpu")
            model_spectral.load_state_dict(checkpoint['state_dict'], strict=False)
            '''
            # 重置 spectral_ViT 的权重
            def weight_reset(m):
                if hasattr(m, 'reset_parameters'):
                    m.reset_parameters()
            model_spectral.apply(weight_reset)
            '''

        else:
            print("=> no checkpoint found at '{}'".format(args.pretrained_spectral))


    combined_model = combinedModel(model_spital, model_spectral, spital_dim=64, spectral_dim=64)

    '''
    # 冻结两条支線模型的参数
    for param in combined_model.spital_ViT.parameters():
        param.requires_grad = False

    for param in combined_model.spectral_ViT.parameters():
        param.requires_grad = False
    '''
    # 只训练分类层
    #optimizer = torch.optim.Adam(combined_model.classifier.parameters(), lr=1e-4)
    optimizer = torch.optim.Adam(combined_model.parameters(), lr=1e-4)
    # 损失函数（二分类）
    criterion = FocalLoss(alpha=0.25, gamma=2)
    #criterion = nn.BCELoss()



    # 将模型移动到 GPU
    model = combined_model.cuda(args.gpu)

    # 数据加载
    i = 0
    out_win = 11
    in_win = 9
    num_around = out_win**2 - in_win**2
    train_data_in = sio.loadmat('./dataset_spectral/micai2.mat')['micai2']
    train_label = sio.loadmat('./dataset_spectral/micai2_gt.mat')['micai2_gt']



    train_data_in = lowresiez(train_data_in, 0.5 ** i)
    train_label = cv2.resize(train_label, dsize=None, fx=0.5 ** i, fy=0.5 ** i, interpolation=cv2.INTER_NEAREST)



    r, c, bands = np.shape(train_data_in)
    train_data_spectral = simsiam.loader_spectral.dual_windows_data_processing(train_data_in, out_win, in_win)
    train_data_spectral = np.array(train_data_spectral)
    train_data_spectral = np.reshape(train_data_spectral, (r, c, num_around, bands))
    train_data_spectral = split_and_reshape_4d_array(train_data_spectral)
    train_data_spectral = torch.FloatTensor(train_data_spectral)

    train_label = split_and_flatten(train_label)
    train_label = torch.FloatTensor(train_label)



    train_data_spital = train_data_in.reshape(-1, bands)



    # 创建 PCA 对象并拟合数据，保留 3 个主成分
    pca = PCA(n_components=3)
    train_data_spital_pca = pca.fit_transform(train_data_spital)

    # 将数据重新调整为原始图像形状 (r, c, 3)
    train_data_spital_pca = train_data_spital_pca.reshape(r, c, -1)
    train_data_spital_pca = image_split_and_resize(train_data_spital_pca)
    train_data_spital_pca = torch.FloatTensor(train_data_spital_pca)

    train_data = torch.utils.data.TensorDataset(train_data_spital_pca, train_data_spectral, train_label)

    # 假设我们希望每隔10个样本取一个样本
    step_size = 10

    # 创建下采样后的数据集
    subsampled_train_data = SubsampledDataset(train_data, step_size)

    # 然后使用这个下采样后的数据集创建 DataLoader
    train_loader = torch.utils.data.DataLoader(
        subsampled_train_data, batch_size=args.batch_size, shuffle=True,
        num_workers=args.workers, pin_memory=True, drop_last=True)
    '''
    train_loader = torch.utils.data.DataLoader(
        train_data, batch_size=args.batch_size, shuffle=True,
        num_workers=args.workers, pin_memory=True, drop_last=True)
    '''


    '''
    traindir = os.path.join(args.data, 'train')
    valdir = os.path.join(args.data, 'val')
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

    train_loader = torch.utils.data.DataLoader(
        datasets.ImageFolder(traindir, transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize,
        ])),
        batch_size=args.batch_size, shuffle=True, num_workers=args.workers, pin_memory=True)

    val_loader = torch.utils.data.DataLoader(
        datasets.ImageFolder(valdir, transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            normalize,
        ])),
        batch_size=256, shuffle=False, num_workers=args.workers, pin_memory=True)


    if args.evaluate:
        validate(val_loader, model, criterion, args)
        return
    '''
    loss_history = []  # 放在 main() 里，for epoch 之前

    for epoch in range(args.start_epoch, args.epochs):
        # 训练一个周期

        avg_loss = train(train_loader, model, criterion, optimizer, epoch, args)
        loss_history.append(avg_loss)

        # 画图 + 保存



        plt.figure()
        plt.plot(range(1, len(loss_history) + 1), loss_history, marker='o')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('Training Loss Curve')
        plt.savefig('loss_curve.svg')  # 也可改成 .svg / .png


        # 持久化

        with open('./loss/loss_history_focalloss.pkl', 'wb') as f:
            pickle.dump(loss_history, f)


        '''
        # 验证
        acc1 = validate(val_loader, model, criterion, args)

        # 保存最佳模型
        global best_acc1
        is_best = acc1 > best_acc1
        best_acc1 = max(acc1, best_acc1)
        '''

        save_checkpoint({
            'epoch': epoch + 1,
            'arch': args.arch,
            'state_dict': model.state_dict(),
            'best_acc1': best_acc1,
            'optimizer': optimizer.state_dict(),
        }, is_best=0)





def split_and_flatten(array, x=50):
    """
    将二维数组分割为多个小块，并将每个小块展平为一维数组

    参数:
        array: 输入数组 (numpy 数组, 形状为 r×c)
        x: 小块的尺寸，默认值为50

    返回:
        output: 处理后的数组 (形状为 p×y)
    """
    # 计算输入数组的维度
    r, c = array.shape

    # 计算可以分割的小块数量
    num_tiles_r = r // x
    num_tiles_c = c // x

    # 计算 y
    y = x * x

    # 分割并展平每个小块
    tiles = []
    for i in range(num_tiles_r):
        for j in range(num_tiles_c):
            # 提取小块
            tile = array[i * x:(i + 1) * x, j * x:(j + 1) * x]
            # 展平小块
            flattened_tile = tile.flatten()
            tiles.append(flattened_tile)

    # 将列表转换为numpy数组
    output = np.array(tiles)  # 转换为 p×y

    return output




def split_and_reshape_4d_array(array, x=50):
    """
    将四维数组分割为多个小块，并将每个小块的高度和宽度合并到一个维度

    参数:
        array: 输入数组 (numpy 数组, 形状为 r×c×z×bands)
        x: 小块的尺寸，默认值为50

    返回:
        output: 处理后的数组 (形状为 p×(x*x)×z×bands)
    """
    # 计算输入数组的维度
    r, c, z, bands = array.shape

    # 计算可以分割的小块数量
    num_tiles_r = r // x
    num_tiles_c = c // x

    # 创建列表存储所有处理后的小块
    processed_tiles = []

    # 分割并处理每个小块
    for i in range(num_tiles_r):
        for j in range(num_tiles_c):
            # 提取小块
            tile = array[i * x:(i + 1) * x, j * x:(j + 1) * x, :, :]

            # 将高度和宽度合并到一个维度 (x*x)
            reshaped_tile = tile.reshape(x * x, z, bands)

            processed_tiles.append(reshaped_tile)

    # 将列表转换为numpy数组
    output = np.array(processed_tiles)  # 转换为 p×(x*x)×z×bands

    return output




def image_split_and_resize(image, tile_size=50, target_size=224):
    """
    将图像分割为多个小块并插值到指定大小

    参数:
        image: 输入图像 (numpy 数组, 形状为 r×c×3)
        tile_size: 分割的小块大小 (默认 50)
        target_size: 插值目标大小 (默认 224)

    返回:
        output: 处理后的图像块数组 (形状为 p×3×target_size×target_size)
    """
    # 计算图像尺寸
    r, c, _ = image.shape

    # 计算可以分割的完整小块数量
    num_tiles_r = r // tile_size
    num_tiles_c = c // tile_size

    # 计算剩余部分
    remainder_r = r % tile_size
    remainder_c = c % tile_size

    # 创建列表存储所有小块
    tiles = []

    # 分割完整小块
    for i in range(num_tiles_r):
        for j in range(num_tiles_c):
            # 提取小块
            tile = image[i * tile_size:(i + 1) * tile_size, j * tile_size:(j + 1) * tile_size]
            # 插值到目标大小
            resized_tile = cv2.resize(tile, (target_size, target_size), interpolation=cv2.INTER_LINEAR)
            tiles.append(resized_tile)

    # 处理边缘剩余部分（不足50x50的情况）
    # 处理右侧剩余部分
    if remainder_c > 0:
        for i in range(num_tiles_r):
            # 提取剩余列
            tile = image[i * tile_size:(i + 1) * tile_size, -tile_size:]
            # 插值到目标大小
            resized_tile = cv2.resize(tile, (target_size, target_size), interpolation=cv2.INTER_LINEAR)
            tiles.append(resized_tile)

    # 处理底部剩余部分
    if remainder_r > 0:
        for j in range(num_tiles_c):
            # 提取剩余行
            tile = image[-tile_size:, j * tile_size:(j + 1) * tile_size]
            # 插值到目标大小
            resized_tile = cv2.resize(tile, (target_size, target_size), interpolation=cv2.INTER_LINEAR)
            tiles.append(resized_tile)

    # 处理右下角剩余部分
    if remainder_r > 0 and remainder_c > 0:
        # 提取右下角剩余部分
        tile = image[-tile_size:, -tile_size:]
        # 插值到目标大小
        resized_tile = cv2.resize(tile, (target_size, target_size), interpolation=cv2.INTER_LINEAR)
        tiles.append(resized_tile)

    # 将列表转换为numpy数组并调整维度
    output = np.array(tiles).transpose((0, 3, 1, 2))  # 转换为 p×3×224×224

    return output






def train(train_loader, model, criterion, optimizer, epoch, args):




    # 切换到训练模式
    model.train()


    for i, (images_spital, images_spectral, label) in enumerate(train_loader):
        # 记录数据加载时间
        total_loss = 0.0


        if args.gpu is not None:
            images_spital = images_spital.cuda(args.gpu, non_blocking=True)
            images_spectral = images_spectral.cuda(args.gpu, non_blocking=True)
            label = label.cuda(args.gpu, non_blocking=True)

        # 计算输出和损失
        images_spectral = images_spectral.squeeze(0)
        output = model(images_spital, images_spectral)
        output = output.squeeze(1)
        label = label.squeeze(0)

        loss = criterion(output, label)
        total_loss += loss.item()  # 把每个 batch 的 loss 累加进来
        print('Epoch: {}, Batch: {}, Loss: {:.4f}'.format(epoch, i, loss.item()))


        '''
        # 记录准确率和损失
        acc1, acc5 = accuracy(output, target, topk=(1, 5))
        losses.update(loss.item(), images.size(0))
        top1.update(acc1[0], images.size(0))
        top5.update(acc5[0], images.size(0))
        '''
        # 计算梯度并执行 SGD 步骤
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        avg_loss = total_loss / len(train_loader)
        print(f"Epoch {epoch + 1}, Loss: {avg_loss:.4f}")
        return avg_loss  # 新增


def save_checkpoint(state, is_best, filename='final_checkpoint.pth.tar'):
    torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename, 'model_best.pth.tar')



if __name__ == '__main__':
    best_acc1 = 0
    main()