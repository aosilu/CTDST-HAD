#!/usr/bin/env python
# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import argparse
import builtins
import math
import time
import os
import thop
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
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve
from sklearn.metrics import auc
from get_weight import gweight



def norm_data(input):
    max_input = np.max(input)
    min_input = np.min(input)
    output = (input - min_input)/(max_input - min_input)
    return output


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

def upresize(img, th):
    r = np.size(img, 0)
    c = np.size(img, 1)
    img_new = cv2.resize(img, dsize=None, fx=th, fy=th, interpolation=cv2.INTER_NEAREST)
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
            self.joint_model_parameters = './final_checkpoint.pth.tar'  # 预训练模型路径

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



    combined_model = combinedModel(model_spital, model_spectral, spital_dim=64, spectral_dim=64)



    if args.joint_model_parameters:
        if os.path.isfile(args.joint_model_parameters):
            print("=> loading checkpoint '{}'".format(args.joint_model_parameters))
            checkpoint = torch.load(args.joint_model_parameters, map_location="cpu")
            combined_model.load_state_dict(checkpoint['state_dict'], strict=False)
            print("=> loaded pre-trained model '{}'".format(args.joint_model_parameters))
        else:
            print("=> no checkpoint found at '{}'".format(args.joint_model_parameters))




    # 将模型移动到 GPU
    model = combined_model.cuda(args.gpu)
    model.eval()
    ''''
    # 1. 把模型先放 CPU，省顯存
    model = model.cpu()

    # 2. 算「可訓練」參數總量
    total_parameter = sum(p.numel() for p in model.parameters() if p.requires_grad)

    # 3. 印出帶單位的漂亮字串
    print(f'Trainable params: {total_parameter / 1e6:.2f} M  ({total_parameter:,})')
   '''
    '''
    # model 已定義好，先切到 CPU
    model = model.cpu()
    x = torch.randn(1, 40, 89)  # 改成你的實際輸入尺寸
    y = torch.randn(1, 3, 224, 224)  # 改成你的實際輸入尺寸
    flops, params = thop.profile(model, inputs=(y,x), verbose=False)

    print(f'Params : {params / 1e6:.2f} M')
    print(f'FLOPs  : {flops / 1e9:.2f} G')  # 單樣本前向
    '''

    # 数据加载
    i = 0
    out_win = 7
    in_win = 3
    num_around = out_win**2 - in_win**2

    train_data_in = sio.loadmat('./dataset_spectral/e.mat')['e'][:, :, 0:89]
    test_label_in = sio.loadmat('./dataset_spectral/e.mat')['e_gt']
    #加載ＳＡＥ
    #detection_results_s = sio.loadmat('./result_SAE/detection_results_sae_d.mat')['detection_results_s']

    train_data_s = train_data_in



    #得到ＳＡＥ
    detection_results_s = gweight(train_data_s)
    sio.savemat('./result_SAE/detection_results_sae_e.mat', {'detection_results_s': detection_results_s})

    #detection_results_s = cv2.blur(detection_results_s, ksize=(11, 11))

    train_data_in = lowresiez(train_data_in, 0.5 ** i)
    #test_label_inl = cv2.resize(test_label_in, dsize=None, fx=0.5 ** i, fy=0.5 ** i, interpolation=cv2.INTER_NEAREST)


    r, c, bands = np.shape(train_data_in)
    train_data_spectral = simsiam.loader_spectral.dual_windows_data_processing(train_data_in, out_win, in_win)
    train_data_spectral = np.array(train_data_spectral)
    train_data_spectral = np.reshape(train_data_spectral, (r, c, num_around, bands))
    train_data_spectral = split_and_reshape_4d_array(train_data_spectral)
    train_data_spectral = torch.FloatTensor(train_data_spectral)




    train_data_spital = train_data_in.reshape(-1, bands)



    # 创建 PCA 对象并拟合数据，保留 3 个主成分
    pca = PCA(n_components=3)
    train_data_spital_pca = pca.fit_transform(train_data_spital)

    # 将数据重新调整为原始图像形状 (r, c, 3)
    train_data_spital_pca = train_data_spital_pca.reshape(r, c, -1)
    train_data_spital_pca = image_split_and_resize(train_data_spital_pca)
    train_data_spital_pca = torch.FloatTensor(train_data_spital_pca)

    train_data = torch.utils.data.TensorDataset(train_data_spital_pca, train_data_spectral)


    train_loader = torch.utils.data.DataLoader(
        train_data, batch_size=args.batch_size, shuffle=False,
        num_workers=args.workers, pin_memory=True, drop_last=True)

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

    start_time = time.time()
    detection_results = test(train_loader, model, args)
    end_time = time.time()
    elapsed_time = end_time - start_time
    print(f"restore_array 函數運行時間：{elapsed_time} 秒")

    detection_results = restore_array(detection_results, r, c)



    # 归一化
    detection_results = norm_data(detection_results)




    detection_results = upresize(detection_results, 2**i)

    detection_results = cv2.blur(detection_results, ksize=(3, 3))




    detection_results = norm_data(1 * detection_results * test_label_in + detection_results_s)




    max_val = np.max(detection_results)
    min_val = np.min(detection_results)
    print(f"Maximum value: {max_val}")
    print(f"Minimum value: {min_val}")



    plt.figure()
    plt.imshow(detection_results, cmap='jet')
    plt.show()


    sio.savemat('./result/result_ctdst_f.mat', {'ctdst_f': detection_results})
    test_label_in = test_label_in.reshape(-1)
    detection_results = detection_results.reshape(-1)

    plot_roc_curve(test_label_in, detection_results, 1)




def plot_roc_curve(y, pred, pos_label_num):
    # ref: https://scikit-learn.org/stable/auto_examples/model_selection/plot_roc.html#sphx-glr-auto-examples-model-selection-plot-roc-py
    fpr, tpr, threshold = roc_curve(y, pred, pos_label=pos_label_num)
    roc_auc = auc(fpr, tpr)
    print('roc_auc:', roc_auc)
    plt.figure()
    lw = 2
    plt.plot(fpr, tpr, color='darkorange',
             lw=lw, label='ROC curve (area = %0.4f)' % roc_auc)
    plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate ')
    plt.ylabel('True Positive Rate')
    #plt.title(title)
    plt.legend(loc="lower right")

    #plt.savefig('result/'+title+'.svg', dpi=600)
    plt.show()


def restore_array(detection_results, r, c, window_size=10):
    # 計算分割後的塊數量
    p = detection_results.size // (window_size * window_size)

    # 重塑 detection_results 為 (p, window_size*window_size) 的形狀
    reshaped_results = detection_results.reshape(p, window_size * window_size)

    # 重塑 reshaped_results 為 (p, window_size, window_size) 的形狀
    stacked_results = reshaped_results.reshape(p, window_size, window_size)

    # 計算原始矩陣的行數和列數
    rows = r // window_size
    cols = c // window_size

    # 創建一個空矩陣來存儲還原後的結果
    restored_array = np.zeros((r, c))

    # 將分割後的塊重新組合成原始矩陣
    for i in range(rows):
        for j in range(cols):
            idx = i * cols + j
            if idx < p:  # 防止 idx 超出 p 的範圍
                restored_array[i * window_size:(i + 1) * window_size, j * window_size:(j + 1) * window_size] = \
                stacked_results[idx]

    return restored_array





def split_and_flatten(array, x=10):
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




def split_and_reshape_4d_array(array, x=10):
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




def image_split_and_resize(image, tile_size=10, target_size=224):
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






def test(train_loader, model, args):
    detection_results = []

    for i, (images_spital, images_spectral) in enumerate(train_loader):

        if args.gpu is not None:
            images_spital = images_spital.cuda(args.gpu, non_blocking=True)
            images_spectral = images_spectral.cuda(args.gpu, non_blocking=True)
            #label = label.cuda(args.gpu, non_blocking=True)


        images_spectral = images_spectral.squeeze(0)
        preds = model(images_spital, images_spectral)
        #preds = preds.squeeze(1)
        detection_results = np.append(detection_results, preds.data.cpu().numpy())
    return detection_results


def validate(val_loader, model, criterion, args):
    batch_time = AverageMeter('Time', ':6.3f')
    losses = AverageMeter('Loss', ':.4e')
    top1 = AverageMeter('Acc@1', ':6.2f')
    top5 = AverageMeter('Acc@5', ':6.2f')
    progress = ProgressMeter(
        len(val_loader),
        [batch_time, losses, top1, top5],
        prefix='Test: ')

    # 切换到评估模式
    model.eval()

    with torch.no_grad():
        end = time.time()
        for i, (images, target) in enumerate(val_loader):
            if args.gpu is not None:
                images = images.cuda(args.gpu, non_blocking=True)
            target = target.cuda(args.gpu, non_blocking=True)

            # 计算输出和损失
            output = model(images)
            loss = criterion(output, target)

            # 记录准确率和损失
            acc1, acc5 = accuracy(output, target, topk=(1, 5))
            losses.update(loss.item(), images.size(0))
            top1.update(acc1[0], images.size(0))
            top5.update(acc5[0], images.size(0))

            # 记录批处理时间
            batch_time.update(time.time() - end)
            end = time.time()

            if i % args.print_freq == 0:
                progress.display(i)

        print(' * Acc@1 {top1.avg:.3f} Acc@5 {top5.avg:.3f}'.format(top1=top1, top5=top5))

    return top1.avg


def save_checkpoint(state, is_best, filename='final_checkpoint.pth.tar'):
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


def accuracy(output, target, topk=(1,)):
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res


if __name__ == '__main__':
    best_acc1 = 0
    main()