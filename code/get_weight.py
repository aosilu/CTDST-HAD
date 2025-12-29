import torch
from torch.utils.data import DataLoader
from torch.utils.data import TensorDataset
from torch import nn, optim
import scipy.io as sio
import numpy as np
from sklearn.metrics import roc_curve
from sklearn.metrics import auc
import matplotlib.pyplot as plt
from sklearn.utils.multiclass import type_of_target
import time
import cv2

def norm_data(input):
    max_input = np.max(input)
    min_input = np.min(input)
    output = (input - min_input)/(max_input - min_input)
    return output

def gweight(X_train_in):
    # 自编码器
    class AutoEncoder(nn.Module):
        def __init__(self):
            super(AutoEncoder, self).__init__()
            self.encoder = nn.Sequential(
                nn.Linear(89, 50),
                nn.LeakyReLU(),
                nn.Linear(50, 30),
                nn.LeakyReLU(),

            )
            self.decoder = nn.Sequential(
                nn.Linear(30, 50),
                nn.LeakyReLU(),
                nn.Linear(50, 89),
                nn.Tanh()
            )

        def forward(self, x):
            encoded = self.encoder(x)
            decoded = self.decoder(encoded)
            return encoded, decoded




    # 参数设置
    epochs = 200
    # 读取数据
    start = time.time()
    X_train_in = X_train_in.astype(float)
    X_train_in = X_train_in / np.max(X_train_in)
    r = np.size(X_train_in, 0)
    c = np.size(X_train_in, 1)
    bands = np.size(X_train_in, 2)
    X_train_input = X_train_in.reshape(-1, bands)
    # 数据类型转换
    trainData = torch.FloatTensor(X_train_input)

    # 构建张量数据集
    train_dataset = TensorDataset(trainData, trainData)

    trainDataLoader = DataLoader(dataset=train_dataset, batch_size=2000, shuffle=False)

    # 使用GPU训练，可以在菜单 "代码执行工具" -> "更改运行时类型" 里进行设置
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    # 网络放到GPU上
    autoencoder = AutoEncoder().to(device)
    # 初始化
    optimizer = optim.Adam(autoencoder.parameters(), lr=1e-3)
    loss_func = nn.MSELoss()
    loss_train = np.zeros((epochs, 1))

    # 训练
    for epoch in range(epochs):
        # 不需要label，所以用一个占位符"_"代替
        for batchidx, (x, _) in enumerate(trainDataLoader):
            x = x.to(device)
            # 编码和解码
            encoded, decoded = autoencoder(x)
            # 计算loss
            loss = loss_func(decoded, x)
            # 更新
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        loss_train[epoch, 0] = loss.item()
        print('epoch=%d finished' % (epoch + 1))
        '''
        print('%d  index=%d  Epoch: %04d, Training loss=%.8f' %
    
              (p, index, epoch + 1, loss.item()))
        '''

    '''
    # 绘制loss曲线
    fig = plt.figure(figsize=(6, 3))
    ax = plt.subplot(1, 1, 1)
    ax.grid()
    ax.plot(loss_train, color=[245 / 255, 124 / 255, 0 / 255], linestyle='-', linewidth=2)
    ax.set_xlabel('Epoches')
    ax.set_ylabel('Loss')
    plt.show()
    '''
    # 利用训练好的自编码器重构测试数据
    trainData = trainData.to(device)
    _, decodedTestdata = autoencoder(trainData)
    decodedTestdata = decodedTestdata.double()
    reconstructedData = decodedTestdata.detach().cpu().numpy()
    output = reconstructedData.reshape(r, c, bands)

    error = np.abs(X_train_in - output)
    # sio.savemat('result/smallplane11.mat', {'smallplane11':r})

    result_sorce = np.linalg.norm(error, ord=None, axis=2)
    result_sorce = norm_data(result_sorce)

    #result_sorce = z_score_normalize(result_sorce)

    #sio.savemat('result/result_smallplane11_l2.mat', {'result_smallplane11_l2': result_sorce})

    #plt.figure()
    #plt.imshow(result_sorce, cmap='jet')
    #plt.show()

    #result_sorce = binarization_result(result_sorce)
    #result_sorce = cv2.blur(result_sorce, ksize=(5, 5))

    #plt.figure()
    #plt.imshow(result_sorce, cmap='jet')
    #plt.show()





    return result_sorce

