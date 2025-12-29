import matplotlib.pyplot as plt
import math
import os
import random
import shutil
import time
import warnings
import scipy.io as sio
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
import numpy as np
import pickle
import simsiam.loader_spectral
import simsiam.builder_spectral

from mymodel_spectral import spectral_ViT  # 导入自定义的 ViT 模型



# 1. 随机通道掩码 Channel Drop
def channel_drop(spectral, p=0.15):
    """
    spectral: (B, 40, 89)
    p: 每个波段被置 0 的概率
    """
    mask = torch.rand_like(spectral[:, :, :1]) > p   # (B, 40, 1)
    return spectral * mask

# 2. 光谱高斯噪声 + 1-D 高斯模糊（组合）
def gaussian_noise_blur(spectral, noise_std=0.02, blur_kernel=3, blur_sigma=1.0):
    """
    spectral: (B×40, 89) 或 (B, 40, 89) → 统一 reshape 为 (-1, 89)
    """
    *other, W = spectral.shape
    x = spectral.view(-1, 1, W)                      # (-1, 1, 89)  1-D 卷积格式
    # ---- 高斯噪声 ----
    noise = torch.randn_like(x) * noise_std
    x = x + noise

    # ---- 构建 1-D 高斯核 ----
    pad = blur_kernel // 2
    k = torch.arange(blur_kernel, dtype=torch.float32, device=x.device)
    k -= pad
    k = torch.exp(-0.5 * (k / blur_sigma) ** 2)
    k /= k.sum()
    k = k.view(1, 1, blur_kernel)                    # (1, 1, K)

    # ---- 1-D 卷积 ----
    x = F.conv1d(x, k, padding=pad)                  # (-1, 1, 89)
    return x.view(*other, W)                         # 恢复原始形状

# 3. 随机强度缩放
def random_scaling(spectral, low=0.8, high=1.2):
    """
    spectral: (B, 40, 89)
    整条光谱乘一个全局系数 s ~ U(low, high)
    """
    scale = torch.empty(spectral.size(0), 1, 1, device=spectral.device).uniform_(low, high)
    return spectral * scale

# 4. 随机波长偏移（循环平移）
def wavelength_shift(spectral, max_shift=2):
    """
    spectral: (B, 40, 89)
    max_shift: 最大平移像素（整数，沿波段维）
    """
    B, H, W = spectral.shape
    shift = torch.randint(-max_shift, max_shift + 1, (B, 1, 1), device=spectral.device)
    # 生成循环索引
    idx = (torch.arange(W, device=spectral.device) + shift) % W
    return torch.gather(spectral, 2, idx.expand(B, H, W))
class SubsampledDataset(torch.utils.data.Dataset):
    def __init__(self, dataset, step_size):
        self.dataset = dataset
        self.step_size = step_size
        self.indices = list(range(0, len(dataset), step_size))

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        return self.dataset[self.indices[idx]]
# 定义 VAE 模型
class PhysicsInformedVAE(nn.Module):
    def __init__(self, input_dim=89, latent_dim=10):
        super().__init__()
        self.input_dim   = input_dim
        self.latent_dim  = latent_dim

        # ---------- 编码器 ----------
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU()
        )
        self.fc_mu       = nn.Linear(64, latent_dim)
        self.fc_log_var  = nn.Linear(64, latent_dim)

        # ---------- 解码器 ----------
        # 半经验系数（89 维，可学习）
        self.f1 = nn.Parameter(torch.randn(input_dim))
        self.f2 = nn.Parameter(torch.randn(input_dim))
        self.f3 = nn.Parameter(torch.randn(input_dim))

    # ---------- 编码 ----------
    def encode(self, x):
        h   = self.encoder(x)
        mu  = self.fc_mu(h)
        log_var = self.fc_log_var(h)
        return mu, log_var

    # ---------- 重参数 ----------
    def reparameterize(self, mu, log_var):
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        return mu + eps * std

    # ---------- 解码 ----------
    def decode(self, z):
        _pi = 3.141592653589793
        # 取前 3 维，映射到 [0, π] / [0, 2π] 并防止越界
        raw   = z[:, :3]
        alpha = torch.sigmoid(raw[:, 0]) * (_pi - 1e-4)          # [0, π-ε]
        beta  = torch.sigmoid(raw[:, 1]) * (_pi - 1e-4)          # [0, π-ε]
        phi   = torch.sigmoid(raw[:, 2]) * (2 * _pi - 1e-4)      # [0, 2π-ε]
        alpha, beta, phi = [t.unsqueeze(-1) for t in (alpha, beta, phi)]

        # ---------- RossThick ----------
        cos_g = torch.cos(alpha) * torch.cos(beta) + \
                torch.sin(alpha) * torch.sin(beta) * torch.cos(phi)
        cos_g = torch.clamp(cos_g, -1 + 1e-4, 1 - 1e-4)
        gamma = torch.acos(cos_g)

        denom = torch.cos(alpha) + torch.cos(beta)
        denom = torch.clamp(denom, min=1e-4)
        K_vol = ((_pi / 2 - gamma) * cos_g + torch.sin(gamma)) / denom - 4 / _pi

        # ---------- LiSparseR ----------
        tan_a = torch.tan(alpha).clamp(-1e3, 1e3)
        tan_b = torch.tan(beta).clamp(-1e3, 1e3)
        chi = torch.sqrt(tan_a ** 2 + tan_b ** 2 -
                         2 * tan_a * tan_b * torch.cos(phi))
        chi = torch.clamp(chi, min=1e-6)

        cos_eta = 2 * torch.sqrt(chi ** 2 + (tan_a * tan_b * torch.sin(phi)) ** 2) / \
                  (1 / torch.cos(alpha) + 1 / torch.cos(beta))
        cos_eta = torch.clamp(cos_eta, -1 + 1e-4, 1 - 1e-4)
        eta = torch.acos(cos_eta)

        Omega = (1 / _pi) * (eta - torch.sin(eta) * cos_eta) * \
                (1 / torch.cos(alpha) + 1 / torch.cos(beta))
        K_geo = Omega - 1 / torch.cos(alpha) - 1 / torch.cos(beta) + \
                0.5 * (1 + cos_g) * 1 / torch.cos(alpha) * 1 / torch.cos(beta)

        # ---------- 物理公式重建 ----------
        x_rec = self.f1 + self.f2 * K_vol + self.f3 * K_geo
        return x_rec

    # ---------- 前向 ----------
    def forward(self, x):
        mu, log_var = self.encode(x)
        z = self.reparameterize(mu, log_var)
        rec = self.decode(z)
        return rec, mu, log_var

def main():
    # 使用硬参数代替 parser
    class Args:
        def __init__(self):
            self.workers = 32
            self.epochs = 20
            self.start_epoch = 0
            self.arch = 'vit'
            self.batch_size = 2500
            self.lr = 0.05
            self.momentum = 0.9
            self.weight_decay = 1e-4
            self.print_freq = 10
            self.resume = ''
            self.gpu = 0  # 使用 GPU 0
            self.dim = 64
            self.pred_dim = 32
            self.fix_pred_lr = False
            self.seed = None  # 添加 seed 属性

    args = Args()

    if args.seed is not None:
        random.seed(args.seed)
        torch.manual_seed(args.seed)
        cudnn.deterministic = True
        warnings.warn('You have chosen to seed training. '
                      'This will turn on the CUDNN deterministic setting, '
                      'This can slow down your training considerably! '
                      'You may see unexpected behavior when restarting '
                      'from checkpoints.')

    # 单 GPU 训练
    main_worker(args)


def main_worker(args):
    # 创建对比学习模型
    print("=> creating model")
    encoder = spectral_ViT(
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
    model = simsiam.builder_spectral.SimSiam(
        base_encoder=encoder,
        dim=args.dim,
        pred_dim=args.pred_dim
    )

    # 初始化 VAE 模型
    vae_model = PhysicsInformedVAE()

    # 将模型移动到 GPU
    model = model.cuda(args.gpu)
    vae_model = vae_model.cuda(args.gpu)

    # 定义对比学习损失函数
    criterion = nn.CosineSimilarity(dim=1).cuda(args.gpu)

    # 定义对比学习优化器
    init_lr = args.lr * args.batch_size / 256
    optimizer = torch.optim.SGD(model.parameters(), init_lr,
                                momentum=args.momentum,
                                weight_decay=args.weight_decay)

    # 定义 VAE 优化器
    vae_optimizer = torch.optim.Adam(vae_model.parameters(), lr=1e-4)

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
    out_win = 11
    in_win = 9
    train_data_in = sio.loadmat('./dataset_spectral/micai2.mat')['micai2']
    train_data = simsiam.loader_spectral.dual_windows_data_processing(train_data_in, out_win, in_win)
    train_data = np.array(train_data)
    train_data = torch.FloatTensor(train_data)
    train_data = torch.utils.data.TensorDataset(train_data)

    # 假设我们希望每隔10个样本取一个样本
    step_size = 100

    # 创建下采样后的数据集
    subsampled_train_data = SubsampledDataset(train_data, step_size)




    train_loader = torch.utils.data.DataLoader(
        subsampled_train_data, batch_size=args.batch_size, shuffle=True,
        num_workers=args.workers, pin_memory=True, drop_last=True)
    losses = []
    for epoch in range(args.start_epoch, args.epochs):
        adjust_learning_rate(optimizer, init_lr, epoch, args)

        # 训练一个周期
        train(train_loader, model, criterion, optimizer, epoch, args, vae_model, vae_optimizer, losses)

        save_checkpoint({
            'epoch': epoch + 1,
            'arch': args.arch,
            'state_dict': model.state_dict(),
            'optimizer': optimizer.state_dict(),
        }, is_best=False, filename='spectral_checkpoint_{:04d}.pth.tar'.format(epoch))
    plt.plot(range(1, args.epochs + 1), losses)
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training Loss Curve')



    #plt.savefig('spectral-loss_curve-L2.svg', dpi=300)  # 保存图像为 PNG 格式，dpi 为 300
    #print("损失曲线图已保存为 'spectral-loss_curve-L2.svg'")


    plt.show()

    with open('losses-weight.pkl', 'wb') as f:
        pickle.dump(losses, f)



def train(train_loader, model, criterion, optimizer, epoch, args, vae_model, vae_optimizer, losses):


    # 切换到训练模式
    model.train()
    vae_model.train()

    total_loss = 0.0
    for i, (spectral,) in enumerate(train_loader):
        # 数据维度: (batch_size, 40, 89)
        spectral0 = spectral
        spectral0 = spectral0.cuda(args.gpu)
        batch_size, num_spectra, spectral_dim = spectral.size()



        # 将数据移动到 GPU
        spectral = spectral.view(-1, spectral_dim).cuda(args.gpu, non_blocking=True)

        # 對數據進行歸一化處理
        # 假設你要將數據歸一化到 [0, 1] 範圍
        max_val = spectral.max(dim=1, keepdim=True).values  # 沿著 spectral_dim 取最大值
        min_val = spectral.min(dim=1, keepdim=True).values  # 沿著 spectral_dim 取最小值
        spectral_normalized = (spectral - min_val) / (max_val - min_val + 1e-8)  # 防止除以零
        spectral = spectral_normalized  # 將歸一化後的數據賦值回 spectral


        # 训练 VAE 并生成增强视图
        vae_optimizer.zero_grad()
        reconstructed, mu, log_var = vae_model(spectral)
        vae_loss = vae_loss_function(reconstructed, spectral, mu, log_var)
        #print('Epoch: {}, Batch: {}, VAELoss: {:.4f}'.format(epoch, i, vae_loss.item()))
        vae_loss.backward()
        vae_optimizer.step()

        '''
        # 使用训练后的 VAE 生成两个增强视图
        with torch.no_grad():
            # 第一次推理生成第一个视图
            spectral1, mu1, log_var1 = vae_model(spectral)
            #z1 = vae_model.reparameterize(mu1, log_var1)
            #spectral1 = vae_model.decode(z1)

            # 第二次推理生成第二个视图
            spectral2, mu2, log_var2 = vae_model(spectral)
            #z2 = vae_model.reparameterize(mu2, log_var2)
            #spectral2 = vae_model.decode(z2)
        '''


        #spectral2 = channel_drop(spectral0, p=0.15)
        #spectral2 = gaussian_noise_blur(spectral0, 0.02, 3, 1.0)
        #spectral2 = random_scaling(spectral0, 0.8, 1.2)
        spectral2 = wavelength_shift(spectral0, max_shift=2)
        spectral2 = spectral2.cuda(args.gpu)





        # 还原为原始的批次形状
        #spectral1 = spectral1.view(batch_size, num_spectra, spectral_dim)
        spectral2 = spectral2.view(batch_size, num_spectra, spectral_dim)
        # 计算对比学习的输出和损失


        mask = spectral2.clone()
        p1, p2, z1, z2 = model(x1=spectral0, x2=spectral2, mask=mask)




        #'''
        #weightloss
        similarity_p1_z2 = criterion(p1, z2)
        similarity_p2_z1 = criterion(p2, z1)
        # 对正样本对的权重进行调整
        weight_p1_z2 = 1.0 + torch.exp(-similarity_p1_z2)  # 为相似度较低的正样本对分配更大的权重
        weight_p2_z1 = 1.0 + torch.exp(-similarity_p2_z1)

        loss = -((weight_p1_z2 * similarity_p1_z2).mean() + (weight_p2_z1 * similarity_p2_z1).mean()) * 0.5
        #'''





        #0-LOSS
        #loss = -(criterion(p1, z2).mean() + criterion(p2, z1).mean()) * 0.5

        '''
        # 添加 L2 正则化
        loss_cosine = -(criterion(p1, z2).mean() + criterion(p2, z1).mean()) * 0.5
        l2_reg = torch.tensor(0.).cuda(args.gpu)
        for param in model.parameters():
            l2_reg += torch.norm(param)
        loss = loss_cosine + 1e-5 * l2_reg  # 1e-5 是正则化系数，你可以调整
        '''


        '''
        #溫度參數
        temperature = 0.2  # 温度参数

        similarity_p1_z2 = criterion(p1, z2) / temperature
        similarity_p2_z1 = criterion(p2, z1) / temperature

        loss = -(similarity_p1_z2.mean() + similarity_p2_z1.mean()) * 0.5
        '''



        total_loss += loss.item()

        #print('Epoch: {}, Batch: {}, Loss: {:.4f}'.format(epoch, i, loss.item()))

        # 计算梯度并执行 SGD 步骤
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    avg_loss = total_loss / len(train_loader)
    print(f"Epoch {epoch + 1}, Loss: {avg_loss:.4f}")
    losses.append(avg_loss)


def vae_loss_function(recon_x, x, mu, log_var):
    # 使用 mean 而不是 sum
    BCE = F.mse_loss(recon_x, x, reduction='mean')
    KLD = -0.5 * torch.mean(1 + log_var - mu.pow(2) - log_var.exp())
    return BCE + KLD


def save_checkpoint(state, is_best, filename='checkpoint.pth.tar'):
    torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename, 'model_best.pth.tar')




def adjust_learning_rate(optimizer, init_lr, epoch, args):
    cur_lr = init_lr * 0.5 * (1. + math.cos(math.pi * epoch / args.epochs))
    for param_group in optimizer.param_groups:
        param_group['lr'] = cur_lr


if __name__ == '__main__':
    main()