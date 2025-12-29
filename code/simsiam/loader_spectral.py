import numpy as np
import torch
class HighSpectralTransform:
    def __init__(self, noise_level=0.1):
        self.noise_level = noise_level

    def __call__(self, x):
        # 添加高斯噪声作为数据增强
        noise = np.random.normal(0, self.noise_level, x.shape)
        return x + noise


class TwoCropsTransform:
    """Take two random crops of one spectrum as the query and key."""

    def __init__(self, vae_model, device):
        self.vae_model = vae_model
        self.device = device

    def __call__(self, x):
        # 将数据移动到 GPU
        x = x.unsqueeze(0).to(self.device)  # 增加批次维度

        # 使用 VAE 生成增强数据
        with torch.no_grad():
            # 第一次推理生成第一个视图
            _, mu1, log_var1 = self.vae_model(x)
            z1 = self.vae_model.reparameterize(mu1, log_var1)
            generated_x1 = self.vae_model.decode(z1)

            # 第二次推理生成第二个视图
            _, mu2, log_var2 = self.vae_model(x)
            z2 = self.vae_model.reparameterize(mu2, log_var2)
            generated_x2 = self.vae_model.decode(z2)

        # 移除批次维度
        x1 = generated_x1.squeeze(0)
        x2 = generated_x2.squeeze(0)

        return x1.cpu(), x2.cpu()


def bbox_mask_generator(inner, outer):
    mask = np.ones([outer]*2)
    zero_index = int((outer-inner)/2)
    mask[zero_index:-zero_index, zero_index:-zero_index] = 0
    #mask[int((outer+1)/2), int((outer+1)/2)] = 1
    mask = mask.astype(bool)
    return mask

def train_padding(img, out_win, r, c, bands):
    radius = int((out_win - 1) / 2)
    out_win = radius
    img_padding = np.zeros((r + 2*out_win, c + 2*out_win, bands))
    img_padding[out_win:r+out_win, out_win:c+out_win, :]=img
    img_padding[0:out_win, out_win:c+out_win, :] = np.flip(img[0:out_win, :, :], 1)    #上边镜像
    img_padding[r+out_win:r+2*out_win, out_win:c+out_win, :] = np.flip(img[r-out_win:r, :, :], 1)#下边镜像
    img_padding[out_win:r+out_win, 0:out_win, :] = np.flip(img[:, 0:out_win, :], 0) #左镜像
    img_padding[out_win:r+out_win, c+out_win:c+2*out_win, :] = np.flip(img[:, c-out_win:c, :], 0)  # 右镜像
    img_padding[0:out_win, 0:out_win, :] = np.flip(img[0:out_win, 0:out_win, :],  (0, 1))  #左上
    img_padding[0:out_win, c+out_win:c+2*out_win, :] = np.flip(img[0:out_win, c-out_win:c, :], (0, 1))#右上
    img_padding[r+out_win:r+2*out_win, 0:out_win, :] = np.flip(img[r-out_win:r, 0:out_win, :], (0, 1))#左下
    img_padding[r+out_win:r+2*out_win, c+out_win:c+2*out_win, :] = np.flip(img[r-out_win:r, c-out_win:c, :], (0, 1))#右下
    return img_padding

def dual_windows_data_processing(img, out_win, in_win):
    r = np.size(img,0)
    c = np.size(img, 1)
    bands = np.size(img, 2)
    img = img.astype(np.float32)
    data_new = []
    mask = bbox_mask_generator(in_win, out_win)
    radius = int((out_win - 1) / 2)
    #img = np.pad(img, ((radius, radius), (radius, radius)), 'reflect')
    img_padding = train_padding(img, out_win, r, c, bands)
    H = img_padding.shape[0]
    W = img_padding.shape[1]

    for i in range(radius, H-radius):
        for j in range(radius, W-radius):
            pixel = img_padding[i, j, None]
            bbox = img_padding[i-radius:i+radius+1, j-radius:j+radius+1]
            bbox = bbox[mask]
            #bbox = bbox[bbox[:, 0] > 0]
            #bbox_new = np.insert(bbox, 0, pixel, axis=0)
            #bbox_new = np.abs(bbox - pixel)
            bbox_new = bbox - pixel
            bbox_new = bbox_new.astype(np.float32)
            data_new.append(bbox_new)
    return data_new