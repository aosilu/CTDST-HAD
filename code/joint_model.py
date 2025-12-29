import torch
import torch.nn as nn

class ChannelAttention(nn.Module):
    def __init__(self, in_channels, reduction=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Sequential(
            nn.Linear(in_channels, in_channels // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(in_channels // reduction, in_channels, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        # x 的形状为 (batch_size, channels)
        # 对于 1D 特征，使用 AdaptiveAvgPool1d 进行全局平均池化
        y = self.avg_pool(x.unsqueeze(-1)).squeeze(-1)
        y = self.fc(y)
        return x * y

class combinedModel(nn.Module):
    def __init__(self, spital_ViT, spectral_ViT, spital_dim, spectral_dim):
        super().__init__()
        self.spital_ViT = spital_ViT
        self.spectral_ViT = spectral_ViT

        # 拼接后的特征维度是两个模型输出维度之和
        self.concat_dim = spital_dim + spectral_dim

        # 添加通道注意力机制
        self.channel_attention = ChannelAttention(self.concat_dim, reduction=16)

        # 使用 sigmoid 适合二分类
        self.classifier = nn.Sequential(
            nn.Linear(self.concat_dim, 1),
            nn.Sigmoid()
        )
        #spital reduce
        self.spital_reduce = nn.Sequential(
            nn.Linear(1024, 256),
            nn.LeakyReLU(),
            nn.Linear(256, 64),
            nn.LeakyReLU(),

        )



    def forward(self, spital_input, spectral_input):
        # 获取两条支线模型的输出特征
        spital_features = self.spital_ViT(spital_input)
        spectral_features = self.spectral_ViT(spectral_input, mask=spectral_input)

        # 拼接特征
        spital_features_expanded = spital_features.expand(spectral_features.size(0), -1)
        spital_features_expanded = self.spital_reduce(spital_features_expanded)
        combined_features = torch.cat((spectral_features, spital_features_expanded), dim=1)

        # 应用通道注意力机制
        #attended_features = self.channel_attention(combined_features)

        # 分类
        output = self.classifier(combined_features)
        return output