import torch
import torch.nn as nn


class SimSiam(nn.Module):
    def __init__(self, base_encoder, dim=64, pred_dim=32):
        super(SimSiam, self).__init__()

        # 创建编码器
        self.encoder = base_encoder
        '''
        # 构建投影头
        if hasattr(self.encoder, 'fc'):
            # 如果编码器有 'fc' 属性（如 ResNet）
            prev_dim = self.encoder.fc.in_features
            self.encoder.fc = nn.Sequential(
                self.encoder.fc,
                nn.BatchNorm1d(dim, affine=False)
            )
        elif hasattr(self.encoder, 'mlp_head'):
            # 如果编码器有 'mlp_head' 属性（如 ViT）
            prev_dim = self.encoder.mlp_head[-1].in_features
            self.encoder.mlp_head = nn.Sequential(
                *list(self.encoder.mlp_head.children())[:-1],  # 移除最后一个线性层
                nn.BatchNorm1d(dim, affine=False)
            )
        else:
            raise AttributeError("Encoder must have either 'fc' or 'mlp_head' attribute")
        '''
        # 构建预测器
        self.predictor = nn.Sequential(
            nn.Linear(dim, pred_dim, bias=False),
            nn.BatchNorm1d(pred_dim),
            nn.ReLU(inplace=True),
            nn.Linear(pred_dim, dim)
        )

    def forward(self, x1, x2, mask):

        z1 = self.encoder(x1, mask)
        z2 = self.encoder(x2, mask)

        p1 = self.predictor(z1)
        p2 = self.predictor(z2)

        return p1, p2, z1.detach(), z2.detach()