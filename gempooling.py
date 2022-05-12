# -*- coding: utf-8 -*-
# @Time   : 2021/6/14 - 12:26
# @File   : gem_torch.py
# @Author : surui

import torch
import torch.nn as nn
import torch.nn.functional as F


class GeMPooling(nn.Module):
    def __init__(self, feature_size, pool_size=7, init_norm=3.0, eps=1e-6, normalize=False, **kwargs):
        super(GeMPooling, self).__init__(**kwargs)
        self.feature_size = feature_size  # Final layer channel size, the pow calc at -1 axis
        self.pool_size = pool_size
        self.init_norm = init_norm
        self.p = torch.nn.Parameter(torch.ones(self.feature_size) * self.init_norm, requires_grad=True)
        self.p.data.fill_(init_norm)
        self.normalize = normalize
        self.avg_pooling = nn.AvgPool2d((self.pool_size, self.pool_size))
        self.eps = eps

    def forward(self, features):
        # filter invalid value: set minimum to 1e-6
        # features-> (B, H, W, C)
        features = features.clamp(min=self.eps).pow(self.p)
        features = features.permute((0, 3, 1, 2))
        features = self.avg_pooling(features)
        features = torch.squeeze(features)
        features = features.permute((0, 2, 3, 1))
        features = torch.pow(features, (1.0 / self.p))
        # unit vector
        if self.normalize:
            features = F.normalize(features, dim=-1, p=2)
        return features


if __name__ == '__main__':
    x = torch.randn((8, 7, 7, 768)) * 0.02

    gem = GeMPooling(768, pool_size=3, init_norm=3.0)

    print("input : ", x)
    print("=========================")
    print(gem(x))
