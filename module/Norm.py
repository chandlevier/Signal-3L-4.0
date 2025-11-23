import torch
import torch.nn as nn
from torch.nn import Parameter
import torch.nn.functional as F


class NormedLinear(nn.Module):

    def __init__(self, in_features, out_features):  # in_features=256, out_features=64
        super(NormedLinear, self).__init__()
        self.weight = Parameter(torch.Tensor(in_features, out_features))# 构造可训练权重矩阵
        self.weight.data.uniform_(-1, 1).renorm_(2, 1, 1e-5).mul_(1e5)  # 初始化


    def forward(self, x):
        normed_x = F.normalize(x, dim=-1)
        normed_weight = F.normalize(self.weight, dim=0)

        out = torch.matmul(normed_x, normed_weight)
        
        return out