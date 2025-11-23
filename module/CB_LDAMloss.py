import os
import torch
import psutil
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torch.cuda


class CB_LDAMLoss(nn.Module):
    def __init__(self, cls_num_list, max_m=0.5, ce_weight=None, s=30, beta=0.999): # s是softmax温度系数，beta是CBloss系数，都是超参数
        super(CB_LDAMLoss, self).__init__()
        self.cls_num_list = cls_num_list
        self.num_classes = len(cls_num_list)
        self.beta = beta

        m_list = 1.0 / np.sqrt(np.sqrt(cls_num_list))
        m_list = m_list * (max_m / np.max(m_list))  # margin item式子中的C
        m_list = torch.cuda.FloatTensor(m_list)
        self.m_list = m_list
        assert s > 0
        self.s = s  # 交叉熵函数的scaling factor(温度系数), 应该属于超参调整范围内
        self.ce_weight = ce_weight
        
        self.beta = 0.993
        effective_num = 1.0 - np.power(self.beta, self.cls_num_list)
        self.class_weights = (1.0 - self.beta) / np.array(effective_num)
        self.class_weights = self.class_weights / np.sum(self.class_weights) * self.num_classes

    def forward(self, x, target):   # x shape: (b, 6), target shape: (b)
        # target = target.view(-1).long()
        index = torch.zeros_like(x, dtype=torch.uint8)
        index.scatter_(1, target.data.view(-1, 1), 1)
        
        index_float = index.type(torch.cuda.FloatTensor)
        batch_m = torch.matmul(self.m_list[None, :], index_float.transpose(0, 1))
        batch_m = batch_m.view((-1, 1))
        x_m = x - batch_m
    
        output = torch.where(index, x_m, x) # (batchsize, 6)

        # Calculate CB Loss
        # 根据 target 获取对应的权重
        weights_for_samples = torch.tensor(self.class_weights).float().to(x.device)[target] # 返回一个与 target 相同形状的权重张量
        cb_ldam_loss = F.cross_entropy(self.s*output, target, reduction='none', weight=self.ce_weight) * weights_for_samples
        cb_ldam_loss = cb_ldam_loss.mean()

        return cb_ldam_loss

    def get_class_weights(self):
        return self.class_weights
    
class LDAMLoss(nn.Module):
    def __init__(self, cls_num_list, max_m=0.5, ce_weight=None, s=30): # s是softmax温度系数，beta是CBloss系数，都是超参数
        '''
        :param cls_num_list : 每个类别样本个数
        :param max_m : LDAM中最大margin参数,default =0.5
        :param weight : 
        :param s : 缩放因子,控制logits的范围
        '''
        super(LDAMLoss, self).__init__()
        m_list = 1.0 / np.sqrt(np.sqrt(cls_num_list)) # n_j 四次方根的倒数
        m_list = m_list * (max_m / np.max(m_list)) # 归一化，C相当于 max_m/ np.max(m_list)，确保没有大于max_m的

        m_list = torch.FloatTensor(m_list).cuda()
        self.m_list = m_list
        assert s > 0
        self.s = s
        self.weight = ce_weight
        
        self.cls_num_list = cls_num_list
        self.num_classes = len(cls_num_list)        

    def forward(self, x, target):   # x shape: (b, 6), target shape: (b)
        # target = target.view(-1).long()
        index = torch.zeros_like(x, dtype=torch.uint8)
        index.scatter_(1, target.data.view(-1, 1), 1)
        
        index_float = index.type(torch.cuda.FloatTensor)
        batch_m = torch.matmul(self.m_list[None, :], index_float.transpose(0, 1))
        batch_m = batch_m.view((-1, 1))
        x_m = x - batch_m
    
        output = torch.where(index, x_m, x) # (batchsize, 6)
        return F.cross_entropy(self.s*output, target, reduction='none', weight=self.weight).mean()


    def get_class_weights(self):
        return None
