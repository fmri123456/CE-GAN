import math
import load_data
import numpy as np
import torch
import torch.nn as nn
from torch.nn import Parameter
import torch.nn.functional as F
from numpy import linalg as la
# 判别器
class discriminator(nn.Module):
    def __init__(self):
        super(discriminator, self).__init__()
        self.theta = Parameter(torch.randn(70, 70), requires_grad=True)
        self.linear = nn.Sequential(
            nn.Linear(161*70, 128),  # w_1 * x + b_1
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 2),
            nn.Softmax()
        )


    def forward(self,F,W):
        W= W.numpy()
        F =F.numpy()
        hum_count = W.shape[0]
        row = W.shape[1]
        D = np.zeros(shape=(hum_count, row, row))
        V = np.zeros(shape=(hum_count, row, row))
        E = np.ones(shape=(hum_count, row, row))
        A = W + E
        for n in range(hum_count):
            for i in range(row):
                a = A[n]
                a_not_zero = int(np.count_nonzero(a))  # 遍历脑区基因每一行的节点度，即元素非0个数
                D[n][i][i] = a_not_zero
        v, Q = la.eig(D)  # 求D的特征值和特征向量
        for p in range(hum_count):  # 求特征值的对角矩阵
            for q in range(row):
                V[p, q, q] = v[p, q]
        V = np.sqrt(V)
        D_12 = np.linalg.inv((Q * V * Q))  # D的负1/2次幂
        D_12 = torch.Tensor(D_12)
        A = torch.Tensor(A)
        F = torch.Tensor(F)
        F_n = torch.matmul(torch.matmul(torch.matmul(torch.matmul(D_12,A),D_12),F),self.theta)
        m = nn.ReLU()
        F_n = m(F_n)
        F_flatten = torch.flatten(F_n, 1, -1)
        out = self.linear(F_flatten)
        return out
